"""
SVT-v18: Dual-Pathway ObjectFile with Corrected Conflict Training

v17 critical bug: conflict augmentation swapped features AND identity labels
together, training the model to follow swapped features under conflict.
This is the OPPOSITE of conditional binding.

v18 fix:
  1. Under conflict, identity labels follow TRAJECTORY (not swapped features)
  2. Dual independent scorers: Feature Scorer + Trajectory Scorer
  3. Agreement-based switching: agree -> feature scorer, disagree -> trajectory
  4. Separate training signals for each scorer

This tests whether the corrected training signal enables conditional binding.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.dual_pathway_object_file import DualPathwayObjectFile
from diagnostics.subspace_intervention import SubspaceInterventionTester


def gen_train(n=1000, nobj=2, fdim=2, seed=0):
    rng = np.random.RandomState(seed)
    eps = []
    for _ in range(n):
        ep = _generate_single_episode(t_obs=10, t_pred=20, num_objects=nobj, arena_size=64.0,
            feature_mode="feature_bearing", feature_dim=fdim, randomize_object_order=True,
            identity_test=True, swap_probability=0.5, force_type="attractor",
            field_strength=0.5, damping=0.95, noise_std=0.1, rng=rng)
        eps.append(ep)
    return _stack_episodes(eps, "feature_bearing")


def train_model(model, train_data, epochs=60, batch_size=64, lr=1e-3, p_conflict=0.0):
    obs_pos = torch.FloatTensor(train_data["observed_positions"])
    fut_pos = torch.FloatTensor(train_data["future_positions"])
    ids = torch.LongTensor(train_data["identity_labels"])
    obs_feat = torch.FloatTensor(train_data["object_features_obs"]) if "object_features_obs" in train_data else None
    fut_feat = torch.FloatTensor(train_data["object_features_fut"]) if "object_features_fut" in train_data else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches = len(obs_pos) // batch_size

    for epoch in range(epochs):
        indices = np.random.permutation(len(obs_pos))
        total_loss = 0
        for bi in range(n_batches):
            idx = indices[bi * batch_size:(bi + 1) * batch_size]
            loss, id_loss, smh_loss, cg_loss = model.compute_loss(
                obs_pos[idx], fut_pos[idx], ids[idx],
                observed_features=obs_feat[idx] if obs_feat is not None else None,
                future_features=fut_feat[idx] if fut_feat is not None else None,
                p_conflict=p_conflict)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")


def swap_features(fut_feat):
    if fut_feat is None:
        return None
    swapped = fut_feat.copy()
    if swapped.ndim == 4:
        swapped[:, :, 0, :], swapped[:, :, 1, :] = fut_feat[:, :, 1, :].copy(), fut_feat[:, :, 0, :].copy()
    elif swapped.ndim == 3:
        swapped[:, 0, :], swapped[:, 1, :] = fut_feat[:, 1, :].copy(), fut_feat[:, 0, :].copy()
    return swapped


def main():
    seed = 42
    print("Generating datasets...")
    eval_ds = generate_v3_dataset(n_train=1000, n_test=200, num_objects=2, feature_dim=2,
        feature_mode="feature_bearing", force_train_type="attractor", force_test_type="vortex",
        randomize_object_order=True, disjoint_init_split=True, seed=seed)
    train_data = gen_train(n=1000, seed=seed)
    swap_test = eval_ds["identity_test_swap_only"]
    clean_test = eval_ds["clean_test_id"]

    tester = SubspaceInterventionTester(num_objects=2)

    configs = [
        {"name": "DualPath_pconf0", "p_conflict": 0.0},
        {"name": "DualPath_pconf02", "p_conflict": 0.2},
        {"name": "DualPath_pconf04", "p_conflict": 0.4},
        {"name": "DualPath_pconf06", "p_conflict": 0.6},
    ]

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {cfg['name']}")
        print(f"{'='*70}")

        model = DualPathwayObjectFile(
            num_objects=2, feature_dim=2, hidden_dim=128, slot_dim=64,
            t_obs=10, t_pred=20, identity_weight=1.0, smh_weight=1.0,
            traj_weight=0.1, conflict_switch_temp=0.1)

        train_model(model, train_data, epochs=60, batch_size=64, lr=1e-3,
                    p_conflict=cfg["p_conflict"])

        tester.full_diagnosis(model, cfg["name"], train_data, clean_test, swap_test)

        obs_pos_s = swap_test["observed_positions"]
        obs_feat_s = swap_test.get("object_features_obs")
        fut_pos_s = swap_test["future_positions"]
        fut_feat_s = swap_test.get("object_features_fut")
        true_id_s = swap_test["identity_labels"]

        print("\n  --- Pathway Analysis ---")

        for method_name, method in [("combined", "combined"), ("feature_only", "feature_only"), ("trajectory_only", "trajectory_only")]:
            pred = model.predict_identity(obs_pos_s, obs_feat_s, future_positions=fut_pos_s,
                                          future_features=fut_feat_s, method=method)
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            bd = compute_identity_breakdown(pred, true_id_s)
            print(f"  swap_only ({method_name}): {bd['identity_swap_only']:.4f}")

        fut_feat_conflict = swap_features(fut_feat_s)
        for method_name, method in [("combined", "combined"), ("feature_only", "feature_only"), ("trajectory_only", "trajectory_only")]:
            pred_c = model.predict_identity(obs_pos_s, obs_feat_s, future_positions=fut_pos_s,
                                            future_features=fut_feat_conflict, method=method)
            if isinstance(pred_c, torch.Tensor):
                pred_c = pred_c.cpu().numpy()
            bd_c = compute_identity_breakdown(pred_c, true_id_s)
            print(f"  conflict ({method_name}): {bd_c['identity_swap_only']:.4f}")

        print("\n  --- Dual Score Analysis ---")
        ds_clean = model.get_dual_scores(
            clean_test["observed_positions"], clean_test.get("object_features_obs"),
            clean_test["future_positions"], clean_test.get("object_features_fut"))
        ds_conflict = model.get_dual_scores(
            obs_pos_s, obs_feat_s, fut_pos_s, fut_feat_conflict)

        print(f"  Agreement rate (clean):    {ds_clean['agreement'].mean():.4f}")
        print(f"  Agreement rate (conflict): {ds_conflict['agreement'].mean():.4f}")

        feat_pred_clean = ds_clean['feat_assignment']
        traj_pred_clean = ds_clean['traj_assignment']
        feat_pred_conflict = ds_conflict['feat_assignment']
        traj_pred_conflict = ds_conflict['traj_assignment']

        true_id_clean = clean_test["identity_labels"]
        feat_acc_clean = (feat_pred_clean == true_id_clean).all(axis=1).mean()
        traj_acc_clean = (traj_pred_clean == true_id_clean).all(axis=1).mean()
        feat_acc_conflict = (feat_pred_conflict == true_id_s).all(axis=1).mean()
        traj_acc_conflict = (traj_pred_conflict == true_id_s).all(axis=1).mean()

        print(f"  Feature scorer acc (clean):    {feat_acc_clean:.4f}")
        print(f"  Trajectory scorer acc (clean): {traj_acc_clean:.4f}")
        print(f"  Feature scorer acc (conflict): {feat_acc_conflict:.4f}")
        print(f"  Trajectory scorer acc (conflict): {traj_acc_conflict:.4f}")

        print(f"  Feat scores range (clean): [{ds_clean['feat_scores'].min():.3f}, {ds_clean['feat_scores'].max():.3f}]")
        print(f"  Traj scores range (clean): [{ds_clean['traj_scores'].min():.3f}, {ds_clean['traj_scores'].max():.3f}]")

    tester.print_fingerprint_map()

    print("\n" + "="*70)
    print("v18 KEY ANALYSIS: Does corrected conflict training enable conditional binding?")
    print("="*70)
    for name, r in tester.results.items():
        print(f"  {name}: State={r['state']}, Read={r['readability']:.3f}, Caus={r['causality']:.3f}, Swap={r['swap_accuracy']:.3f}")

    print("\n--- v17 vs v18 Training Signal Comparison ---")
    print("  v17: conflict augmentation swaps features AND labels -> model learns to follow features")
    print("  v18: conflict augmentation swaps features but KEEPS original labels -> model learns to follow trajectory under conflict")
    print("  If v18 achieves higher conflict resolution than v17, the training signal was the bottleneck.")


if __name__ == "__main__":
    main()
