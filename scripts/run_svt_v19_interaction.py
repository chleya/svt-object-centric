"""
SVT-v19: Interaction-Aware Dual-Pathway ObjectFile

v18h showed object proximity (r=0.44) is the #1 failure mode.
v19 adds interaction-aware trajectory encoding:
  1. Cross-object attention in trajectory encoder
  2. Relative position encoding (distance + direction)
  3. Proximity gating (switch between independent/joint encoding)
  4. Force prediction auxiliary loss

Compares:
  - v18 DualPathwayObjectFile (baseline, independent encoding)
  - v19 InteractionAwareDualPathway (interaction-aware encoding)
  - Proximity-stratified analysis to verify improvement
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.dual_pathway_object_file import DualPathwayObjectFile
from models.interaction_aware_dual_pathway import InteractionAwareDualPathway
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


def train_model(model, train_data, epochs=80, batch_size=64, lr=1e-3, p_conflict=0.4):
    obs_pos = torch.FloatTensor(train_data["observed_positions"])
    fut_pos = torch.FloatTensor(train_data["future_positions"])
    ids = torch.LongTensor(train_data["identity_labels"])
    obs_feat = torch.FloatTensor(train_data["object_features_obs"]) if "object_features_obs" in train_data else None
    fut_feat = torch.FloatTensor(train_data["object_features_fut"]) if "object_features_fut" in train_data else None

    traj_params = []
    feat_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'traj_encoder' in name or 'trajectory_scorer' in name or 'force_predictor' in name:
            traj_params.append(param)
        elif 'feat_encoder' in name or 'feature_scorer' in name:
            feat_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": feat_params, "lr": lr * 0.5},
        {"params": traj_params, "lr": lr * 2.0},
        {"params": other_params, "lr": lr},
    ], weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n_batches = len(obs_pos) // batch_size

    for epoch in range(epochs):
        if epoch < 15:
            for p in feat_params:
                p.requires_grad = False
        else:
            for p in feat_params:
                p.requires_grad = True

        indices = np.random.permutation(len(obs_pos))
        total_loss = 0
        for bi in range(n_batches):
            idx = indices[bi * batch_size:(bi + 1) * batch_size]
            loss, _, _, _ = model.compute_loss(
                obs_pos[idx], fut_pos[idx], ids[idx],
                observed_features=obs_feat[idx] if obs_feat is not None else None,
                future_features=fut_feat[idx] if fut_feat is not None else None,
                p_conflict=p_conflict)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 20 == 0:
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
        {
            "name": "DualPath_v18_baseline",
            "model_cls": DualPathwayObjectFile,
            "model_kwargs": {"num_objects": 2, "feature_dim": 2, "hidden_dim": 128,
                             "slot_dim": 64, "t_obs": 10, "t_pred": 20, "conflict_switch_temp": 0.1},
        },
        {
            "name": "InteractionAware_v19",
            "model_cls": InteractionAwareDualPathway,
            "model_kwargs": {"num_objects": 2, "feature_dim": 2, "hidden_dim": 128,
                             "slot_dim": 64, "t_obs": 10, "t_pred": 20,
                             "force_weight": 0.1, "conflict_switch_temp": 0.1},
        },
    ]

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {cfg['name']}")
        print(f"{'='*70}")

        model = cfg["model_cls"](**cfg["model_kwargs"])
        train_model(model, train_data, epochs=80, p_conflict=0.4)

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
        ds = model.get_dual_scores(
            clean_test["observed_positions"], clean_test.get("object_features_obs"),
            clean_test["future_positions"], clean_test.get("object_features_fut"))
        print(f"  Agreement rate (clean): {ds['agreement'].mean():.4f}")

        ds_c = model.get_dual_scores(
            obs_pos_s, obs_feat_s, fut_pos_s, fut_feat_conflict)
        print(f"  Agreement rate (conflict): {ds_c['agreement'].mean():.4f}")

        traj_pred = ds['traj_assignment']
        true_id_clean = clean_test["identity_labels"]
        traj_acc = (traj_pred == true_id_clean).all(axis=1).mean()
        print(f"  Trajectory scorer acc (clean): {traj_acc:.4f}")

        traj_pred_s = ds_c['traj_assignment']
        traj_acc_s = (traj_pred_s == true_id_s).all(axis=1).mean()
        print(f"  Trajectory scorer acc (conflict): {traj_acc_s:.4f}")

        if 'min_distances' in ds:
            min_dists = ds['min_distances']
            traj_correct = (traj_pred == true_id_clean).all(axis=1)

            print("\n  --- Proximity-Stratified Trajectory Accuracy ---")
            dist_bins = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100)]
            for lo, hi in dist_bins:
                mask = (min_dists.min(axis=1) >= lo) & (min_dists.min(axis=1) < hi)
                if mask.sum() > 0:
                    acc = traj_correct[mask].mean()
                    print(f"    min_dist [{lo:3d}, {hi:3d}): n={mask.sum():4d}, traj_acc={acc:.4f}")

    tester.print_fingerprint_map()

    print("\n" + "="*70)
    print("v19 ANALYSIS: Does interaction-aware encoding improve trajectory scoring?")
    print("="*70)
    for name, r in tester.results.items():
        print(f"  {name}: State={r['state']}, Read={r['readability']:.3f}, "
              f"Caus={r['causality']:.3f}, Swap={r['swap_accuracy']:.3f}")


if __name__ == "__main__":
    main()
