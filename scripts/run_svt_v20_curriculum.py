"""
SVT-v20: Proximity Curriculum Learning

v19b proximity-enhanced scoring: close-range 33% -> 82%.
But 82% is still far from the 100% ceiling.

v20 hypothesis: The model struggles with close-range scenarios because
they're underrepresented in random training data. Curriculum learning
that progressively introduces close-range examples should help.

Approach:
  Phase 1 (epochs 1-20): Only samples with min_dist > 30 (easy)
  Phase 2 (epochs 21-40): Samples with min_dist > 15 (medium)
  Phase 3 (epochs 41-60): All samples including min_dist < 15 (hard)
  Phase 4 (epochs 61-80): All samples with extra close-range augmentation

Also tests: close-range data augmentation (generate extra samples
where objects start close together).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.dual_pathway_object_file import DualPathwayObjectFile
from diagnostics.subspace_intervention import SubspaceInterventionTester


def gen_train(n=1000, nobj=2, fdim=2, seed=0, min_dist_threshold=0):
    rng = np.random.RandomState(seed)
    eps = []
    attempts = 0
    while len(eps) < n and attempts < n * 10:
        attempts += 1
        ep = _generate_single_episode(t_obs=10, t_pred=20, num_objects=nobj, arena_size=64.0,
            feature_mode="feature_bearing", feature_dim=fdim, randomize_object_order=True,
            identity_test=True, swap_probability=0.5, force_type="attractor",
            field_strength=0.5, damping=0.95, noise_std=0.1, rng=rng)
        obs_pos = ep["observed_positions"]
        if nobj >= 2:
            dists = np.sqrt(((obs_pos[:, 0, :] - obs_pos[:, 1, :]) ** 2).sum(axis=-1))
            min_d = dists.min()
        else:
            min_d = 100.0
        if min_d >= min_dist_threshold:
            eps.append(ep)
    return _stack_episodes(eps, "feature_bearing")


def gen_close_range_augmented(n=200, nobj=2, fdim=2, seed=0):
    rng = np.random.RandomState(seed)
    eps = []
    attempts = 0
    while len(eps) < n and attempts < n * 20:
        attempts += 1
        ep = _generate_single_episode(t_obs=10, t_pred=20, num_objects=nobj, arena_size=20.0,
            feature_mode="feature_bearing", feature_dim=fdim, randomize_object_order=True,
            identity_test=True, swap_probability=0.5, force_type="attractor",
            field_strength=0.5, damping=0.95, noise_std=0.1, rng=rng)
        eps.append(ep)
    return _stack_episodes(eps, "feature_bearing")


def compute_min_distances(positions):
    B = positions.shape[0]
    N = positions.shape[2]
    min_dists = np.zeros(B)
    for b in range(B):
        md = 1e6
        for j in range(N):
            for k in range(j+1, N):
                d = np.sqrt(((positions[b, :, j, :] - positions[b, :, k, :]) ** 2).sum(axis=-1)).min()
                md = min(md, d)
        min_dists[b] = md
    return min_dists


def train_curriculum(model, train_data, close_aug_data, epochs=80, batch_size=64, lr=1e-3, p_conflict=0.4):
    obs_pos = torch.FloatTensor(train_data["observed_positions"])
    fut_pos = torch.FloatTensor(train_data["future_positions"])
    ids = torch.LongTensor(train_data["identity_labels"])
    obs_feat = torch.FloatTensor(train_data["object_features_obs"]) if "object_features_obs" in train_data else None
    fut_feat = torch.FloatTensor(train_data["object_features_fut"]) if "object_features_fut" in train_data else None

    min_dists = compute_min_distances(train_data["observed_positions"])

    if close_aug_data is not None:
        aug_obs_pos = torch.FloatTensor(close_aug_data["observed_positions"])
        aug_fut_pos = torch.FloatTensor(close_aug_data["future_positions"])
        aug_ids = torch.LongTensor(close_aug_data["identity_labels"])
        aug_obs_feat = torch.FloatTensor(close_aug_data["object_features_obs"]) if "object_features_obs" in close_aug_data else None
        aug_fut_feat = torch.FloatTensor(close_aug_data["object_features_fut"]) if "object_features_fut" in close_aug_data else None

    feat_params = list(model.feature_scorer.parameters()) + list(model.obs_feat_encoder.parameters()) + list(model.fut_feat_encoder.parameters())
    traj_params = list(model.trajectory_scorer.parameters()) + list(model.obs_traj_encoder.parameters()) + list(model.fut_traj_encoder.parameters())
    other_params = [p for p in model.parameters() if p not in set(feat_params) and p not in set(traj_params)]

    optimizer = torch.optim.Adam([
        {"params": feat_params, "lr": lr * 0.5},
        {"params": traj_params, "lr": lr * 2.0},
        {"params": other_params, "lr": lr},
    ], weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        if epoch < 15:
            for p in feat_params:
                p.requires_grad = False
        else:
            for p in feat_params:
                p.requires_grad = True

        if epoch < 20:
            mask = min_dists >= 30
        elif epoch < 40:
            mask = min_dists >= 15
        else:
            mask = np.ones(len(min_dists), dtype=bool)

        if mask.sum() < batch_size:
            mask = np.ones(len(min_dists), dtype=bool)

        cur_obs_pos = obs_pos[mask]
        cur_fut_pos = fut_pos[mask]
        cur_ids = ids[mask]
        cur_obs_feat = obs_feat[mask] if obs_feat is not None else None
        cur_fut_feat = fut_feat[mask] if fut_feat is not None else None

        if close_aug_data is not None and epoch >= 40:
            n_aug = min(len(aug_obs_pos), batch_size)
            cur_obs_pos = torch.cat([cur_obs_pos, aug_obs_pos[:n_aug]], dim=0)
            cur_fut_pos = torch.cat([cur_fut_pos, aug_fut_pos[:n_aug]], dim=0)
            cur_ids = torch.cat([cur_ids, aug_ids[:n_aug]], dim=0)
            if cur_obs_feat is not None and aug_obs_feat is not None:
                cur_obs_feat = torch.cat([cur_obs_feat, aug_obs_feat[:n_aug]], dim=0)
            if cur_fut_feat is not None and aug_fut_feat is not None:
                cur_fut_feat = torch.cat([cur_fut_feat, aug_fut_feat[:n_aug]], dim=0)

        n_batches = len(cur_obs_pos) // batch_size
        if n_batches == 0:
            n_batches = 1

        indices = np.random.permutation(len(cur_obs_pos))
        total_loss = 0
        for bi in range(n_batches):
            idx = indices[bi * batch_size:(bi + 1) * batch_size]
            if len(idx) < 2:
                continue
            loss, _, _, _ = model.compute_loss(
                cur_obs_pos[idx], cur_fut_pos[idx], cur_ids[idx],
                observed_features=cur_obs_feat[idx] if cur_obs_feat is not None else None,
                future_features=cur_fut_feat[idx] if cur_fut_feat is not None else None,
                p_conflict=p_conflict)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 20 == 0:
            phase = "easy" if epoch < 20 else ("medium" if epoch < 40 else "hard+aug")
            print(f"    Epoch {epoch+1}/{epochs} [{phase}], Loss: {total_loss/n_batches:.4f}")


def train_baseline(model, train_data, epochs=80, batch_size=64, lr=1e-3, p_conflict=0.4):
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
    close_aug = gen_close_range_augmented(n=200, seed=seed+100)
    swap_test = eval_ds["identity_test_swap_only"]
    clean_test = eval_ds["clean_test_id"]

    tester = SubspaceInterventionTester(num_objects=2)

    configs = [
        {
            "name": "DualPath_baseline",
            "model_kwargs": {"num_objects": 2, "feature_dim": 2, "hidden_dim": 128,
                             "slot_dim": 64, "t_obs": 10, "t_pred": 20, "conflict_switch_temp": 0.1},
            "curriculum": False,
        },
        {
            "name": "DualPath_curriculum",
            "model_kwargs": {"num_objects": 2, "feature_dim": 2, "hidden_dim": 128,
                             "slot_dim": 64, "t_obs": 10, "t_pred": 20, "conflict_switch_temp": 0.1},
            "curriculum": True,
        },
    ]

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {cfg['name']}")
        print(f"{'='*70}")

        model = DualPathwayObjectFile(**cfg["model_kwargs"])

        if cfg["curriculum"]:
            train_curriculum(model, train_data, close_aug, epochs=80, p_conflict=0.4)
        else:
            train_baseline(model, train_data, epochs=80, p_conflict=0.4)

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

        print("\n  --- Proximity-Stratified Analysis ---")
        ds = model.get_dual_scores(
            clean_test["observed_positions"], clean_test.get("object_features_obs"),
            clean_test["future_positions"], clean_test.get("object_features_fut"))

        traj_pred = ds['traj_assignment']
        true_id_clean = clean_test["identity_labels"]
        traj_correct = (traj_pred == true_id_clean).all(axis=1)

        obs_pos_clean = clean_test["observed_positions"]
        min_dists = compute_min_distances(obs_pos_clean)

        dist_bins = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100)]
        for lo, hi in dist_bins:
            mask = (min_dists >= lo) & (min_dists < hi)
            if mask.sum() > 0:
                acc = traj_correct[mask].mean()
                print(f"    min_dist [{lo:3d}, {hi:3d}): n={mask.sum():4d}, traj_acc={acc:.4f}")

    tester.print_fingerprint_map()

    print("\n" + "="*70)
    print("v20 ANALYSIS: Does curriculum learning improve close-range accuracy?")
    print("="*70)
    for name, r in tester.results.items():
        print(f"  {name}: State={r['state']}, Read={r['readability']:.3f}, "
              f"Caus={r['causality']:.3f}, Swap={r['swap_accuracy']:.3f}")


if __name__ == "__main__":
    main()
