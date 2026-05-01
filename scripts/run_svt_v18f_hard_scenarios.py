"""
SVT-v18f: Harder Scenarios — 3 Objects and Occlusion

v18-v18e validated dual-pathway on 2-object scenarios.
v18f tests whether the principle scales to harder settings:
  1. 3 objects (exponentially more assignment permutations)
  2. Occlusion (objects disappear during observation)
  3. 3 objects + occlusion combined
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


def gen_train_3obj(n=800, fdim=2, seed=0):
    rng = np.random.RandomState(seed)
    eps = []
    for _ in range(n):
        ep = _generate_single_episode(t_obs=10, t_pred=20, num_objects=3, arena_size=80.0,
            feature_mode="feature_bearing", feature_dim=fdim, randomize_object_order=True,
            identity_test=True, swap_probability=0.5, force_type="attractor",
            field_strength=0.5, damping=0.95, noise_std=0.1, rng=rng)
        eps.append(ep)
    return _stack_episodes(eps, "feature_bearing")


def add_occlusion(data, p_occlude=0.3, seed=0):
    rng = np.random.RandomState(seed)
    obs_pos = data["observed_positions"].copy()
    obs_feat = data["object_features_obs"].copy() if "object_features_obs" in data else None
    N = obs_pos.shape[2]
    t_obs = obs_pos.shape[1]

    for b in range(len(obs_pos)):
        for j in range(N):
            if rng.rand() < p_occlude:
                start = rng.randint(3, t_obs - 2)
                obs_pos[b, start:, j, :] = obs_pos[b, start, j, :]
                if obs_feat is not None:
                    if obs_feat.ndim == 4:
                        obs_feat[b, start:, j, :] = 0.0
                    elif obs_feat.ndim == 3:
                        obs_feat[b, j, :] = 0.0

    result = dict(data)
    result["observed_positions"] = obs_pos
    if obs_feat is not None:
        result["object_features_obs"] = obs_feat
    return result


def train_model(model, train_data, epochs=60, batch_size=64, lr=1e-3, p_conflict=0.4):
    obs_pos = torch.FloatTensor(train_data["observed_positions"])
    fut_pos = torch.FloatTensor(train_data["future_positions"])
    ids = torch.LongTensor(train_data["identity_labels"])
    obs_feat = torch.FloatTensor(train_data["object_features_obs"]) if "object_features_obs" in train_data else None
    fut_feat = torch.FloatTensor(train_data["object_features_fut"]) if "object_features_fut" in train_data else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches = len(obs_pos) // batch_size
    for epoch in range(epochs):
        indices = np.random.permutation(len(obs_pos))
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
        if (epoch + 1) % 30 == 0:
            print(f"    Epoch {epoch+1}/{epochs}")


def swap_features_n(fut_feat, n_swap=2):
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

    eval_ds_2obj = generate_v3_dataset(n_train=1000, n_test=200, num_objects=2, feature_dim=2,
        feature_mode="feature_bearing", force_train_type="attractor", force_test_type="vortex",
        randomize_object_order=True, disjoint_init_split=True, seed=seed)
    train_2obj = gen_train(n=1000, seed=seed)
    swap_2obj = eval_ds_2obj["identity_test_swap_only"]
    clean_2obj = eval_ds_2obj["clean_test_id"]

    train_3obj = gen_train_3obj(n=800, seed=seed)

    swap_2obj_occluded = add_occlusion(swap_2obj, p_occlude=0.3, seed=seed)
    clean_2obj_occluded = add_occlusion(clean_2obj, p_occlude=0.3, seed=seed)

    configs = [
        {
            "name": "2obj_clean",
            "model_kwargs": {"num_objects": 2, "feature_dim": 2, "hidden_dim": 128,
                             "slot_dim": 64, "t_obs": 10, "t_pred": 20, "conflict_switch_temp": 0.1},
            "train_data": train_2obj,
            "swap_test": swap_2obj,
            "clean_test": clean_2obj,
        },
        {
            "name": "2obj_occluded",
            "model_kwargs": {"num_objects": 2, "feature_dim": 2, "hidden_dim": 128,
                             "slot_dim": 64, "t_obs": 10, "t_pred": 20, "conflict_switch_temp": 0.1},
            "train_data": train_2obj,
            "swap_test": swap_2obj_occluded,
            "clean_test": clean_2obj_occluded,
        },
        {
            "name": "3obj",
            "model_kwargs": {"num_objects": 3, "feature_dim": 2, "hidden_dim": 128,
                             "slot_dim": 64, "t_obs": 10, "t_pred": 20, "conflict_switch_temp": 0.1},
            "train_data": train_3obj,
            "swap_test": None,
            "clean_test": None,
        },
    ]

    results = {}

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {cfg['name']}")
        print(f"{'='*70}")

        model = DualPathwayObjectFile(**cfg["model_kwargs"])
        train_model(model, cfg["train_data"], epochs=60, p_conflict=0.4)

        swap_test = cfg["swap_test"]
        clean_test = cfg["clean_test"]

        if swap_test is not None:
            true_id = swap_test["identity_labels"]
            for method in ["combined", "feature_only", "trajectory_only"]:
                pred = model.predict_identity(
                    swap_test["observed_positions"], swap_test.get("object_features_obs"),
                    future_positions=swap_test["future_positions"],
                    future_features=swap_test.get("object_features_fut"), method=method)
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
                bd = compute_identity_breakdown(pred, true_id)
                print(f"  swap ({method}): {bd['identity_swap_only']:.4f}")

            fut_feat_conflict = swap_features_n(swap_test.get("object_features_fut"))
            for method in ["combined", "feature_only", "trajectory_only"]:
                pred_c = model.predict_identity(
                    swap_test["observed_positions"], swap_test.get("object_features_obs"),
                    future_positions=swap_test["future_positions"],
                    future_features=fut_feat_conflict, method=method)
                if isinstance(pred_c, torch.Tensor):
                    pred_c = pred_c.cpu().numpy()
                bd_c = compute_identity_breakdown(pred_c, true_id)
                print(f"  conflict ({method}): {bd_c['identity_swap_only']:.4f}")

            if hasattr(model, 'get_dual_scores'):
                ds = model.get_dual_scores(
                    clean_test["observed_positions"], clean_test.get("object_features_obs"),
                    clean_test["future_positions"], clean_test.get("object_features_fut"))
                print(f"  Agreement rate (clean): {ds['agreement'].mean():.4f}")

                ds_c = model.get_dual_scores(
                    swap_test["observed_positions"], swap_test.get("object_features_obs"),
                    swap_test["future_positions"], fut_feat_conflict)
                print(f"  Agreement rate (conflict): {ds_c['agreement'].mean():.4f}")

                feat_pred = ds['feat_assignment']
                traj_pred = ds['traj_assignment']
                true_id_clean = clean_test["identity_labels"]
                feat_acc = (feat_pred == true_id_clean).all(axis=1).mean()
                traj_acc = (traj_pred == true_id_clean).all(axis=1).mean()
                print(f"  Feature scorer acc (clean): {feat_acc:.4f}")
                print(f"  Trajectory scorer acc (clean): {traj_acc:.4f}")
        else:
            print("  (3-object: direct evaluation only)")
            train_ids = cfg["train_data"]["identity_labels"]
            pred = model.predict_identity(
                cfg["train_data"]["observed_positions"],
                cfg["train_data"].get("object_features_obs"),
                future_positions=cfg["train_data"]["future_positions"],
                future_features=cfg["train_data"].get("object_features_fut"),
                method="combined")
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            bd = compute_identity_breakdown(pred, train_ids)
            print(f"  train combined: {bd['identity_swap_only']:.4f}")

            pred_feat = model.predict_identity(
                cfg["train_data"]["observed_positions"],
                cfg["train_data"].get("object_features_obs"),
                future_positions=cfg["train_data"]["future_positions"],
                future_features=cfg["train_data"].get("object_features_fut"),
                method="feature_only")
            if isinstance(pred_feat, torch.Tensor):
                pred_feat = pred_feat.cpu().numpy()
            bd_feat = compute_identity_breakdown(pred_feat, train_ids)
            print(f"  train feature_only: {bd_feat['identity_swap_only']:.4f}")

            pred_traj = model.predict_identity(
                cfg["train_data"]["observed_positions"],
                cfg["train_data"].get("object_features_obs"),
                future_positions=cfg["train_data"]["future_positions"],
                future_features=cfg["train_data"].get("object_features_fut"),
                method="trajectory_only")
            if isinstance(pred_traj, torch.Tensor):
                pred_traj = pred_traj.cpu().numpy()
            bd_traj = compute_identity_breakdown(pred_traj, train_ids)
            print(f"  train trajectory_only: {bd_traj['identity_swap_only']:.4f}")

    print("\n" + "="*70)
    print("v18f ANALYSIS: Does dual-pathway scale to harder scenarios?")
    print("="*70)
    print("  Key questions:")
    print("  1. Does occlusion degrade trajectory scorer more than feature scorer?")
    print("  2. Does 3-object scenario maintain conditional binding?")
    print("  3. Is agreement-based switching still reliable under harder conditions?")


if __name__ == "__main__":
    main()
