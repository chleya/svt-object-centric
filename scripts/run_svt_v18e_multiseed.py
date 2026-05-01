"""
SVT-v18e: Multi-Seed Robustness Verification

v18-v18d results are promising but based on single seed (42).
v18e verifies robustness across 5 random seeds for:
  1. DualPathwayObjectFile (our model)
  2. SlotAttention + DualPath wrapper (representative published model)

Reports mean +/- std for all key metrics.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.dual_pathway_object_file import DualPathwayObjectFile
from models.slot_attention_model import SetBasedSlotAttentionModel
from models.dual_pathway_wrapper import DualPathwayWrapper
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


def train_dualpath(model, train_data, epochs=60, batch_size=64, lr=1e-3, p_conflict=0.4):
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


def train_wrapped(model, train_data, epochs=60, batch_size=64, lr=1e-3, p_conflict=0.4):
    obs_pos = torch.FloatTensor(train_data["observed_positions"])
    fut_pos = torch.FloatTensor(train_data["future_positions"])
    ids = torch.LongTensor(train_data["identity_labels"])
    obs_feat = torch.FloatTensor(train_data["object_features_obs"]) if "object_features_obs" in train_data else None
    fut_feat = torch.FloatTensor(train_data["object_features_fut"]) if "object_features_fut" in train_data else None

    traj_params = list(model.traj_identity_head.parameters())
    base_params = list(model.base_model.parameters())
    other_params = [p for p in model.parameters() if p not in set(traj_params) and p not in set(base_params)]
    optimizer = torch.optim.Adam([
        {"params": base_params, "lr": lr * 0.5},
        {"params": traj_params, "lr": lr * 2.0},
        {"params": other_params, "lr": lr},
    ])
    n_batches = len(obs_pos) // batch_size
    for epoch in range(epochs):
        if epoch < 10:
            for p in base_params:
                p.requires_grad = False
        else:
            for p in base_params:
                p.requires_grad = True
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


def swap_features(fut_feat):
    if fut_feat is None:
        return None
    swapped = fut_feat.copy()
    if swapped.ndim == 4:
        swapped[:, :, 0, :], swapped[:, :, 1, :] = fut_feat[:, :, 1, :].copy(), fut_feat[:, :, 0, :].copy()
    elif swapped.ndim == 3:
        swapped[:, 0, :], swapped[:, 1, :] = fut_feat[:, 1, :].copy(), fut_feat[:, 0, :].copy()
    return swapped


def eval_full(model, swap_test, clean_test):
    results = {}

    for method in ["combined", "feature_only", "trajectory_only"]:
        try:
            pred = model.predict_identity(
                swap_test["observed_positions"], swap_test.get("object_features_obs"),
                future_positions=swap_test["future_positions"],
                future_features=swap_test.get("object_features_fut"), method=method)
        except TypeError:
            pred = model.predict_identity(
                swap_test["observed_positions"], swap_test.get("object_features_obs"))
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        bd = compute_identity_breakdown(pred, swap_test["identity_labels"])
        results[f"swap_{method}"] = bd["identity_swap_only"]

    fut_feat_conflict = swap_features(swap_test.get("object_features_fut"))
    for method in ["combined", "feature_only", "trajectory_only"]:
        try:
            pred_c = model.predict_identity(
                swap_test["observed_positions"], swap_test.get("object_features_obs"),
                future_positions=swap_test["future_positions"],
                future_features=fut_feat_conflict, method=method)
        except TypeError:
            pred_c = model.predict_identity(
                swap_test["observed_positions"], swap_test.get("object_features_obs"))
        if isinstance(pred_c, torch.Tensor):
            pred_c = pred_c.cpu().numpy()
        bd_c = compute_identity_breakdown(pred_c, swap_test["identity_labels"])
        results[f"conflict_{method}"] = bd_c["identity_swap_only"]

    try:
        pred_clean = model.predict_identity(
            clean_test["observed_positions"], clean_test.get("object_features_obs"),
            future_positions=clean_test["future_positions"],
            future_features=clean_test.get("object_features_fut"), method="combined")
    except TypeError:
        pred_clean = model.predict_identity(
            clean_test["observed_positions"], clean_test.get("object_features_obs"))
    if isinstance(pred_clean, torch.Tensor):
        pred_clean = pred_clean.cpu().numpy()
    bd_clean = compute_identity_breakdown(pred_clean, clean_test["identity_labels"])
    results["clean_combined"] = bd_clean["identity_swap_only"]

    return results


def main():
    seeds = [42, 123, 456, 789, 2024]
    all_results = {"DualPath": [], "SlotAttn_DualPath": []}

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED: {seed}")
        print(f"{'='*70}")

        print("Generating datasets...")
        eval_ds = generate_v3_dataset(n_train=1000, n_test=200, num_objects=2, feature_dim=2,
            feature_mode="feature_bearing", force_train_type="attractor", force_test_type="vortex",
            randomize_object_order=True, disjoint_init_split=True, seed=seed)
        train_data = gen_train(n=1000, seed=seed)
        swap_test = eval_ds["identity_test_swap_only"]
        clean_test = eval_ds["clean_test_id"]

        torch.manual_seed(seed)
        np.random.seed(seed)

        print("\n  Training DualPathwayObjectFile...")
        dp_model = DualPathwayObjectFile(
            num_objects=2, feature_dim=2, hidden_dim=128, slot_dim=64,
            t_obs=10, t_pred=20, conflict_switch_temp=0.1)
        train_dualpath(dp_model, train_data, epochs=60, p_conflict=0.4)
        dp_results = eval_full(dp_model, swap_test, clean_test)
        all_results["DualPath"].append(dp_results)
        print(f"  DualPath: swap={dp_results['swap_combined']:.4f}, conflict={dp_results['conflict_combined']:.4f}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        print("  Training SlotAttention + DualPath...")
        base = SetBasedSlotAttentionModel(
            t_obs=10, t_pred=20, num_objects=2, dim=2,
            feature_dim=2, slot_dim=64, hidden_dim=128)
        sa_wrapped = DualPathwayWrapper(
            base, dim=2, slot_dim=64, hidden_dim=128,
            t_obs=10, num_objects=2, conflict_switch_temp=0.1)
        train_wrapped(sa_wrapped, train_data, epochs=60, p_conflict=0.4)
        sa_results = eval_full(sa_wrapped, swap_test, clean_test)
        all_results["SlotAttn_DualPath"].append(sa_results)
        print(f"  SlotAttn+DP: swap={sa_results['swap_combined']:.4f}, conflict={sa_results['conflict_combined']:.4f}")

    print("\n" + "="*70)
    print("v18e MULTI-SEED ROBUSTNESS RESULTS")
    print("="*70)

    for model_name, results_list in all_results.items():
        print(f"\n  {model_name}:")
        for metric in ["swap_combined", "conflict_combined", "clean_combined",
                        "swap_feature_only", "swap_trajectory_only",
                        "conflict_feature_only", "conflict_trajectory_only"]:
            values = [r[metric] for r in results_list]
            mean = np.mean(values)
            std = np.std(values)
            print(f"    {metric:35s}: {mean:.4f} +/- {std:.4f}")

    print("\n--- Robustness Assessment ---")
    for model_name, results_list in all_results.items():
        conflict_vals = [r["conflict_combined"] for r in results_list]
        swap_vals = [r["swap_combined"] for r in results_list]
        mean_c = np.mean(conflict_vals)
        std_c = np.std(conflict_vals)
        mean_s = np.mean(swap_vals)
        std_s = np.std(swap_vals)
        robust = mean_c > 0.5 and std_c < 0.15
        print(f"  {model_name}: conflict_res={mean_c:.3f}+/-{std_c:.3f}, "
              f"swap={mean_s:.3f}+/-{std_s:.3f}, robust={robust}")


if __name__ == "__main__":
    main()
