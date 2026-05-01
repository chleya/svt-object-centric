"""
SVT-v19c: Proximity-Enhanced Wrapper for Published Models

v19b showed proximity-enhanced scoring improves close-range accuracy
from 33% to 82% on DualPathwayObjectFile.

v19c tests whether the same improvement applies to published models
when using the proximity-enhanced DualPathwayWrapper.

Compares:
  - SlotAttention + DualPath (v18d baseline, no proximity)
  - SlotAttention + DualPath + Proximity (v19c)
  - RIMs + DualPath (v18d baseline)
  - RIMs + DualPath + Proximity (v19c)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.slot_attention_model import SetBasedSlotAttentionModel
from models.rims_model import RIMsModel
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
        if (epoch + 1) % 30 == 0:
            print(f"    Epoch {epoch+1}/{epochs}")


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
            "name": "SlotAttn_DualPath",
            "base_cls": SetBasedSlotAttentionModel,
            "base_kwargs": {"t_obs": 10, "t_pred": 20, "num_objects": 2, "dim": 2,
                            "feature_dim": 2, "slot_dim": 64, "hidden_dim": 128},
            "use_proximity": False,
        },
        {
            "name": "SlotAttn_DualPath_Prox",
            "base_cls": SetBasedSlotAttentionModel,
            "base_kwargs": {"t_obs": 10, "t_pred": 20, "num_objects": 2, "dim": 2,
                            "feature_dim": 2, "slot_dim": 64, "hidden_dim": 128},
            "use_proximity": True,
        },
        {
            "name": "RIMs_DualPath",
            "base_cls": RIMsModel,
            "base_kwargs": {"t_obs": 10, "t_pred": 20, "num_objects": 2, "dim": 2,
                            "feature_dim": 2, "num_rims": 2, "rim_dim": 64, "hidden_dim": 128},
            "use_proximity": False,
        },
        {
            "name": "RIMs_DualPath_Prox",
            "base_cls": RIMsModel,
            "base_kwargs": {"t_obs": 10, "t_pred": 20, "num_objects": 2, "dim": 2,
                            "feature_dim": 2, "num_rims": 2, "rim_dim": 64, "hidden_dim": 128},
            "use_proximity": True,
        },
    ]

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {cfg['name']}")
        print(f"{'='*70}")

        base_model = cfg["base_cls"](**cfg["base_kwargs"])
        wrapped = DualPathwayWrapper(
            base_model, dim=2, slot_dim=64, hidden_dim=128,
            t_obs=10, num_objects=2, conflict_switch_temp=0.1,
            use_proximity=cfg["use_proximity"])
        train_wrapped(wrapped, train_data, epochs=60, p_conflict=0.4)

        tester.full_diagnosis(wrapped, cfg["name"], train_data, clean_test, swap_test)

        obs_pos_s = swap_test["observed_positions"]
        obs_feat_s = swap_test.get("object_features_obs")
        fut_pos_s = swap_test["future_positions"]
        fut_feat_s = swap_test.get("object_features_fut")
        true_id_s = swap_test["identity_labels"]

        for method in ["combined", "feature_only", "trajectory_only"]:
            try:
                pred = wrapped.predict_identity(
                    obs_pos_s, obs_feat_s, future_positions=fut_pos_s,
                    future_features=fut_feat_s, method=method)
            except TypeError:
                pred = wrapped.predict_identity(obs_pos_s, obs_feat_s)
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            bd = compute_identity_breakdown(pred, true_id_s)
            print(f"  swap ({method}): {bd['identity_swap_only']:.4f}")

        fut_feat_conflict = swap_features(fut_feat_s)
        for method in ["combined", "feature_only", "trajectory_only"]:
            try:
                pred_c = wrapped.predict_identity(
                    obs_pos_s, obs_feat_s, future_positions=fut_pos_s,
                    future_features=fut_feat_conflict, method=method)
            except TypeError:
                pred_c = wrapped.predict_identity(obs_pos_s, obs_feat_s)
            if isinstance(pred_c, torch.Tensor):
                pred_c = pred_c.cpu().numpy()
            bd_c = compute_identity_breakdown(pred_c, true_id_s)
            print(f"  conflict ({method}): {bd_c['identity_swap_only']:.4f}")

    tester.print_fingerprint_map()

    print("\n" + "="*70)
    print("v19c ANALYSIS: Does proximity-enhanced scoring help published models?")
    print("="*70)
    for name, r in tester.results.items():
        print(f"  {name}: State={r['state']}, Read={r['readability']:.3f}, "
              f"Caus={r['causality']:.3f}, Swap={r['swap_accuracy']:.3f}")


if __name__ == "__main__":
    main()
