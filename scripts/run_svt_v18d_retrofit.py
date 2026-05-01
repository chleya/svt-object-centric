"""
SVT-v18d: Dual-Pathway Retrofit for Published Object-Centric Models

v18 showed DualPathwayObjectFile achieves conditional binding (0.912).
v18d tests whether the same principle can be RETROFITTED to published models.

Approach: Wrap published models with a trajectory identity head and
agreement-based switching. This tests whether the dual-pathway
principle is general or specific to our architecture.

Models tested:
  1. Slot Attention (Locatello et al., 2020)
  2. RIMs (Goyal et al., 2021)
  3. SAVi (Kipf et al., 2022)
  4. DINOSAUR (Seitzer et al., 2024)

Each model is tested:
  a) Original (feature-only identity) — expected: feature-reader profile
  b) With dual-pathway wrapper — expected: conditional binding if principle is general
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.slot_attention_model import SetBasedSlotAttentionModel
from models.rims_model import RIMsModel
from models.savi_model import SAViModel
from models.dinosaur_model import DINOSAURModel
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


def train_base_model(model, train_data, epochs=40, batch_size=64, lr=1e-3):
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
                future_features=fut_feat[idx] if fut_feat is not None else None)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")


def train_wrapped_model(model, train_data, epochs=60, batch_size=64, lr=1e-3, p_conflict=0.4):
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
        indices = np.random.permutation(len(obs_pos))
        total_loss = 0

        if epoch < 10:
            for p in base_params:
                p.requires_grad = False
        else:
            for p in base_params:
                p.requires_grad = True

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


def eval_model(model, swap_test, method="combined"):
    obs_pos_s = swap_test["observed_positions"]
    obs_feat_s = swap_test.get("object_features_obs")
    fut_pos_s = swap_test["future_positions"]
    fut_feat_s = swap_test.get("object_features_fut")
    true_id_s = swap_test["identity_labels"]

    try:
        pred = model.predict_identity(obs_pos_s, obs_feat_s, future_positions=fut_pos_s,
                                       future_features=fut_feat_s, method=method)
    except TypeError:
        try:
            pred = model.predict_identity(obs_pos_s, obs_feat_s,
                                           future_features=fut_feat_s)
        except TypeError:
            pred = model.predict_identity(obs_pos_s, obs_feat_s)
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    bd = compute_identity_breakdown(pred, true_id_s)

    fut_feat_conflict = swap_features(fut_feat_s)
    try:
        pred_c = model.predict_identity(obs_pos_s, obs_feat_s, future_positions=fut_pos_s,
                                         future_features=fut_feat_conflict, method=method)
    except TypeError:
        try:
            pred_c = model.predict_identity(obs_pos_s, obs_feat_s,
                                             future_features=fut_feat_conflict)
        except TypeError:
            pred_c = model.predict_identity(obs_pos_s, obs_feat_s)
    if isinstance(pred_c, torch.Tensor):
        pred_c = pred_c.cpu().numpy()
    bd_c = compute_identity_breakdown(pred_c, true_id_s)

    return bd["identity_swap_only"], bd_c["identity_swap_only"]


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

    model_configs = [
        {
            "name": "SlotAttention",
            "base_cls": SetBasedSlotAttentionModel,
            "base_kwargs": {"t_obs": 10, "t_pred": 20, "num_objects": 2, "dim": 2,
                            "feature_dim": 2, "slot_dim": 64, "hidden_dim": 128},
        },
        {
            "name": "RIMs",
            "base_cls": RIMsModel,
            "base_kwargs": {"t_obs": 10, "t_pred": 20, "num_objects": 2, "dim": 2,
                            "feature_dim": 2, "num_rims": 2, "rim_dim": 64, "hidden_dim": 128},
        },
        {
            "name": "SAVi",
            "base_cls": SAViModel,
            "base_kwargs": {"t_obs": 10, "t_pred": 20, "num_objects": 2, "dim": 2,
                            "feature_dim": 2, "slot_dim": 64, "hidden_dim": 128},
        },
        {
            "name": "DINOSAUR",
            "base_cls": DINOSAURModel,
            "base_kwargs": {"t_obs": 10, "t_pred": 20, "num_objects": 2, "dim": 2,
                            "feature_dim": 2, "slot_dim": 64, "hidden_dim": 128},
        },
    ]

    for cfg in model_configs:
        print(f"\n{'='*70}")
        print(f"MODEL: {cfg['name']} — Original (feature-only)")
        print(f"{'='*70}")

        base_model = cfg["base_cls"](**cfg["base_kwargs"])
        train_base_model(base_model, train_data, epochs=40)

        swap_normal, swap_conflict = eval_model(base_model, swap_test)
        print(f"  Original: swap={swap_normal:.4f}, conflict={swap_conflict:.4f}")

        print(f"\n{'='*70}")
        print(f"MODEL: {cfg['name']} — Dual-Pathway Wrapper")
        print(f"{'='*70}")

        base_model2 = cfg["base_cls"](**cfg["base_kwargs"])
        wrapped = DualPathwayWrapper(
            base_model2, dim=2, slot_dim=64, hidden_dim=128,
            t_obs=10, num_objects=2, conflict_switch_temp=0.1)
        train_wrapped_model(wrapped, train_data, epochs=60, p_conflict=0.4)

        tester.full_diagnosis(wrapped, f"{cfg['name']}_DualPath", train_data, clean_test, swap_test)

        for method in ["combined", "feature_only", "trajectory_only"]:
            swap_n, swap_c = eval_model(wrapped, swap_test, method=method)
            print(f"  {method}: swap={swap_n:.4f}, conflict={swap_c:.4f}")

    tester.print_fingerprint_map()

    print("\n" + "="*70)
    print("v18d ANALYSIS: Does dual-pathway retrofit fix published models?")
    print("="*70)
    for name, r in tester.results.items():
        print(f"  {name}: State={r['state']}, Read={r['readability']:.3f}, "
              f"Caus={r['causality']:.3f}, Swap={r['swap_accuracy']:.3f}")

    print("\n--- Key Question ---")
    print("  If wrapped models achieve conditional binding:")
    print("    -> Dual-pathway principle is GENERAL and can retrofit any model")
    print("  If wrapped models still show feature-reader profile:")
    print("    -> The base model's feature dominance is too strong to overcome")


if __name__ == "__main__":
    main()
