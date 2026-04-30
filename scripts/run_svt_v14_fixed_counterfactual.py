"""
SVT-v14: Fixed Counterfactual Training + Subspace Intervention Diagnosis

v13 bug: counterfactual pressures were applied to ALL samples including swap episodes,
creating conflicting gradients that destroyed identity encoding.

v14 fix: only apply counterfactual pressures on CLEAN (non-swap) episodes.
This matches the Neural Stage finding that counterfactual training is the
strongest pressure for relation internalization, but only when the signal is clean.

After training, run subspace intervention diagnosis to check:
  - Is identity now readable from the representation? (State A vs B/C/D)
  - Is identity causally used? (State C vs D)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.counterfactual_object_file_v2 import CounterfactualObjectFileV2
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


def train_model(model, train_data, epochs=40, batch_size=64, lr=1e-3):
    obs_pos = torch.FloatTensor(train_data["observed_positions"])
    fut_pos = torch.FloatTensor(train_data["future_positions"])
    ids = torch.LongTensor(train_data["identity_labels"])
    obs_feat = torch.FloatTensor(train_data["object_features_obs"]) if "object_features_obs" in train_data else None
    fut_feat = torch.FloatTensor(train_data["object_features_fut"]) if "object_features_fut" in train_data else None

    N = model.num_objects
    is_swap = torch.zeros(ids.shape[0], dtype=torch.bool)
    for b in range(ids.shape[0]):
        for j in range(N):
            if ids[b, j] != j:
                is_swap[b] = True
                break

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches = len(obs_pos) // batch_size

    for epoch in range(epochs):
        indices = np.random.permutation(len(obs_pos))
        total_loss = 0
        for bi in range(n_batches):
            idx = indices[bi * batch_size:(bi + 1) * batch_size]
            loss, id_loss, smh_loss, cf_loss = model.compute_loss(
                obs_pos[idx], fut_pos[idx], ids[idx],
                observed_features=obs_feat[idx] if obs_feat is not None else None,
                future_features=fut_feat[idx] if fut_feat is not None else None,
                is_swap=is_swap[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f} "
                  f"(id={id_loss.item():.3f} smh={smh_loss.item():.3f} cf={cf_loss.item():.3f})")


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
            "name": "Baseline (no SMH, no CF)",
            "smh_weight": 0.0, "invariance_weight": 0.0,
            "sensitivity_weight": 0.0, "counterfactual_weight": 0.0,
        },
        {
            "name": "SMH-only (active)",
            "smh_weight": 1.0, "invariance_weight": 0.0,
            "sensitivity_weight": 0.0, "counterfactual_weight": 0.0,
        },
        {
            "name": "CF-only (clean-masked)",
            "smh_weight": 0.0, "invariance_weight": 0.5,
            "sensitivity_weight": 0.5, "counterfactual_weight": 1.0,
        },
        {
            "name": "SMH + CF (clean-masked)",
            "smh_weight": 1.0, "invariance_weight": 0.5,
            "sensitivity_weight": 0.5, "counterfactual_weight": 1.0,
        },
        {
            "name": "Strong SMH + CF (clean-masked)",
            "smh_weight": 3.0, "invariance_weight": 0.5,
            "sensitivity_weight": 0.5, "counterfactual_weight": 1.0,
        },
    ]

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {cfg['name']}")
        print(f"{'='*70}")

        model = CounterfactualObjectFileV2(
            num_objects=2, feature_dim=2, hidden_dim=128, slot_dim=64,
            t_obs=10, t_pred=20,
            identity_weight=1.0,
            smh_weight=cfg["smh_weight"],
            invariance_weight=cfg["invariance_weight"],
            sensitivity_weight=cfg["sensitivity_weight"],
            counterfactual_weight=cfg["counterfactual_weight"],
            traj_weight=0.1)

        train_model(model, train_data, epochs=40, batch_size=64, lr=1e-3)

        tester.full_diagnosis(model, cfg["name"], train_data, clean_test, swap_test)

        obs_pos_swap = swap_test["observed_positions"]
        obs_feat_swap = swap_test.get("object_features_obs")
        fut_feat_swap = swap_test.get("object_features_fut")
        fut_pos_swap = swap_test["future_positions"]
        true_id = swap_test["identity_labels"]

        for method in ["combined", "smh"]:
            pred = model.predict_identity(obs_pos_swap, obs_feat_swap,
                                          future_positions=fut_pos_swap,
                                          future_features=fut_feat_swap,
                                          method=method)
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            bd = compute_identity_breakdown(pred, true_id)
            print(f"  {method}: swap_only={bd['identity_swap_only']:.4f}")

    tester.print_fingerprint_map()

    print("\n" + "="*70)
    print("v13 vs v14 COMPARISON")
    print("="*70)
    print("""
v13 bug: CF pressures on ALL samples (including swap episodes)
  → conflicting gradients destroyed identity encoding
  → all configs with CF were State A

v14 fix: CF pressures only on CLEAN (non-swap) episodes
  → clean counterfactual signal
  → should preserve identity encoding while adding structural pressure

Expected improvement:
  - Baseline/SMH-only: should still be State D (trivially, from logits)
  - CF-only: should now be readable (not State A like v13)
  - SMH+CF: should be State D with better swap accuracy
""")


if __name__ == "__main__":
    main()
