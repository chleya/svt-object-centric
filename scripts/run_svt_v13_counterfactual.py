"""
SVT-v13: Counterfactual Training + Structure Monitoring Head

v12 finding: ALL learned models are State A (identity not readable).
This experiment tests whether counterfactual training + SMH can force
identity encoding into the model's intermediate representation.

Configurations tested:
  1. No SMH, no counterfactual (baseline = LearnedObjectFile equivalent)
  2. Active SMH only (force identity encoding via probe loss)
  3. Counterfactual only (invariance + sensitivity + counterfactual losses)
  4. Active SMH + Counterfactual (full system)
  5. Active SMH + Counterfactual + high SMH weight (strong identity pressure)

After training, we run subspace intervention diagnosis on each model
to check if identity is now readable and causal.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.counterfactual_object_file import CounterfactualObjectFile
from diagnostics.subspace_intervention import SubspaceInterventionTester, IdentityProbe


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
                future_features=fut_feat[idx] if fut_feat is not None else None)
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
            "smh_mode": "active", "traj_weight": 0.1,
        },
        {
            "name": "SMH-only (active)",
            "smh_weight": 1.0, "invariance_weight": 0.0,
            "sensitivity_weight": 0.0, "counterfactual_weight": 0.0,
            "smh_mode": "active", "traj_weight": 0.1,
        },
        {
            "name": "CF-only",
            "smh_weight": 0.0, "invariance_weight": 0.5,
            "sensitivity_weight": 0.5, "counterfactual_weight": 1.0,
            "smh_mode": "active", "traj_weight": 0.1,
        },
        {
            "name": "SMH + CF",
            "smh_weight": 1.0, "invariance_weight": 0.5,
            "sensitivity_weight": 0.5, "counterfactual_weight": 1.0,
            "smh_mode": "active", "traj_weight": 0.1,
        },
        {
            "name": "Strong SMH + CF",
            "smh_weight": 3.0, "invariance_weight": 0.5,
            "sensitivity_weight": 0.5, "counterfactual_weight": 1.0,
            "smh_mode": "active", "traj_weight": 0.1,
        },
    ]

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {cfg['name']}")
        print(f"{'='*70}")

        model = CounterfactualObjectFile(
            num_objects=2, feature_dim=2, hidden_dim=128, slot_dim=64,
            t_obs=10, t_pred=20,
            identity_weight=1.0,
            smh_weight=cfg["smh_weight"],
            invariance_weight=cfg["invariance_weight"],
            sensitivity_weight=cfg["sensitivity_weight"],
            counterfactual_weight=cfg["counterfactual_weight"],
            traj_weight=cfg["traj_weight"],
            smh_mode=cfg["smh_mode"])

        train_model(model, train_data, epochs=40, batch_size=64, lr=1e-3)

        tester.full_diagnosis(
            model, cfg["name"],
            train_data, clean_test, swap_test,
            method="combined")

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
    print("CRITICAL COMPARISON")
    print("="*70)
    print(f"\n{'Config':<35} {'Read':>7} {'Caus':>7} {'Swap':>7} {'State':<30}")
    print("-" * 86)
    for name, r in tester.results.items():
        print(f"{name:<35} {r['readability']:>7.3f} {r['causality']:>7.3f} "
              f"{r['swap_accuracy']:>7.3f} {r['state']:<30}")

    print("\n" + "="*70)
    print("KEY QUESTIONS")
    print("="*70)
    print("""
1. Does SMH force identity into the representation?
   → Compare Baseline (smh=0) vs SMH-only (smh=1) readability

2. Does counterfactual training help beyond SMH?
   → Compare SMH-only vs SMH+CF readability and causality

3. Does strong SMH over-regularize?
   → Compare SMH+CF vs Strong SMH+CF swap accuracy

4. Does any config reach State D (causal identity)?
   → If yes: counterfactual+SMH is the solution
   → If no: need even stronger identity pressure or different architecture
""")


if __name__ == "__main__":
    main()
