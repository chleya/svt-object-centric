"""
SVT-v15: Graph-Structured ObjectFile (S4 Substrate)

v14 finding: MLP binding cannot learn conditional identity adjudication.
v15 solution: Graph-structured binding with learned edge weights.

The key architectural difference:
  - MLP: pairwise matching via cat+MLP → unconditional
  - Graph: edge weights are FUNCTIONS of both nodes + relation type → conditional

Edge relation types:
  0: feature-based (high when features match)
  1: trajectory-based (high when trajectories are continuous)
  2: conflict (high when feature and trajectory disagree)

This allows the model to learn conditional adjudication:
  - When feature and trajectory agree → use both
  - When they conflict → use trajectory (via conflict edge)

After training, run subspace intervention + edge weight analysis.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.graph_object_file import GraphObjectFile
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
                  f"(id={id_loss.item():.3f} smh={smh_loss.item():.3f})")


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
        {"name": "GraphObjectFile (n_rel=3)", "n_relation_types": 3},
        {"name": "GraphObjectFile (n_rel=2)", "n_relation_types": 2},
        {"name": "GraphObjectFile (n_rel=1)", "n_relation_types": 1},
    ]

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {cfg['name']}")
        print(f"{'='*70}")

        model = GraphObjectFile(
            num_objects=2, feature_dim=2, hidden_dim=128, slot_dim=64,
            t_obs=10, t_pred=20,
            n_message_passes=2,
            n_relation_types=cfg["n_relation_types"],
            identity_weight=1.0, smh_weight=1.0, traj_weight=0.1)

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

        print("\n  Edge weight analysis on swap test:")
        edge_w = model.get_edge_weights(obs_pos_swap, obs_feat_swap, fut_pos_swap, fut_feat_swap)
        print(f"    Edge weights shape: {edge_w.shape}")
        if edge_w.ndim == 4:
            n_rel = edge_w.shape[-1]
            for r in range(n_rel):
                w_mean = edge_w[:, :, :, r].mean()
                print(f"    Relation type {r}: mean weight = {w_mean:.4f}")

    tester.print_fingerprint_map()

    print("\n" + "="*70)
    print("v14 vs v15 COMPARISON")
    print("="*70)
    print("""
v14 (MLP binding + CF training): State A (identity not readable)
  → MLP cannot learn conditional identity adjudication

v15 (Graph binding): ???
  → Graph edges can encode conditional dependencies
  → Key question: does graph structure enable State D?

If v15 reaches State D:
  → Confirms that S4 substrate is necessary for conditional identity binding
  → Provides a path forward for genuine object-file architectures

If v15 remains State A:
  → The problem is deeper than architecture
  → Need fundamentally different training or representation approach
""")


if __name__ == "__main__":
    main()
