"""
SVT-v16: GraphObjectFile + Conflict-Augmented Training

v15b showed GraphObjectFile reaches State D but edge weights don't shift
under conflict. v16 adds conflict-augmented training to teach the model
to shift edge weights when features are misleading.

Hypothesis: Graph structure (S4) + conflict data = conditional identity binding
  - S4 provides the structural capacity (3 relation types)
  - Conflict data provides the training signal (when to shift)
  - Together they should enable conditional adjudication

Configs tested:
  1. GraphObjectFile (n_rel=3, p_conflict=0) — baseline (v15)
  2. GraphObjectFile (n_rel=3, p_conflict=0.2) — mild conflict
  3. GraphObjectFile (n_rel=3, p_conflict=0.4) — moderate conflict
  4. GraphObjectFile (n_rel=3, p_conflict=0.2, conflict_edge_loss) — mild + edge pressure
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


def train_model(model, train_data, epochs=40, batch_size=64, lr=1e-3, p_conflict=0.0):
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
            loss, id_loss, smh_loss, _ = model.compute_loss(
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
        {"name": "Graph_pconf0", "p_conflict": 0.0},
        {"name": "Graph_pconf02", "p_conflict": 0.2},
        {"name": "Graph_pconf04", "p_conflict": 0.4},
    ]

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {cfg['name']}")
        print(f"{'='*70}")

        model = GraphObjectFile(
            num_objects=2, feature_dim=2, hidden_dim=128, slot_dim=64,
            t_obs=10, t_pred=20, n_message_passes=2, n_relation_types=3,
            identity_weight=1.0, smh_weight=1.0, traj_weight=0.1)

        train_model(model, train_data, epochs=40, batch_size=64, lr=1e-3,
                    p_conflict=cfg["p_conflict"])

        tester.full_diagnosis(model, cfg["name"], train_data, clean_test, swap_test)

        obs_pos_s = swap_test["observed_positions"]
        obs_feat_s = swap_test.get("object_features_obs")
        fut_pos_s = swap_test["future_positions"]
        fut_feat_s = swap_test.get("object_features_fut")
        true_id_s = swap_test["identity_labels"]

        pred = model.predict_identity(obs_pos_s, obs_feat_s, future_positions=fut_pos_s, future_features=fut_feat_s)
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        bd = compute_identity_breakdown(pred, true_id_s)
        print(f"  swap_only (normal): {bd['identity_swap_only']:.4f}")

        fut_feat_conflict = swap_features(fut_feat_s)
        pred_c = model.predict_identity(obs_pos_s, obs_feat_s, future_positions=fut_pos_s, future_features=fut_feat_conflict)
        if isinstance(pred_c, torch.Tensor):
            pred_c = pred_c.cpu().numpy()
        bd_c = compute_identity_breakdown(pred_c, true_id_s)
        print(f"  swap_only (conflict): {bd_c['identity_swap_only']:.4f}")

        edge_w_clean = model.get_edge_weights(clean_test["observed_positions"],
                                               clean_test.get("object_features_obs"),
                                               clean_test["future_positions"],
                                               clean_test.get("object_features_fut"))
        edge_w_conflict = model.get_edge_weights(obs_pos_s, obs_feat_s, fut_pos_s, fut_feat_conflict)

        print(f"  Edge weights (clean vs conflict):")
        for r in range(edge_w_clean.shape[-1]):
            c_val = edge_w_clean[:, :, :, r].mean()
            cf_val = edge_w_conflict[:, :, :, r].mean()
            delta = cf_val - c_val
            print(f"    Rel {r}: clean={c_val:.4f} conflict={cf_val:.4f} delta={delta:+.4f}")

    tester.print_fingerprint_map()

    print("\n" + "="*70)
    print("v16 KEY ANALYSIS")
    print("="*70)
    for name, r in tester.results.items():
        print(f"\n  {name}:")
        print(f"    State: {r['state']}")
        print(f"    Readability: {r['readability']:.4f}")
        print(f"    Causality: {r['causality']:.4f}")
        print(f"    Swap Acc: {r['swap_accuracy']:.4f}")

    print("""
CRITICAL QUESTION:
  Does conflict-augmented training on graph structure enable
  conditional identity binding (edge weight shift under conflict)?

  If yes: S4 + conflict data = solution path
  If no: need even stronger architectural constraints
""")


if __name__ == "__main__":
    main()
