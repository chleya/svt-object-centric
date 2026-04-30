"""
SVT-v15b: GraphObjectFile Conflict Test

v15 showed GraphObjectFile reaches State D on clean data.
v15b tests whether it maintains identity under feature-trajectory conflict.

Key question: Do edge weights change under conflict conditions?
If yes → the model uses conditional adjudication
If no → the model just learned a better feature matching
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.graph_object_file import GraphObjectFile


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
        for bi in range(n_batches):
            idx = indices[bi * batch_size:(bi + 1) * batch_size]
            loss, id_loss, smh_loss, _ = model.compute_loss(
                obs_pos[idx], fut_pos[idx], ids[idx],
                observed_features=obs_feat[idx] if obs_feat is not None else None,
                future_features=fut_feat[idx] if fut_feat is not None else None)
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


def main():
    seed = 42
    print("Generating datasets...")
    eval_ds = generate_v3_dataset(n_train=1000, n_test=200, num_objects=2, feature_dim=2,
        feature_mode="feature_bearing", force_train_type="attractor", force_test_type="vortex",
        randomize_object_order=True, disjoint_init_split=True, seed=seed)
    train_data = gen_train(n=1000, seed=seed)
    swap_test = eval_ds["identity_test_swap_only"]
    clean_test = eval_ds["clean_test_id"]

    print("\nTraining GraphObjectFile (n_rel=3)...")
    model = GraphObjectFile(
        num_objects=2, feature_dim=2, hidden_dim=128, slot_dim=64,
        t_obs=10, t_pred=20, n_message_passes=2, n_relation_types=3,
        identity_weight=1.0, smh_weight=1.0, traj_weight=0.1)
    train_model(model, train_data, epochs=40, batch_size=64, lr=1e-3)

    print("\n" + "="*70)
    print("TEST 1: Clean identity (no conflict)")
    print("="*70)
    obs_pos = clean_test["observed_positions"]
    obs_feat = clean_test.get("object_features_obs")
    fut_pos = clean_test["future_positions"]
    fut_feat = clean_test.get("object_features_fut")
    true_id = clean_test["identity_labels"]

    pred = model.predict_identity(obs_pos, obs_feat, future_positions=fut_pos, future_features=fut_feat)
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    bd = compute_identity_breakdown(pred, true_id)
    print(f"  overall={bd['identity_overall']:.4f}")

    edge_w_clean = model.get_edge_weights(obs_pos, obs_feat, fut_pos, fut_feat)
    print(f"  Edge weights (clean):")
    for r in range(edge_w_clean.shape[-1]):
        print(f"    Relation {r}: {edge_w_clean[:, :, :, r].mean():.4f}")

    print("\n" + "="*70)
    print("TEST 2: Feature-trajectory conflict (swapped features)")
    print("="*70)
    obs_pos_s = swap_test["observed_positions"]
    obs_feat_s = swap_test.get("object_features_obs")
    fut_pos_s = swap_test["future_positions"]
    fut_feat_s = swap_test.get("object_features_fut")
    true_id_s = swap_test["identity_labels"]

    fut_feat_conflict = swap_features(fut_feat_s)

    pred_conflict = model.predict_identity(obs_pos_s, obs_feat_s, future_positions=fut_pos_s, future_features=fut_feat_conflict)
    if isinstance(pred_conflict, torch.Tensor):
        pred_conflict = pred_conflict.cpu().numpy()
    bd_conflict = compute_identity_breakdown(pred_conflict, true_id_s)
    print(f"  swap_only={bd_conflict['identity_swap_only']:.4f}")

    edge_w_conflict = model.get_edge_weights(obs_pos_s, obs_feat_s, fut_pos_s, fut_feat_conflict)
    print(f"  Edge weights (conflict):")
    for r in range(edge_w_conflict.shape[-1]):
        print(f"    Relation {r}: {edge_w_conflict[:, :, :, r].mean():.4f}")

    print("\n" + "="*70)
    print("TEST 3: Feature ablation (no features)")
    print("="*70)
    pred_ablat = model.predict_identity(obs_pos, None, future_positions=fut_pos, future_features=None)
    if isinstance(pred_ablat, torch.Tensor):
        pred_ablat = pred_ablat.cpu().numpy()
    bd_ablat = compute_identity_breakdown(pred_ablat, true_id)
    print(f"  overall={bd_ablat['identity_overall']:.4f}")

    edge_w_ablat = model.get_edge_weights(obs_pos, None, fut_pos, None)
    print(f"  Edge weights (ablation):")
    for r in range(edge_w_ablat.shape[-1]):
        print(f"    Relation {r}: {edge_w_ablat[:, :, :, r].mean():.4f}")

    print("\n" + "="*70)
    print("TEST 4: Normal swap test (no artificial conflict)")
    print("="*70)
    pred_swap = model.predict_identity(obs_pos_s, obs_feat_s, future_positions=fut_pos_s, future_features=fut_feat_s)
    if isinstance(pred_swap, torch.Tensor):
        pred_swap = pred_swap.cpu().numpy()
    bd_swap = compute_identity_breakdown(pred_swap, true_id_s)
    print(f"  swap_only={bd_swap['identity_swap_only']:.4f}")

    edge_w_swap = model.get_edge_weights(obs_pos_s, obs_feat_s, fut_pos_s, fut_feat_s)
    print(f"  Edge weights (swap):")
    for r in range(edge_w_swap.shape[-1]):
        print(f"    Relation {r}: {edge_w_swap[:, :, :, r].mean():.4f}")

    print("\n" + "="*70)
    print("EDGE WEIGHT COMPARISON")
    print("="*70)
    print(f"{'Condition':<25} {'Rel 0 (feat)':>14} {'Rel 1 (traj)':>14} {'Rel 2 (conf)':>14}")
    print("-" * 67)
    for name, ew in [("Clean", edge_w_clean), ("Swap (normal)", edge_w_swap),
                     ("Conflict (swapped)", edge_w_conflict), ("Ablation", edge_w_ablat)]:
        vals = [f"{ew[:, :, :, r].mean():.4f}" for r in range(ew.shape[-1])]
        print(f"{name:<25} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14}")

    print("\n" + "="*70)
    print("KEY ANALYSIS")
    print("="*70)

    conflict_rel0 = edge_w_conflict[:, :, :, 0].mean()
    conflict_rel1 = edge_w_conflict[:, :, :, 1].mean()
    conflict_rel2 = edge_w_conflict[:, :, :, 2].mean()
    clean_rel0 = edge_w_clean[:, :, :, 0].mean()
    clean_rel1 = edge_w_clean[:, :, :, 1].mean()
    clean_rel2 = edge_w_clean[:, :, :, 2].mean()

    if conflict_rel0 < clean_rel0 and conflict_rel1 > clean_rel1:
        print("  ✅ Edge weights SHIFT under conflict:")
        print("    Feature edge decreases, trajectory edge increases")
        print("    → Model uses conditional adjudication!")
    elif conflict_rel2 > clean_rel2:
        print("  ⚠️ Conflict edge increases under conflict:")
        print("    Model detects conflict but may not resolve it correctly")
    else:
        print("  ❌ Edge weights do NOT shift under conflict:")
        print("    Model does not use conditional adjudication")

    print(f"\n  Conflict resolution accuracy: {bd_conflict['identity_swap_only']:.4f}")
    if bd_conflict['identity_swap_only'] > 0.3:
        print("  → GraphObjectFile partially resolves conflict!")
    else:
        print("  → GraphObjectFile still fails under conflict (like published models)")


if __name__ == "__main__":
    main()
