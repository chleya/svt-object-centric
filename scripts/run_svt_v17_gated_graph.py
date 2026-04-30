"""
SVT-v17: Gated Graph ObjectFile — Independent Conflict Detector + Graph-Level Gating

v16 finding: S4 + conflict training not sufficient — softmax edge network
cannot learn conditional weight modulation.

v17 solution: Independent conflict detector that GATES edge weights.
  - Conflict detector: separate network measuring feature-trajectory disagreement
  - Gate: when conflict detected, suppress feature edge, boost trajectory/conflict edges
  - Conflict-augmented training + conflict gate loss

This is the ConflictFirst gate idea implemented at the graph level.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.gated_graph_object_file import GatedGraphObjectFile
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
            loss, id_loss, smh_loss, cg_loss = model.compute_loss(
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
        {"name": "GatedGraph_pconf0", "p_conflict": 0.0},
        {"name": "GatedGraph_pconf02", "p_conflict": 0.2},
        {"name": "GatedGraph_pconf04", "p_conflict": 0.4},
    ]

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {cfg['name']}")
        print(f"{'='*70}")

        model = GatedGraphObjectFile(
            num_objects=2, feature_dim=2, hidden_dim=128, slot_dim=64,
            t_obs=10, t_pred=20, n_relation_types=3,
            identity_weight=1.0, smh_weight=1.0, traj_weight=0.1,
            conflict_gate_weight=0.5)

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

        ew_clean, cs_clean = model.get_edge_weights(
            clean_test["observed_positions"], clean_test.get("object_features_obs"),
            clean_test["future_positions"], clean_test.get("object_features_fut"))
        ew_conflict, cs_conflict = model.get_edge_weights(
            obs_pos_s, obs_feat_s, fut_pos_s, fut_feat_conflict)

        print(f"  Edge weights (clean vs conflict):")
        for r in range(ew_clean.shape[-1]):
            c_val = ew_clean[:, :, :, r].mean()
            cf_val = ew_conflict[:, :, :, r].mean()
            delta = cf_val - c_val
            print(f"    Rel {r}: clean={c_val:.4f} conflict={cf_val:.4f} delta={delta:+.4f}")

        print(f"  Conflict signal: clean={cs_clean.mean():.4f} conflict={cs_conflict.mean():.4f} delta={cs_conflict.mean()-cs_clean.mean():+.4f}")

    tester.print_fingerprint_map()

    print("\n" + "="*70)
    print("v17 KEY ANALYSIS: Does independent conflict gating enable conditional binding?")
    print("="*70)
    for name, r in tester.results.items():
        print(f"  {name}: State={r['state']}, Read={r['readability']:.3f}, Caus={r['causality']:.3f}, Swap={r['swap_accuracy']:.3f}")


if __name__ == "__main__":
    main()
