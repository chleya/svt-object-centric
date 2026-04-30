"""
SVT-v18b: Training Signal Ablation — Is the fix sufficient for graph models?

v18 showed that DualPathwayObjectFile with corrected training achieves
conditional binding (conflict res = 0.879). But was the DUAL PATHWAY
architecture necessary, or would the corrected training signal alone
fix the graph models?

This ablation tests:
  1. GatedGraphObjectFile with CORRECTED training signal (no label swap)
  2. GraphObjectFile with CORRECTED training signal (no label swap)
  3. DualPathwayObjectFile (re-verify v18 result)

If corrected training alone fixes the graph models, then the training
signal was the ONLY bottleneck. If not, then the dual-pathway architecture
provides additional value beyond the training signal.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.gated_graph_object_file import GatedGraphObjectFile
from models.graph_object_file import GraphObjectFile
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


def train_model(model, train_data, epochs=60, batch_size=64, lr=1e-3, p_conflict=0.0):
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


def eval_conflict_resolution(model, swap_test, method="combined"):
    obs_pos_s = swap_test["observed_positions"]
    obs_feat_s = swap_test.get("object_features_obs")
    fut_pos_s = swap_test["future_positions"]
    fut_feat_s = swap_test.get("object_features_fut")
    true_id_s = swap_test["identity_labels"]

    pred_normal = model.predict_identity(obs_pos_s, obs_feat_s, future_positions=fut_pos_s,
                                          future_features=fut_feat_s, method=method)
    if isinstance(pred_normal, torch.Tensor):
        pred_normal = pred_normal.cpu().numpy()
    bd_normal = compute_identity_breakdown(pred_normal, true_id_s)

    fut_feat_conflict = swap_features(fut_feat_s)
    pred_conflict = model.predict_identity(obs_pos_s, obs_feat_s, future_positions=fut_pos_s,
                                            future_features=fut_feat_conflict, method=method)
    if isinstance(pred_conflict, torch.Tensor):
        pred_conflict = pred_conflict.cpu().numpy()
    bd_conflict = compute_identity_breakdown(pred_conflict, true_id_s)

    return bd_normal["identity_swap_only"], bd_conflict["identity_swap_only"]


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

    p_conflict = 0.4

    configs = [
        {
            "name": "GraphObjectFile_fixed",
            "model_cls": GraphObjectFile,
            "model_kwargs": {"num_objects": 2, "feature_dim": 2, "hidden_dim": 128,
                             "slot_dim": 64, "t_obs": 10, "t_pred": 20, "n_relation_types": 3},
        },
        {
            "name": "GatedGraph_fixed",
            "model_cls": GatedGraphObjectFile,
            "model_kwargs": {"num_objects": 2, "feature_dim": 2, "hidden_dim": 128,
                             "slot_dim": 64, "t_obs": 10, "t_pred": 20, "n_relation_types": 3,
                             "conflict_gate_weight": 0.5},
        },
        {
            "name": "DualPathway_verify",
            "model_cls": DualPathwayObjectFile,
            "model_kwargs": {"num_objects": 2, "feature_dim": 2, "hidden_dim": 128,
                             "slot_dim": 64, "t_obs": 10, "t_pred": 20,
                             "conflict_switch_temp": 0.1},
        },
    ]

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {cfg['name']} (p_conflict={p_conflict})")
        print(f"{'='*70}")

        model = cfg["model_cls"](**cfg["model_kwargs"])
        train_model(model, train_data, epochs=60, batch_size=64, lr=1e-3,
                    p_conflict=p_conflict)

        tester.full_diagnosis(model, cfg["name"], train_data, clean_test, swap_test)

        swap_normal, swap_conflict = eval_conflict_resolution(model, swap_test)
        print(f"\n  Conflict resolution analysis:")
        print(f"    Normal swap accuracy:  {swap_normal:.4f}")
        print(f"    Conflict swap accuracy: {swap_conflict:.4f}")
        print(f"    Conflict resolution:    {swap_conflict:.4f}")

        if hasattr(model, 'get_dual_scores'):
            ds = model.get_dual_scores(
                clean_test["observed_positions"], clean_test.get("object_features_obs"),
                clean_test["future_positions"], clean_test.get("object_features_fut"))
            print(f"    Agreement rate (clean): {ds['agreement'].mean():.4f}")

            obs_pos_s = swap_test["observed_positions"]
            obs_feat_s = swap_test.get("object_features_obs")
            fut_pos_s = swap_test["future_positions"]
            fut_feat_s = swap_test.get("object_features_fut")
            fut_feat_conflict = swap_features(fut_feat_s)
            ds_c = model.get_dual_scores(obs_pos_s, obs_feat_s, fut_pos_s, fut_feat_conflict)
            print(f"    Agreement rate (conflict): {ds_c['agreement'].mean():.4f}")

        if hasattr(model, 'get_edge_weights'):
            try:
                result = model.get_edge_weights(
                    clean_test["observed_positions"], clean_test.get("object_features_obs"),
                    clean_test["future_positions"], clean_test.get("object_features_fut"))
                if isinstance(result, tuple):
                    ew_clean, cs_clean = result
                else:
                    ew_clean = result
                    cs_clean = None

                obs_pos_s = swap_test["observed_positions"]
                obs_feat_s = swap_test.get("object_features_obs")
                fut_pos_s = swap_test["future_positions"]
                fut_feat_s = swap_test.get("object_features_fut")
                fut_feat_conflict = swap_features(fut_feat_s)
                result_c = model.get_edge_weights(
                    obs_pos_s, obs_feat_s, fut_pos_s, fut_feat_conflict)
                if isinstance(result_c, tuple):
                    ew_conflict, cs_conflict = result_c
                else:
                    ew_conflict = result_c
                    cs_conflict = None

                print(f"    Edge weights (clean vs conflict):")
                for r in range(ew_clean.shape[-1]):
                    c_val = ew_clean[:, :, :, r].mean()
                    cf_val = ew_conflict[:, :, :, r].mean()
                    delta = cf_val - c_val
                    print(f"      Rel {r}: clean={c_val:.4f} conflict={cf_val:.4f} delta={delta:+.4f}")
                if cs_clean is not None and cs_conflict is not None:
                    print(f"    Conflict signal: clean={cs_clean.mean():.4f} conflict={cs_conflict.mean():.4f}")
            except Exception as e:
                print(f"    Edge weight analysis failed: {e}")

    tester.print_fingerprint_map()

    print("\n" + "="*70)
    print("v18b ABLATION ANALYSIS: Training signal fix vs Architecture")
    print("="*70)
    for name, r in tester.results.items():
        print(f"  {name}: State={r['state']}, Read={r['readability']:.3f}, "
              f"Caus={r['causality']:.3f}, Swap={r['swap_accuracy']:.3f}")

    print("\n--- Interpretation ---")
    print("  If GraphObjectFile_fixed achieves conditional binding:")
    print("    -> Training signal was the ONLY bottleneck; S4 architecture is sufficient")
    print("  If only DualPathway achieves conditional binding:")
    print("    -> Both training signal AND dual-pathway architecture are necessary (S5)")
    print("  If GatedGraph_fixed outperforms GraphObjectFile_fixed:")
    print("    -> Independent conflict detection adds value beyond corrected training")


if __name__ == "__main__":
    main()
