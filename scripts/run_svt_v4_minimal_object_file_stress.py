"""
SVT-v4: Minimal Object-File Stress Test
"""

import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v4_minimal_object_file_stress"
SEED = 0
SWAP_RATIO = 0.3
EPOCHS = 20

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F_torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

from data.generate_nonlinear_feature_ood import _generate_single_episode, _stack_episodes
from data.generate_nonlinear_feature_ood import generate_v3_dataset
from metrics.prediction_metrics import compute_prediction_metrics
from metrics.identity_breakdown import compute_identity_breakdown


def save_csv(results, filename, fieldnames):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, filename), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def fmt(val, decimals=4):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "nan"
    return f"{val:.{decimals}f}"


def generate_swap_augmented_train(n_train=1000, swap_ratio=0.5, seed=0,
                                   force_type="attractor", feature_mode="feature_bearing"):
    rng = np.random.RandomState(seed)
    episodes = []
    for _ in range(n_train):
        ep = _generate_single_episode(
            t_obs=10, t_pred=20, num_objects=2, arena_size=64.0,
            feature_mode=feature_mode, randomize_object_order=True,
            identity_test=True, swap_probability=swap_ratio,
            force_type=force_type, field_strength=0.5,
            damping=0.95, noise_std=0.1, rng=rng,
        )
        episodes.append(ep)
    return _stack_episodes(episodes, feature_mode)


def shuffle_features(features, seed=42):
    if features is None:
        return None
    rng = np.random.RandomState(seed)
    shuffled = features.copy()
    B, T, N, Fd = shuffled.shape
    for b in range(B):
        for t in range(T):
            perm = rng.permutation(N)
            shuffled[b, t] = shuffled[b, t, perm]
    return shuffled


def zero_features(features):
    if features is None:
        return None
    return np.zeros_like(features)


def add_feature_noise(features, noise_std, seed=42):
    if features is None:
        return None
    rng = np.random.RandomState(seed)
    noisy = features.copy()
    noisy += rng.randn(*noisy.shape) * noise_std
    return noisy.astype(np.float32)


def flip_future_features(future_features):
    if future_features is None:
        return None
    flipped = future_features.copy()
    flipped[:, :, 0, :], flipped[:, :, 1, :] = flipped[:, :, 1, :].copy(), flipped[:, :, 0, :].copy()
    return flipped


def add_position_noise(positions, noise_std, seed=42):
    if positions is None:
        return None
    rng = np.random.RandomState(seed)
    noisy = positions.copy()
    noisy += rng.randn(*noisy.shape) * noise_std
    return noisy.astype(np.float32)


def compute_assignment_accuracy(pred_identity, true_identity):
    correct = (pred_identity == true_identity).all(axis=1)
    return float(correct.mean())


def evaluate_learned_model(model, test_data, obs_feat_override=None, fut_feat_override=None,
                           fut_pos_override=None):
    obs_pos = test_data["observed_positions"]
    obs_feat = obs_feat_override if obs_feat_override is not None else test_data.get("object_features_obs")
    fut_feat = fut_feat_override if fut_feat_override is not None else test_data.get("object_features_fut")
    fut_pos = fut_pos_override if fut_pos_override is not None else test_data.get("future_positions")

    pred_future = model.predict_future(obs_pos, obs_feat)
    if isinstance(pred_future, torch.Tensor):
        pred_future = pred_future.cpu().numpy()
    pred_metrics = compute_prediction_metrics(pred_future, test_data["future_positions"])

    pred_identity = model.predict_identity(obs_pos, obs_feat, future_features=fut_feat)
    if isinstance(pred_identity, torch.Tensor):
        pred_identity = pred_identity.cpu().numpy()

    true_identity = test_data["identity_labels"]
    asgn_acc = compute_assignment_accuracy(pred_identity, true_identity)
    breakdown = compute_identity_breakdown(pred_identity, true_identity)

    return pred_metrics, breakdown, asgn_acc


def evaluate_trajectory_only(model, test_data, fut_pos_override=None):
    obs_pos = test_data["observed_positions"]
    fut_pos = fut_pos_override if fut_pos_override is not None else test_data.get("future_positions")

    pred_future = model.predict_future(obs_pos)
    if isinstance(pred_future, torch.Tensor):
        pred_future = pred_future.cpu().numpy()
    pred_metrics = compute_prediction_metrics(pred_future, test_data["future_positions"])

    pred_identity = model.predict_identity(obs_pos, future_positions=fut_pos)
    if isinstance(pred_identity, torch.Tensor):
        pred_identity = pred_identity.cpu().numpy()

    true_identity = test_data["identity_labels"]
    asgn_acc = compute_assignment_accuracy(pred_identity, true_identity)
    breakdown = compute_identity_breakdown(pred_identity, true_identity)

    return pred_metrics, breakdown, asgn_acc


def evaluate_object_file(mechanism, test_data, obs_feat_override=None,
                          fut_feat_override=None, occlusion_mask=None):
    obs_pos = test_data["observed_positions"]
    obs_feat = obs_feat_override if obs_feat_override is not None else test_data.get("object_features_obs")
    fut_feat = fut_feat_override if fut_feat_override is not None else test_data.get("object_features_fut")
    fut_pos = test_data["future_positions"]

    pred_identity = mechanism.predict_identity(obs_pos, obs_feat, fut_pos, fut_feat,
                                                occlusion_mask=occlusion_mask)
    true_identity = test_data["identity_labels"]
    asgn_acc = compute_assignment_accuracy(pred_identity, true_identity)
    breakdown = compute_identity_breakdown(pred_identity, true_identity)

    skill = float("nan")
    return {"skill_score": skill}, breakdown, asgn_acc


def make_occlusion_mask(future_positions, occlusion_ratio):
    if occlusion_ratio <= 0:
        return None
    B, T, N, _ = future_positions.shape
    mask = np.zeros((B, T, N), dtype=bool)
    n_occluded = int(T * occlusion_ratio)
    start = (T - n_occluded) // 2
    mask[:, start:start + n_occluded, :] = True
    return mask


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SVT-v4: Minimal Object-File Stress Test")
    print("=" * 60)
    print(f"PyTorch: {'available' if TORCH_AVAILABLE else 'UNAVAILABLE'} (device={DEVICE})")

    if not TORCH_AVAILABLE:
        print("PyTorch unavailable. Exiting.")
        return

    from models.feature_similarity_models import FeatureOnlyAssignmentHead, HybridTrajectoryFeatureAssignmentHead
    from models.object_file_models import TrajectoryOnlyAssignment, MinimalObjectFileMechanism, LearnedTrajObjectFile
    from utils.torch_training import train_model

    eval_ds = generate_v3_dataset(
        n_train=1000, n_test=200,
        feature_mode="feature_bearing",
        force_train_type="attractor",
        force_test_type="vortex",
        randomize_object_order=True,
        disjoint_init_split=True,
        seed=SEED,
    )

    train_data = generate_swap_augmented_train(
        n_train=1000, swap_ratio=SWAP_RATIO, seed=SEED,
    )

    swap_test = eval_ds["identity_test_swap_only"]
    clean_test = eval_ds["clean_test_id"]

    # =========================================================================
    # Train models
    # =========================================================================
    print("\n=== Training Models ===")

    print("  Training FeatureOnly...")
    model_fo = FeatureOnlyAssignmentHead(identity_weight=1.0)
    train_model(model_fo, train_data, val_data=clean_test,
                epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=True, uses_future_features=True, verbose=False)

    print("  Training TrajectoryOnly...")
    model_traj = TrajectoryOnlyAssignment()
    train_model(model_traj, train_data, val_data=clean_test,
                epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=False, uses_future_features=False, verbose=False)

    print("  Training Hybrid(beta=1.0)...")
    model_hybrid1 = HybridTrajectoryFeatureAssignmentHead(identity_weight=1.0, beta=1.0)
    train_model(model_hybrid1, train_data, val_data=clean_test,
                epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=True, uses_future_features=True, verbose=False)

    print("  Training Hybrid(beta=2.0)...")
    model_hybrid2 = HybridTrajectoryFeatureAssignmentHead(identity_weight=1.0, beta=2.0)
    train_model(model_hybrid2, train_data, val_data=clean_test,
                epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=True, uses_future_features=True, verbose=False)

    obj_file = MinimalObjectFileMechanism(feature_weight=1.0, traj_weight=1.0)
    obj_file_learned = LearnedTrajObjectFile(traj_model=model_traj, feature_weight=1.0, traj_weight=1.0)

    # =========================================================================
    # Part 1: Mechanism Comparison (normal features)
    # =========================================================================
    print("\n=== Mechanism Comparison ===")

    mech_results = []

    mechanisms = [
        ("FeatureOnly", "learned", model_fo),
        ("TrajectoryOnly", "traj_only", model_traj),
        ("Hybrid_b1.0", "learned", model_hybrid1),
        ("Hybrid_b2.0", "learned", model_hybrid2),
        ("ObjectFile", "obj_file", obj_file),
        ("ObjFileLearned", "obj_file_learned", obj_file_learned),
    ]

    for mech_name, mech_type, mech in mechanisms:
        if mech_type == "learned":
            _, bd, asgn_acc = evaluate_learned_model(mech, swap_test)
        elif mech_type == "traj_only":
            _, bd, asgn_acc = evaluate_trajectory_only(mech, swap_test)
        else:
            _, bd, asgn_acc = evaluate_object_file(mech, swap_test)

        # Feature ablation
        obs_feat_shuf = shuffle_features(swap_test["object_features_obs"], seed=SEED)
        fut_feat_shuf = shuffle_features(swap_test["object_features_fut"], seed=SEED + 1000)
        obs_feat_zero = zero_features(swap_test["object_features_obs"])
        fut_feat_zero = zero_features(swap_test["object_features_fut"])

        if mech_type == "learned":
            _, bd_shuf, _ = evaluate_learned_model(mech, swap_test,
                                                    obs_feat_override=obs_feat_shuf,
                                                    fut_feat_override=fut_feat_shuf)
            _, bd_zero, _ = evaluate_learned_model(mech, swap_test,
                                                    obs_feat_override=obs_feat_zero,
                                                    fut_feat_override=fut_feat_zero)
        elif mech_type == "traj_only":
            bd_shuf = bd
            bd_zero = bd
        else:
            _, bd_shuf, _ = evaluate_object_file(mech, swap_test,
                                                   obs_feat_override=obs_feat_shuf,
                                                   fut_feat_override=fut_feat_shuf)
            _, bd_zero, _ = evaluate_object_file(mech, swap_test,
                                                   obs_feat_override=obs_feat_zero,
                                                   fut_feat_override=fut_feat_zero)

        feat_dep = bd["identity_swap_only"] - bd_shuf["identity_swap_only"]
        traj_dep = bd_zero["identity_swap_only"]
        no_swap_gap = bd["identity_no_swap"] - bd["identity_swap_only"] if not np.isnan(bd["identity_no_swap"]) else float("nan")

        mech_results.append({
            "mechanism": mech_name,
            "identity_swap_only": fmt(bd["identity_swap_only"]),
            "identity_overall": fmt(bd["identity_overall"]),
            "feature_dependency_score": fmt(feat_dep),
            "trajectory_dependency_score": fmt(traj_dep),
            "no_swap_bias_gap": fmt(no_swap_gap),
            "confidence_calibration": "nan",
        })

        print(f"  {mech_name}: swap={fmt(bd['identity_swap_only'])} feat_dep={fmt(feat_dep)} traj_dep={fmt(traj_dep)}")

    save_csv(mech_results, "mechanism_comparison.csv",
             ["mechanism", "identity_swap_only", "identity_overall",
              "feature_dependency_score", "trajectory_dependency_score",
              "no_swap_bias_gap", "confidence_calibration"])

    # =========================================================================
    # Part 2: Feature Noise Stress Test
    # =========================================================================
    print("\n=== Feature Noise Stress Test ===")

    noise_results = []

    for noise_std in [0.0, 0.1, 0.3, 0.5]:
        print(f"  noise_std={noise_std}")

        obs_noisy = add_feature_noise(swap_test["object_features_obs"], noise_std, seed=SEED)
        fut_noisy = add_feature_noise(swap_test["object_features_fut"], noise_std, seed=SEED + 1000)

        for mech_name, mech_type, mech in mechanisms:
            if mech_type == "learned":
                _, bd, _ = evaluate_learned_model(mech, swap_test,
                                                   obs_feat_override=obs_noisy,
                                                   fut_feat_override=fut_noisy)
            elif mech_type == "traj_only":
                _, bd, _ = evaluate_trajectory_only(mech, swap_test)
            else:
                _, bd, _ = evaluate_object_file(mech, swap_test,
                                                 obs_feat_override=obs_noisy,
                                                 fut_feat_override=fut_noisy)

            obs_shuf = shuffle_features(swap_test["object_features_obs"], seed=SEED)
            fut_shuf = shuffle_features(swap_test["object_features_fut"], seed=SEED + 1000)
            if mech_type == "learned":
                _, bd_shuf, _ = evaluate_learned_model(mech, swap_test,
                                                        obs_feat_override=obs_shuf,
                                                        fut_feat_override=fut_shuf)
            elif mech_type == "traj_only":
                bd_shuf = bd
            else:
                _, bd_shuf, _ = evaluate_object_file(mech, swap_test,
                                                       obs_feat_override=obs_shuf,
                                                       fut_feat_override=fut_shuf)

            feat_dep = bd["identity_swap_only"] - bd_shuf["identity_swap_only"]

            noise_results.append({
                "mechanism": mech_name,
                "feature_noise_std": str(noise_std),
                "identity_swap_only": fmt(bd["identity_swap_only"]),
                "identity_under_feature_noise": fmt(bd["identity_swap_only"]),
                "feature_dependency_score": fmt(feat_dep),
            })

            print(f"    {mech_name}: swap={fmt(bd['identity_swap_only'])} feat_dep={fmt(feat_dep)}")

    save_csv(noise_results, "feature_noise_results.csv",
             ["mechanism", "feature_noise_std", "identity_swap_only",
              "identity_under_feature_noise", "feature_dependency_score"])

    # =========================================================================
    # Part 3: Occlusion Without Feature
    # =========================================================================
    print("\n=== Occlusion Without Feature Stress Test ===")

    occ_results = []

    for occ_ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(f"  occlusion_ratio={occ_ratio}")

        occ_mask = make_occlusion_mask(swap_test["future_positions"], occ_ratio)

        if occ_ratio > 0:
            fut_feat_occ = swap_test["object_features_fut"].copy()
            if occ_mask is not None:
                for b in range(fut_feat_occ.shape[0]):
                    for t in range(fut_feat_occ.shape[1]):
                        for i in range(fut_feat_occ.shape[2]):
                            if occ_mask[b, t, i]:
                                fut_feat_occ[b, t, i, :] = 0.0
        else:
            fut_feat_occ = swap_test["object_features_fut"]

        for mech_name, mech_type, mech in mechanisms:
            if mech_type == "learned":
                _, bd, _ = evaluate_learned_model(mech, swap_test,
                                                   fut_feat_override=fut_feat_occ)
            elif mech_type == "traj_only":
                _, bd, _ = evaluate_trajectory_only(mech, swap_test)
            else:
                _, bd, _ = evaluate_object_file(mech, swap_test,
                                                 fut_feat_override=fut_feat_occ,
                                                 occlusion_mask=occ_mask)

            occ_results.append({
                "mechanism": mech_name,
                "occlusion_ratio": str(occ_ratio),
                "identity_swap_only": fmt(bd["identity_swap_only"]),
                "identity_under_occlusion_no_feature": fmt(bd["identity_swap_only"]),
            })

            print(f"    {mech_name}: swap={fmt(bd['identity_swap_only'])}")

    save_csv(occ_results, "occlusion_no_feature_results.csv",
             ["mechanism", "occlusion_ratio", "identity_swap_only",
              "identity_under_occlusion_no_feature"])

    # =========================================================================
    # Part 4: Feature-Trajectory Conflict
    # =========================================================================
    print("\n=== Feature-Trajectory Conflict ===")

    is_swap_clean = clean_test["is_swap"]
    no_swap_idx = np.where(~is_swap_clean)[0]
    swap_idx = np.where(is_swap_clean)[0]

    conflict_results = []

    # Type A: feature_wrong (no-swap with flipped future features)
    if len(no_swap_idx) > 0:
        print(f"  Type A: feature_wrong (n={len(no_swap_idx)})")
        no_swap_data = {k: v[no_swap_idx] for k, v in clean_test.items()}
        fut_flipped = flip_future_features(no_swap_data["object_features_fut"])

        true_id = no_swap_data["identity_labels"]
        is_actually_swap = true_id[:, 0] != 0
        n_actual_swap = int(is_actually_swap.sum())
        n_actual_no_swap = int((~is_actually_swap).sum())

        for mech_name, mech_type, mech in mechanisms:
            if mech_type == "learned":
                pred_identity = mech.predict_identity(
                    no_swap_data["observed_positions"],
                    no_swap_data["object_features_obs"],
                    future_features=fut_flipped)
                if isinstance(pred_identity, torch.Tensor):
                    pred_identity = pred_identity.cpu().numpy()
            elif mech_type == "traj_only":
                pred_identity = mech.predict_identity(
                    no_swap_data["observed_positions"],
                    future_positions=no_swap_data["future_positions"])
                if isinstance(pred_identity, torch.Tensor):
                    pred_identity = pred_identity.cpu().numpy()
            else:
                pred_identity = mech.predict_identity(
                    no_swap_data["observed_positions"],
                    no_swap_data["object_features_obs"],
                    no_swap_data["future_positions"],
                    fut_flipped)

            correct = (pred_identity == true_id).all(axis=1)
            identity_acc = float(correct.mean())

            pred_is_swap = pred_identity[:, 0] != 0
            if n_actual_no_swap > 0:
                traj_correct_rate = float((~pred_is_swap[~is_actually_swap]).sum() / n_actual_no_swap)
            else:
                traj_correct_rate = float("nan")
            if n_actual_no_swap > 0:
                feat_wrong_rate = float(pred_is_swap[~is_actually_swap].sum() / n_actual_no_swap)
            else:
                feat_wrong_rate = float("nan")

            conflict_results.append({
                "mechanism": mech_name,
                "conflict_type": "feature_wrong_trajectory_correct",
                "identity_under_conflict": fmt(identity_acc),
                "conflict_resolution_rate_feature_correct": fmt(feat_wrong_rate),
                "conflict_resolution_rate_trajectory_correct": fmt(traj_correct_rate),
            })

            print(f"    {mech_name}: acc={fmt(identity_acc)} traj_correct={fmt(traj_correct_rate)} feat_wrong={fmt(feat_wrong_rate)}")
    else:
        print("  Type A: no no-swap episodes found, skipping")

    # Type B: trajectory_noisy (swap with position noise)
    if len(swap_idx) > 0:
        print(f"  Type B: trajectory_noisy (n={len(swap_idx)})")
        swap_data = {k: v[swap_idx] for k, v in clean_test.items()}

        for pos_noise in [2.0, 5.0, 10.0]:
            fut_noisy = add_position_noise(swap_data["future_positions"], pos_noise, seed=SEED)

            for mech_name, mech_type, mech in mechanisms:
                if mech_type == "learned":
                    _, bd, _ = evaluate_learned_model(mech, swap_data,
                                                       fut_feat_override=swap_data["object_features_fut"])
                elif mech_type == "traj_only":
                    _, bd, _ = evaluate_trajectory_only(mech, swap_data,
                                                         fut_pos_override=fut_noisy)
                else:
                    _, bd, _ = evaluate_object_file(mech, swap_data,
                                                     fut_feat_override=swap_data["object_features_fut"])

                conflict_results.append({
                    "mechanism": mech_name,
                    "conflict_type": f"trajectory_noisy_std{pos_noise}",
                    "identity_under_conflict": fmt(bd["identity_swap_only"]),
                    "conflict_resolution_rate_feature_correct": fmt(bd["identity_swap_only"]),
                    "conflict_resolution_rate_trajectory_correct": fmt(1.0 - bd["identity_swap_only"]),
                })

                if mech_name == "FeatureOnly" or pos_noise == 5.0:
                    print(f"    {mech_name}(noise={pos_noise}): swap={fmt(bd['identity_swap_only'])}")

    save_csv(conflict_results, "conflict_results.csv",
             ["mechanism", "conflict_type", "identity_under_conflict",
              "conflict_resolution_rate_feature_correct",
              "conflict_resolution_rate_trajectory_correct"])

    # =========================================================================
    # Part 5: Object File Trace Examples
    # =========================================================================
    print("\n=== Object File Trace Examples ===")

    n_examples = min(5, swap_test["observed_positions"].shape[0])
    trace_data = {k: v[:n_examples] for k, v in swap_test.items()}

    results_trace, traces = obj_file.predict_identity(
        trace_data["observed_positions"],
        trace_data["object_features_obs"],
        trace_data["future_positions"],
        trace_data["object_features_fut"],
        return_trace=True,
    )

    trace_rows = []
    for ep_traces in traces:
        for step in ep_traces:
            trace_rows.append({
                "episode_idx": step["episode_idx"],
                "timestep": step["timestep"],
                "object_idx": step["object_idx"],
                "identity_key_0": fmt(step["identity_key"][0]),
                "identity_key_1": fmt(step["identity_key"][1]),
                "last_pos_x": fmt(step["last_pos"][0]),
                "last_pos_y": fmt(step["last_pos"][1]),
                "occluded": str(step["occluded"]),
                "confidence": fmt(step["confidence"]),
            })

    save_csv(trace_rows, "object_file_trace_examples.csv",
             ["episode_idx", "timestep", "object_idx", "identity_key_0",
              "identity_key_1", "last_pos_x", "last_pos_y",
              "occluded", "confidence"])

    print(f"  Saved {len(trace_rows)} trace rows")

    # =========================================================================
    # README
    # =========================================================================
    fo_noise_03 = [r for r in noise_results
                   if r["mechanism"] == "FeatureOnly" and r["feature_noise_std"] == "0.3"]
    fo_noise_drop = 0.0
    fo_normal = [r for r in mech_results if r["mechanism"] == "FeatureOnly"]
    if fo_noise_03 and fo_normal:
        fo_noise_drop = float(fo_normal[0]["identity_swap_only"]) - float(fo_noise_03[0]["identity_swap_only"])

    traj_conflict = [r for r in conflict_results
                     if r["mechanism"] == "TrajectoryOnly" and "feature_wrong" in r["conflict_type"]]
    traj_misled = False
    if traj_conflict:
        traj_misled = float(traj_conflict[0]["conflict_resolution_rate_trajectory_correct"]) < 0.5

    hybrid_conflict = [r for r in conflict_results
                       if r["mechanism"] == "Hybrid_b1.0" and "feature_wrong" in r["conflict_type"]]
    hybrid_is_compromise = False
    if hybrid_conflict:
        rate = float(hybrid_conflict[0]["identity_under_conflict"])
        hybrid_is_compromise = 0.3 < rate < 0.7

    of_occ_075 = [r for r in occ_results
                  if r["mechanism"] == "ObjectFile" and r["occlusion_ratio"] == "0.75"]
    fo_occ_075 = [r for r in occ_results
                  if r["mechanism"] == "FeatureOnly" and r["occlusion_ratio"] == "0.75"]
    of_more_stable = False
    if of_occ_075 and fo_occ_075:
        of_more_stable = float(of_occ_075[0]["identity_swap_only"]) > float(fo_occ_075[0]["identity_swap_only"])

    of_normal_row = [r for r in mech_results if r["mechanism"] == "ObjectFile"]
    of_advantage = of_more_stable or (of_normal_row and float(of_normal_row[0]["feature_dependency_score"]) > 0.2)

    if of_advantage:
        recommendation = "improve_object_file_update_rule"
    elif fo_noise_drop > 0.3:
        recommendation = "add_confidence_calibration"
    elif float(fo_normal[0]["identity_swap_only"]) > 0.95 if fo_normal else False:
        recommendation = "benchmark_too_easy"
    else:
        recommendation = "shortcut_still_dominates"

    readme = f"""# SVT-v4: Minimal Object-File Stress Test

## 1. Purpose

v3.6 proved temporal-aligned feature key achieves perfect identity on clean data. v4 adds stress: feature noise, occlusion, and feature-trajectory conflict.

## 2. Mechanisms

1. **FeatureOnly**: assignment = feature cosine similarity (v3.6 temporal-aligned)
2. **TrajectoryOnly**: assignment = nearest predicted position
3. **Hybrid(beta=1.0/2.0)**: assignment = traj_logits + beta * feature_logits
4. **MinimalObjectFile**: rule-based with identity_key, trajectory_state, occlusion handling

## 3. Mechanism Comparison (normal features)

| Mechanism | Swap-Only | Feat Dep | Traj Dep | No-Swap Gap |
|-----------|-----------|----------|----------|-------------|
"""

    for r in mech_results:
        readme += f"| {r['mechanism']} | {r['identity_swap_only']} | {r['feature_dependency_score']} | {r['trajectory_dependency_score']} | {r['no_swap_bias_gap']} |\n"

    readme += f"""
## 4. Feature Noise

| Mechanism | Noise=0.0 | Noise=0.1 | Noise=0.3 | Noise=0.5 |
|-----------|-----------|-----------|-----------|-----------|
"""

    for mech_name in ["FeatureOnly", "TrajectoryOnly", "Hybrid_b1.0", "Hybrid_b2.0", "ObjectFile"]:
        vals = []
        for ns in ["0.0", "0.1", "0.3", "0.5"]:
            rows = [r for r in noise_results if r["mechanism"] == mech_name and r["feature_noise_std"] == ns]
            vals.append(rows[0]["identity_swap_only"] if rows else "nan")
        readme += f"| {mech_name} | {' | '.join(vals)} |\n"

    readme += f"""
## 5. Occlusion Without Feature

| Mechanism | Occ=0.0 | Occ=0.25 | Occ=0.5 | Occ=0.75 | Occ=1.0 |
|-----------|---------|----------|---------|----------|---------|
"""

    for mech_name in ["FeatureOnly", "TrajectoryOnly", "Hybrid_b1.0", "Hybrid_b2.0", "ObjectFile"]:
        vals = []
        for occ in ["0.0", "0.25", "0.5", "0.75", "1.0"]:
            rows = [r for r in occ_results if r["mechanism"] == mech_name and r["occlusion_ratio"] == occ]
            vals.append(rows[0]["identity_swap_only"] if rows else "nan")
        readme += f"| {mech_name} | {' | '.join(vals)} |\n"

    readme += f"""
## 6. Feature-Trajectory Conflict

| Mechanism | Conflict Type | Identity | Feat Correct | Traj Correct |
|-----------|--------------|----------|-------------|-------------|
"""

    for r in conflict_results:
        readme += f"| {r['mechanism']} | {r['conflict_type']} | {r['identity_under_conflict']} | {r['conflict_resolution_rate_feature_correct']} | {r['conflict_resolution_rate_trajectory_correct']} |\n"

    readme += f"""
## 7. Answers

### Q1: Is FeatureOnly fragile under feature noise?

FeatureOnly drop at noise=0.3: **{fmt(fo_noise_drop)}**

{"YES - FeatureOnly is fragile under feature noise." if fo_noise_drop > 0.2 else "NO - FeatureOnly is robust to moderate feature noise."}

### Q2: Is TrajectoryOnly misled by feature-trajectory conflict?

{"YES - TrajectoryOnly is misled when features contradict trajectory." if traj_misled else "NO - TrajectoryOnly ignores features and is not misled."}

### Q3: Is Hybrid just a weighted compromise?

{"YES - Hybrid produces intermediate results suggesting simple weighted averaging." if hybrid_is_compromise else "NO - Hybrid is not simply a weighted compromise."}

### Q4: Is MinimalObjectFile more stable under occlusion/no-feature/conflict?

Occlusion(0.75): ObjectFile vs FeatureOnly: {"ObjectFile more stable" if of_more_stable else "No clear advantage"}

{"YES - MinimalObjectFile shows advantages under stress conditions." if of_advantage else "NO - MinimalObjectFile does not show clear advantages over simpler mechanisms."}

### Q5: Final Recommendation

**{recommendation}**
"""

    with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print("\n" + "=" * 60)
    print("SVT-v4 FINAL SUMMARY")
    print("=" * 60)
    for r in mech_results:
        print(f"  {r['mechanism']}: swap={r['identity_swap_only']} feat_dep={r['feature_dependency_score']} traj_dep={r['trajectory_dependency_score']}")
    print(f"  FeatureOnly noise drop (0.3): {fmt(fo_noise_drop)}")
    print(f"  ObjectFile occlusion advantage: {of_more_stable}")
    print(f"  Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
