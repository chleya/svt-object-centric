"""
SVT-v5: Scaling Up — 3 Objects + Continuous Features + Stronger Trajectory Predictor
"""

import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v5_scaling_up"
SEED = 0
EPOCHS = 30

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

from data.generate_nonlinear_feature_ood import _generate_single_episode, _stack_episodes
from data.generate_nonlinear_feature_ood import generate_v3_dataset
from metrics.prediction_metrics import compute_prediction_metrics
from metrics.identity_breakdown import compute_identity_breakdown
from metrics.object_file_metrics import (
    compute_confidence_calibration, compute_conflict_gate_stats,
    compute_abstention_metrics,
)


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


def generate_swap_train(n_train=1000, num_objects=3, feature_dim=16,
                         swap_ratio=0.5, seed=0, force_type="attractor"):
    rng = np.random.RandomState(seed)
    episodes = []
    for _ in range(n_train):
        ep = _generate_single_episode(
            t_obs=10, t_pred=20, num_objects=num_objects, arena_size=64.0,
            feature_mode="feature_bearing", feature_dim=feature_dim,
            randomize_object_order=True,
            identity_test=True, swap_probability=swap_ratio,
            force_type=force_type, field_strength=0.5,
            damping=0.95, noise_std=0.1, rng=rng,
        )
        episodes.append(ep)
    return _stack_episodes(episodes, "feature_bearing")


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


def flip_future_features(future_features):
    if future_features is None:
        return None
    flipped = future_features.copy()
    flipped[:, :, 0, :], flipped[:, :, 1, :] = flipped[:, :, 1, :].copy(), flipped[:, :, 0, :].copy()
    return flipped


def compute_assignment_accuracy(pred_identity, true_identity):
    correct = (pred_identity == true_identity).all(axis=1)
    return float(correct.mean())


def make_occlusion_mask(future_positions, occlusion_ratio):
    if occlusion_ratio <= 0:
        return None
    B, T, N, _ = future_positions.shape
    mask = np.zeros((B, T, N), dtype=bool)
    n_occluded = int(T * occlusion_ratio)
    start = (T - n_occluded) // 2
    mask[:, start:start + n_occluded, :] = True
    return mask


def evaluate_learned_model(model, test_data, obs_feat_override=None, fut_feat_override=None):
    obs_pos = test_data["observed_positions"]
    obs_feat = obs_feat_override if obs_feat_override is not None else test_data.get("object_features_obs")
    fut_feat = fut_feat_override if fut_feat_override is not None else test_data.get("object_features_fut")

    pred_future = model.predict_future(obs_pos, obs_feat)
    if isinstance(pred_future, torch.Tensor):
        pred_future = pred_future.cpu().numpy()

    pred_identity = model.predict_identity(obs_pos, obs_feat, future_features=fut_feat)
    if isinstance(pred_identity, torch.Tensor):
        pred_identity = pred_identity.cpu().numpy()

    true_identity = test_data["identity_labels"]
    asgn_acc = compute_assignment_accuracy(pred_identity, true_identity)
    breakdown = compute_identity_breakdown(pred_identity, true_identity)
    return breakdown, asgn_acc


def evaluate_trajectory_only(model, test_data):
    obs_pos = test_data["observed_positions"]
    fut_pos = test_data["future_positions"]

    pred_identity = model.predict_identity(obs_pos, future_positions=fut_pos)
    if isinstance(pred_identity, torch.Tensor):
        pred_identity = pred_identity.cpu().numpy()

    true_identity = test_data["identity_labels"]
    asgn_acc = compute_assignment_accuracy(pred_identity, true_identity)
    breakdown = compute_identity_breakdown(pred_identity, true_identity)
    return breakdown, asgn_acc


def evaluate_obj_file(mech, test_data, obs_feat_override=None,
                       fut_feat_override=None, occlusion_mask=None):
    obs_pos = test_data["observed_positions"]
    obs_feat = obs_feat_override if obs_feat_override is not None else test_data.get("object_features_obs")
    fut_feat = fut_feat_override if fut_feat_override is not None else test_data.get("object_features_fut")
    fut_pos = test_data["future_positions"]

    result = mech.predict_identity(obs_pos, obs_feat, fut_pos, fut_feat,
                                   occlusion_mask=occlusion_mask, return_conflict_info=True)

    pred_identity = result[0]
    confidences = result[1] if len(result) > 1 else []
    sources = result[2] if len(result) > 2 else []
    abstain_flags = result[3] if len(result) > 3 else [False] * len(pred_identity)

    true_identity = test_data["identity_labels"]
    asgn_acc = compute_assignment_accuracy(pred_identity, true_identity)
    breakdown = compute_identity_breakdown(pred_identity, true_identity)
    return breakdown, asgn_acc, confidences, sources, abstain_flags


def extract_confidences(confidences, true_identity):
    total_confs = []
    for ep_conf in confidences:
        if isinstance(ep_conf, dict):
            if 'final_confidence' in ep_conf:
                total_confs.append(ep_conf['final_confidence'])
            elif 'total_confs' in ep_conf:
                total_confs.append(max(ep_conf['total_confs']))
            else:
                total_confs.append(1.0)
        else:
            total_confs.append(1.0)
    if len(total_confs) != len(true_identity):
        total_confs = np.ones(len(true_identity))
    else:
        total_confs = np.array(total_confs)
    return total_confs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SVT-v5: Scaling Up — 3 Objects + Continuous Features")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("PyTorch unavailable. Exiting.")
        return

    from models.feature_similarity_models import FeatureOnlyAssignmentHead, HybridTrajectoryFeatureAssignmentHead
    from models.object_file_models import (
        TrajectoryOnlyAssignment, MultiTaskTrajectoryPredictor,
        MinimalObjectFileMechanism, ConflictFirstObjectFile,
    )
    from utils.torch_training import train_model

    # =========================================================================
    # Experiment 1: 2 objects + one-hot features (baseline replication)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Experiment 1: 2 Objects + One-Hot Features (Baseline)")
    print("=" * 60)

    eval_ds_2 = generate_v3_dataset(
        n_train=1000, n_test=200, num_objects=2, feature_dim=2,
        feature_mode="feature_bearing",
        force_train_type="attractor", force_test_type="vortex",
        randomize_object_order=True, disjoint_init_split=True, seed=SEED,
    )
    train_2 = generate_swap_train(n_train=1000, num_objects=2, feature_dim=2, seed=SEED)
    swap_2 = eval_ds_2["identity_test_swap_only"]
    clean_2 = eval_ds_2["clean_test_id"]

    model_fo_2 = FeatureOnlyAssignmentHead(num_objects=2, feature_dim=2)
    train_model(model_fo_2, train_2, val_data=clean_2, epochs=EPOCHS,
                batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=True, uses_future_features=True, verbose=False)

    model_traj_2 = TrajectoryOnlyAssignment(num_objects=2)
    train_model(model_traj_2, train_2, val_data=clean_2, epochs=EPOCHS,
                batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=False, uses_future_features=False, verbose=False)

    model_mtraj_2 = MultiTaskTrajectoryPredictor(num_objects=2)
    train_model(model_mtraj_2, train_2, val_data=clean_2, epochs=EPOCHS,
                batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=False, uses_future_features=False, verbose=False)

    obj_v42_2 = ConflictFirstObjectFile(traj_model=model_traj_2, strategy="margin_gated",
                                         num_objects=2, feature_dim=2, traj_margin_advantage=1.2)
    obj_v42_mt_2 = ConflictFirstObjectFile(traj_model=model_mtraj_2, strategy="margin_gated",
                                            num_objects=2, feature_dim=2, traj_margin_advantage=1.2)

    baseline_results = []

    for name, mtype, model in [
        ("FO_2obj_onehot", "learned", model_fo_2),
        ("Traj_2obj_onehot", "traj_only", model_traj_2),
        ("MTTraj_2obj_onehot", "traj_only", model_mtraj_2),
        ("CF_margin_2obj_onehot", "obj_file", obj_v42_2),
        ("CF_margin_MT_2obj_onehot", "obj_file", obj_v42_mt_2),
    ]:
        if mtype == "learned":
            bd, _ = evaluate_learned_model(model, swap_2)
        elif mtype == "traj_only":
            bd, _ = evaluate_trajectory_only(model, swap_2)
        elif mtype == "obj_file":
            bd, _, _, _, _ = evaluate_obj_file(model, swap_2)

        obs_shuf = shuffle_features(swap_2["object_features_obs"], seed=SEED)
        fut_shuf = shuffle_features(swap_2["object_features_fut"], seed=SEED + 1000)
        obs_zero = zero_features(swap_2["object_features_obs"])
        fut_zero = zero_features(swap_2["object_features_fut"])

        if mtype == "learned":
            bd_shuf, _ = evaluate_learned_model(model, swap_2, obs_feat_override=obs_shuf, fut_feat_override=fut_shuf)
            bd_zero, _ = evaluate_learned_model(model, swap_2, obs_feat_override=obs_zero, fut_feat_override=fut_zero)
        elif mtype == "traj_only":
            bd_shuf = bd
            bd_zero = bd
        elif mtype == "obj_file":
            bd_shuf, _, _, _, _ = evaluate_obj_file(model, swap_2, obs_feat_override=obs_shuf, fut_feat_override=fut_shuf)
            bd_zero, _, _, _, _ = evaluate_obj_file(model, swap_2, obs_feat_override=obs_zero, fut_feat_override=fut_zero)

        feat_dep = bd["identity_swap_only"] - bd_shuf["identity_swap_only"]
        traj_dep = bd_zero["identity_swap_only"]

        baseline_results.append({
            "experiment": "2obj_onehot",
            "mechanism": name,
            "swap_only": fmt(bd["identity_swap_only"]),
            "feat_dep": fmt(feat_dep),
            "traj_dep": fmt(traj_dep),
        })
        print(f"  {name}: swap={fmt(bd['identity_swap_only'])} feat_dep={fmt(feat_dep)} traj_dep={fmt(traj_dep)}")

    # =========================================================================
    # Experiment 2: 3 objects + continuous features
    # =========================================================================
    print("\n" + "=" * 60)
    print("Experiment 2: 3 Objects + Continuous Features (16-dim)")
    print("=" * 60)

    NUM_OBJ_3 = 3
    FEAT_DIM_16 = 16

    eval_ds_3 = generate_v3_dataset(
        n_train=1000, n_test=200, num_objects=NUM_OBJ_3, feature_dim=FEAT_DIM_16,
        feature_mode="feature_bearing",
        force_train_type="attractor", force_test_type="vortex",
        randomize_object_order=True, disjoint_init_split=True, seed=SEED,
    )
    train_3 = generate_swap_train(n_train=1000, num_objects=NUM_OBJ_3,
                                   feature_dim=FEAT_DIM_16, seed=SEED)
    swap_3 = eval_ds_3["identity_test_swap_only"]
    clean_3 = eval_ds_3["clean_test_id"]

    print("  Training FeatureOnly (3obj, 16dim)...")
    model_fo_3 = FeatureOnlyAssignmentHead(num_objects=NUM_OBJ_3, feature_dim=FEAT_DIM_16)
    train_model(model_fo_3, train_3, val_data=clean_3, epochs=EPOCHS,
                batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=True, uses_future_features=True, verbose=False)

    print("  Training TrajectoryOnly (3obj)...")
    model_traj_3 = TrajectoryOnlyAssignment(num_objects=NUM_OBJ_3)
    train_model(model_traj_3, train_3, val_data=clean_3, epochs=EPOCHS,
                batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=False, uses_future_features=False, verbose=False)

    print("  Training MultiTaskTrajectory (3obj)...")
    model_mtraj_3 = MultiTaskTrajectoryPredictor(num_objects=NUM_OBJ_3)
    train_model(model_mtraj_3, train_3, val_data=clean_3, epochs=EPOCHS,
                batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=False, uses_future_features=False, verbose=False)

    obj_v42_traj_3 = ConflictFirstObjectFile(traj_model=model_traj_3, strategy="margin_gated",
                                              num_objects=NUM_OBJ_3, feature_dim=FEAT_DIM_16,
                                              traj_margin_advantage=1.2)
    obj_v42_mtraj_3 = ConflictFirstObjectFile(traj_model=model_mtraj_3, strategy="margin_gated",
                                               num_objects=NUM_OBJ_3, feature_dim=FEAT_DIM_16,
                                               traj_margin_advantage=1.2)

    scaled_results = []

    for name, mtype, model in [
        ("FO_3obj_cont16", "learned", model_fo_3),
        ("Traj_3obj", "traj_only", model_traj_3),
        ("MTTraj_3obj", "traj_only", model_mtraj_3),
        ("CF_margin_traj_3obj_cont16", "obj_file", obj_v42_traj_3),
        ("CF_margin_MT_3obj_cont16", "obj_file", obj_v42_mtraj_3),
    ]:
        if mtype == "learned":
            bd, _ = evaluate_learned_model(model, swap_3)
        elif mtype == "traj_only":
            bd, _ = evaluate_trajectory_only(model, swap_3)
        elif mtype == "obj_file":
            bd, _, _, _, _ = evaluate_obj_file(model, swap_3)

        obs_shuf = shuffle_features(swap_3["object_features_obs"], seed=SEED)
        fut_shuf = shuffle_features(swap_3["object_features_fut"], seed=SEED + 1000)
        obs_zero = zero_features(swap_3["object_features_obs"])
        fut_zero = zero_features(swap_3["object_features_fut"])

        if mtype == "learned":
            bd_shuf, _ = evaluate_learned_model(model, swap_3, obs_feat_override=obs_shuf, fut_feat_override=fut_shuf)
            bd_zero, _ = evaluate_learned_model(model, swap_3, obs_feat_override=obs_zero, fut_feat_override=fut_zero)
        elif mtype == "traj_only":
            bd_shuf = bd
            bd_zero = bd
        elif mtype == "obj_file":
            bd_shuf, _, _, _, _ = evaluate_obj_file(model, swap_3, obs_feat_override=obs_shuf, fut_feat_override=fut_shuf)
            bd_zero, _, _, _, _ = evaluate_obj_file(model, swap_3, obs_feat_override=obs_zero, fut_feat_override=fut_zero)

        feat_dep = bd["identity_swap_only"] - bd_shuf["identity_swap_only"]
        traj_dep = bd_zero["identity_swap_only"]

        scaled_results.append({
            "experiment": "3obj_cont16",
            "mechanism": name,
            "swap_only": fmt(bd["identity_swap_only"]),
            "feat_dep": fmt(feat_dep),
            "traj_dep": fmt(traj_dep),
        })
        print(f"  {name}: swap={fmt(bd['identity_swap_only'])} feat_dep={fmt(feat_dep)} traj_dep={fmt(traj_dep)}")

    # =========================================================================
    # Conflict Test for 3 objects + continuous features
    # =========================================================================
    print("\n=== Conflict Test: 3 Objects + Continuous Features ===")

    is_swap_clean_3 = clean_3["is_swap"]
    no_swap_idx_3 = np.where(~is_swap_clean_3)[0]

    conflict_results_3 = []

    if len(no_swap_idx_3) > 0:
        no_swap_data_3 = {k: v[no_swap_idx_3] for k, v in clean_3.items()}
        fut_flipped_3 = flip_future_features(no_swap_data_3["object_features_fut"])

        true_id_3 = no_swap_data_3["identity_labels"]
        is_swap_3 = true_id_3[:, 0] != np.arange(NUM_OBJ_3)[0]
        for i in range(1, NUM_OBJ_3):
            is_swap_3 = is_swap_3 | (true_id_3[:, i] != np.arange(NUM_OBJ_3)[i])
        n_no_swap_3 = int((~is_swap_3).sum())

        for name, mtype, model in [
            ("FO_3obj_cont16", "learned", model_fo_3),
            ("Traj_3obj", "traj_only", model_traj_3),
            ("MTTraj_3obj", "traj_only", model_mtraj_3),
            ("CF_margin_traj_3obj_cont16", "obj_file", obj_v42_traj_3),
            ("CF_margin_MT_3obj_cont16", "obj_file", obj_v42_mtraj_3),
        ]:
            if mtype == "learned":
                pred_id = model.predict_identity(
                    no_swap_data_3["observed_positions"],
                    no_swap_data_3["object_features_obs"],
                    future_features=fut_flipped_3)
                if isinstance(pred_id, torch.Tensor):
                    pred_id = pred_id.cpu().numpy()
            elif mtype == "traj_only":
                pred_id = model.predict_identity(
                    no_swap_data_3["observed_positions"],
                    future_positions=no_swap_data_3["future_positions"])
                if isinstance(pred_id, torch.Tensor):
                    pred_id = pred_id.cpu().numpy()
            elif mtype == "obj_file":
                result = model.predict_identity(
                    no_swap_data_3["observed_positions"],
                    no_swap_data_3["object_features_obs"],
                    no_swap_data_3["future_positions"],
                    fut_flipped_3, return_conflict_info=True)
                pred_id = result[0]

            correct = (pred_id == true_id_3).all(axis=1)
            identity_acc = float(correct.mean())

            if n_no_swap_3 > 0:
                traj_correct_rate = float(correct[~is_swap_3].mean())
            else:
                traj_correct_rate = float("nan")

            conflict_results_3.append({
                "mechanism": name,
                "conflict_identity": fmt(identity_acc),
                "traj_correct_rate": fmt(traj_correct_rate),
            })
            print(f"    {name}: conflict_acc={fmt(identity_acc)} traj_correct={fmt(traj_correct_rate)}")

    # Also do conflict test for 2obj baseline
    print("\n=== Conflict Test: 2 Objects + One-Hot Features (Baseline) ===")

    is_swap_clean_2 = clean_2["is_swap"]
    no_swap_idx_2 = np.where(~is_swap_clean_2)[0]

    conflict_results_2 = []

    if len(no_swap_idx_2) > 0:
        no_swap_data_2 = {k: v[no_swap_idx_2] for k, v in clean_2.items()}
        fut_flipped_2 = flip_future_features(no_swap_data_2["object_features_fut"])

        true_id_2 = no_swap_data_2["identity_labels"]
        is_swap_2 = true_id_2[:, 0] != 0
        n_no_swap_2 = int((~is_swap_2).sum())

        for name, mtype, model in [
            ("FO_2obj_onehot", "learned", model_fo_2),
            ("Traj_2obj_onehot", "traj_only", model_traj_2),
            ("CF_margin_2obj_onehot", "obj_file", obj_v42_2),
        ]:
            if mtype == "learned":
                pred_id = model.predict_identity(
                    no_swap_data_2["observed_positions"],
                    no_swap_data_2["object_features_obs"],
                    future_features=fut_flipped_2)
                if isinstance(pred_id, torch.Tensor):
                    pred_id = pred_id.cpu().numpy()
            elif mtype == "traj_only":
                pred_id = model.predict_identity(
                    no_swap_data_2["observed_positions"],
                    future_positions=no_swap_data_2["future_positions"])
                if isinstance(pred_id, torch.Tensor):
                    pred_id = pred_id.cpu().numpy()
            elif mtype == "obj_file":
                result = model.predict_identity(
                    no_swap_data_2["observed_positions"],
                    no_swap_data_2["object_features_obs"],
                    no_swap_data_2["future_positions"],
                    fut_flipped_2, return_conflict_info=True)
                pred_id = result[0]

            correct = (pred_id == true_id_2).all(axis=1)
            identity_acc = float(correct.mean())

            if n_no_swap_2 > 0:
                traj_correct_rate = float(correct[~is_swap_2].mean())
            else:
                traj_correct_rate = float("nan")

            conflict_results_2.append({
                "mechanism": name,
                "conflict_identity": fmt(identity_acc),
                "traj_correct_rate": fmt(traj_correct_rate),
            })
            print(f"    {name}: conflict_acc={fmt(identity_acc)} traj_correct={fmt(traj_correct_rate)}")

    # =========================================================================
    # Save all results
    # =========================================================================
    all_results = baseline_results + scaled_results
    save_csv(all_results, "mechanism_comparison.csv",
             ["experiment", "mechanism", "swap_only", "feat_dep", "traj_dep"])

    all_conflict = [{"experiment": "2obj_onehot", **r} for r in conflict_results_2] + \
                   [{"experiment": "3obj_cont16", **r} for r in conflict_results_3]
    save_csv(all_conflict, "conflict_results.csv",
             ["experiment", "mechanism", "conflict_identity", "traj_correct_rate"])

    # =========================================================================
    # README
    # =========================================================================
    readme = f"""# SVT-v5: Scaling Up — 3 Objects + Continuous Features

## 1. Purpose

Test whether the diagnostic chain (clean feature matching ≠ object-file) holds when scaling from 2 objects + one-hot features to 3 objects + 16-dim continuous features. Also test whether a multi-task trajectory predictor (pos+vel+accel) improves ObjectFile performance.

## 2. Mechanism Comparison

| Experiment | Mechanism | Swap-Only | Feat Dep | Traj Dep |
|-----------|-----------|-----------|----------|----------|
"""
    for r in all_results:
        readme += f"| {r['experiment']} | {r['mechanism']} | {r['swap_only']} | {r['feat_dep']} | {r['traj_dep']} |\n"

    readme += f"""
## 3. Conflict Test

| Experiment | Mechanism | Conflict Identity | Traj Correct |
|-----------|-----------|------------------|-------------|
"""
    for r in all_conflict:
        readme += f"| {r['experiment']} | {r['mechanism']} | {r['conflict_identity']} | {r['traj_correct_rate']} |\n"

    readme += """
## 4. Key Questions

1. Does FeatureOnly still fail under conflict with 3 objects + continuous features?
2. Does ConflictFirstObjectFile maintain correct structural bias with 3 objects?
3. Does multi-task trajectory predictor improve ObjectFile performance?
4. Does the diagnostic chain generalize?

## 5. Expected Outcomes

- FeatureOnly should still fail under conflict (diagnostic chain holds)
- ConflictFirstObjectFile should maintain structural bias (possibly with lower absolute numbers)
- Multi-task trajectory predictor should improve swap-only identity
- The trade-off between normal performance and conflict resolution should persist
"""

    with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print("\n" + "=" * 60)
    print("SVT-v5 FINAL SUMMARY")
    print("=" * 60)
    print("\n--- 2 Objects + One-Hot (Baseline) ---")
    for r in baseline_results:
        print(f"  {r['mechanism']}: swap={r['swap_only']} feat_dep={r['feat_dep']} traj_dep={r['traj_dep']}")
    print("\n--- 3 Objects + Continuous 16-dim ---")
    for r in scaled_results:
        print(f"  {r['mechanism']}: swap={r['swap_only']} feat_dep={r['feat_dep']} traj_dep={r['traj_dep']}")
    print("\n--- Conflict: 2obj ---")
    for r in conflict_results_2:
        print(f"  {r['mechanism']}: conflict={r['conflict_identity']} traj_correct={r['traj_correct_rate']}")
    print("\n--- Conflict: 3obj ---")
    for r in conflict_results_3:
        print(f"  {r['mechanism']}: conflict={r['conflict_identity']} traj_correct={r['traj_correct_rate']}")


if __name__ == "__main__":
    main()
