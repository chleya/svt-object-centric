"""
SVT-v4.3: Trajectory-Robust ObjectFile
"""

import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v4_3_trajectory_robust_object_file"
SEED = 0
SWAP_RATIO = 0.3
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
    compute_abstention_metrics, compute_conflict_detection_accuracy,
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


def augment_trajectory_data(data, noise_std=0.5, n_rotations=4, seed=42):
    rng = np.random.RandomState(seed)
    all_obs = [data["observed_positions"]]
    all_fut = [data["future_positions"]]
    all_ids = [data["identity_labels"]]
    all_feat_obs = [data["object_features_obs"]]
    all_feat_fut = [data["object_features_fut"]]
    all_swap = [data["is_swap"]]

    for _ in range(n_rotations):
        noise_obs = data["observed_positions"] + rng.randn(*data["observed_positions"].shape) * noise_std
        noise_fut = data["future_positions"] + rng.randn(*data["future_positions"].shape) * noise_std
        all_obs.append(noise_obs.astype(np.float32))
        all_fut.append(noise_fut.astype(np.float32))
        all_ids.append(data["identity_labels"].copy())
        all_feat_obs.append(data["object_features_obs"].copy())
        all_feat_fut.append(data["object_features_fut"].copy())
        all_swap.append(data["is_swap"].copy())

    return {
        "observed_positions": np.concatenate(all_obs, axis=0),
        "future_positions": np.concatenate(all_fut, axis=0),
        "identity_labels": np.concatenate(all_ids, axis=0),
        "object_features_obs": np.concatenate(all_feat_obs, axis=0),
        "object_features_fut": np.concatenate(all_feat_fut, axis=0),
        "is_swap": np.concatenate(all_swap, axis=0),
    }


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
    pred_metrics = compute_prediction_metrics(pred_future, test_data["future_positions"])

    pred_identity = model.predict_identity(obs_pos, obs_feat, future_features=fut_feat)
    if isinstance(pred_identity, torch.Tensor):
        pred_identity = pred_identity.cpu().numpy()

    true_identity = test_data["identity_labels"]
    asgn_acc = compute_assignment_accuracy(pred_identity, true_identity)
    breakdown = compute_identity_breakdown(pred_identity, true_identity)

    return pred_metrics, breakdown, asgn_acc


def evaluate_trajectory_only(model, test_data):
    obs_pos = test_data["observed_positions"]
    fut_pos = test_data["future_positions"]

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


def evaluate_obj_file_v4(mech, test_data, obs_feat_override=None,
                          fut_feat_override=None, occlusion_mask=None):
    obs_pos = test_data["observed_positions"]
    obs_feat = obs_feat_override if obs_feat_override is not None else test_data.get("object_features_obs")
    fut_feat = fut_feat_override if fut_feat_override is not None else test_data.get("object_features_fut")
    fut_pos = test_data["future_positions"]

    pred_identity = mech.predict_identity(obs_pos, obs_feat, fut_pos, fut_feat,
                                          occlusion_mask=occlusion_mask)
    true_identity = test_data["identity_labels"]
    asgn_acc = compute_assignment_accuracy(pred_identity, true_identity)
    breakdown = compute_identity_breakdown(pred_identity, true_identity)
    return breakdown, asgn_acc


def evaluate_obj_file_with_conflict(mech, test_data, obs_feat_override=None,
                                     fut_feat_override=None, occlusion_mask=None):
    obs_pos = test_data["observed_positions"]
    obs_feat = obs_feat_override if obs_feat_override is not None else test_data.get("object_features_obs")
    fut_feat = fut_feat_override if fut_feat_override is not None else test_data.get("object_features_fut")
    fut_pos = test_data["future_positions"]

    result = mech.predict_identity(
        obs_pos, obs_feat, fut_pos, fut_feat,
        occlusion_mask=occlusion_mask, return_conflict_info=True)

    if len(result) == 4:
        pred_identity, confidences, sources, abstain_flags = result
    elif len(result) == 3:
        pred_identity, confidences, sources = result
        abstain_flags = [False] * (pred_identity.shape[0] if pred_identity.ndim > 1 else 1)
    else:
        pred_identity = result
        confidences = []
        sources = []
        abstain_flags = [False] * (pred_identity.shape[0] if pred_identity.ndim > 1 else 1)

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
    print("SVT-v4.3: Trajectory-Robust ObjectFile")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("PyTorch unavailable. Exiting.")
        return

    from models.feature_similarity_models import FeatureOnlyAssignmentHead, HybridTrajectoryFeatureAssignmentHead
    from models.object_file_models import (
        TrajectoryOnlyAssignment, MinimalObjectFileMechanism,
        ImprovedObjectFile, ConflictFirstObjectFile,
        TrajectoryRobustObjectFile,
    )
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

    print("\n=== Training Models ===")

    print("  Training FeatureOnly...")
    model_fo = FeatureOnlyAssignmentHead(identity_weight=1.0)
    train_model(model_fo, train_data, val_data=clean_test,
                epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=True, uses_future_features=True, verbose=False)

    print("  Training TrajectoryOnly (standard)...")
    model_traj = TrajectoryOnlyAssignment()
    train_model(model_traj, train_data, val_data=clean_test,
                epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=False, uses_future_features=False, verbose=False)

    print("  Training TrajectoryOnly (augmented)...")
    aug_train = augment_trajectory_data(train_data, noise_std=0.5, n_rotations=4, seed=SEED)
    model_traj_aug = TrajectoryOnlyAssignment()
    train_model(model_traj_aug, aug_train, val_data=clean_test,
                epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=False, uses_future_features=False, verbose=False)

    print("  Training Hybrid(beta=1.0)...")
    model_hybrid1 = HybridTrajectoryFeatureAssignmentHead(identity_weight=1.0, beta=1.0)
    train_model(model_hybrid1, train_data, val_data=clean_test,
                epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=True, uses_future_features=True, verbose=False)

    obj_file_v4 = MinimalObjectFileMechanism(feature_weight=1.0, traj_weight=1.0)
    obj_file_v41 = ImprovedObjectFile(traj_model=model_traj)
    obj_file_v42_margin = ConflictFirstObjectFile(
        traj_model=model_traj, strategy="margin_gated", traj_margin_advantage=1.2)

    obj_file_v43_aware = TrajectoryRobustObjectFile(
        traj_model=model_traj, conflict_strategy="approach_aware",
        approach_threshold=5.0, n_voting_steps=5, temporal_decay=0.9)
    obj_file_v43_aware_aug = TrajectoryRobustObjectFile(
        traj_model=model_traj_aug, conflict_strategy="approach_aware",
        approach_threshold=5.0, n_voting_steps=5, temporal_decay=0.9)
    obj_file_v43_veto = TrajectoryRobustObjectFile(
        traj_model=model_traj, conflict_strategy="approach_veto",
        approach_threshold=5.0, n_voting_steps=5, temporal_decay=0.9)
    obj_file_v43_veto_aug = TrajectoryRobustObjectFile(
        traj_model=model_traj_aug, conflict_strategy="approach_veto",
        approach_threshold=5.0, n_voting_steps=5, temporal_decay=0.9)

    all_mechanisms = [
        ("FeatureOnly", "learned", model_fo),
        ("TrajectoryOnly", "traj_only", model_traj),
        ("TrajectoryOnly_aug", "traj_only", model_traj_aug),
        ("Hybrid_b1.0", "learned", model_hybrid1),
        ("ObjectFile_v4", "obj_file_v4", obj_file_v4),
        ("ImprovedObjectFile_v4.1", "obj_file_v41", obj_file_v41),
        ("ConflictFirst_margin_v4.2", "obj_file_v42", obj_file_v42_margin),
        ("TrajRobust_aware_v4.3", "obj_file_v43", obj_file_v43_aware),
        ("TrajRobust_aware_aug_v4.3", "obj_file_v43", obj_file_v43_aware_aug),
        ("TrajRobust_veto_v4.3", "obj_file_v43", obj_file_v43_veto),
        ("TrajRobust_veto_aug_v4.3", "obj_file_v43", obj_file_v43_veto_aug),
    ]

    # =========================================================================
    # Part 1: Mechanism Comparison
    # =========================================================================
    print("\n=== Mechanism Comparison ===")

    mech_results = []

    for mech_name, mech_type, mech in all_mechanisms:
        if mech_type == "learned":
            _, bd, _ = evaluate_learned_model(mech, swap_test)
        elif mech_type == "traj_only":
            _, bd, _ = evaluate_trajectory_only(mech, swap_test)
        elif mech_type == "obj_file_v4":
            bd, _ = evaluate_obj_file_v4(mech, swap_test)
        elif mech_type in ("obj_file_v41", "obj_file_v42", "obj_file_v43"):
            bd, _, _, _, _ = evaluate_obj_file_with_conflict(mech, swap_test)

        obs_shuf = shuffle_features(swap_test["object_features_obs"], seed=SEED)
        fut_shuf = shuffle_features(swap_test["object_features_fut"], seed=SEED + 1000)
        obs_zero = zero_features(swap_test["object_features_obs"])
        fut_zero = zero_features(swap_test["object_features_fut"])

        if mech_type == "learned":
            _, bd_shuf, _ = evaluate_learned_model(mech, swap_test,
                                                    obs_feat_override=obs_shuf, fut_feat_override=fut_shuf)
            _, bd_zero, _ = evaluate_learned_model(mech, swap_test,
                                                    obs_feat_override=obs_zero, fut_feat_override=fut_zero)
        elif mech_type == "traj_only":
            bd_shuf = bd
            bd_zero = bd
        elif mech_type == "obj_file_v4":
            bd_shuf, _ = evaluate_obj_file_v4(mech, swap_test,
                                               obs_feat_override=obs_shuf, fut_feat_override=fut_shuf)
            bd_zero, _ = evaluate_obj_file_v4(mech, swap_test,
                                               obs_feat_override=obs_zero, fut_feat_override=fut_zero)
        elif mech_type in ("obj_file_v41", "obj_file_v42", "obj_file_v43"):
            bd_shuf, _, _, _, _ = evaluate_obj_file_with_conflict(mech, swap_test,
                                                                    obs_feat_override=obs_shuf, fut_feat_override=fut_shuf)
            bd_zero, _, _, _, _ = evaluate_obj_file_with_conflict(mech, swap_test,
                                                                    obs_feat_override=obs_zero, fut_feat_override=fut_zero)

        feat_dep = bd["identity_swap_only"] - bd_shuf["identity_swap_only"]
        traj_dep = bd_zero["identity_swap_only"]

        mech_results.append({
            "mechanism": mech_name,
            "identity_swap_only": fmt(bd["identity_swap_only"]),
            "identity_overall": fmt(bd["identity_overall"]),
            "feature_dependency_score": fmt(feat_dep),
            "trajectory_dependency_score": fmt(traj_dep),
        })
        print(f"  {mech_name}: swap={fmt(bd['identity_swap_only'])} feat_dep={fmt(feat_dep)} traj_dep={fmt(traj_dep)}")

    save_csv(mech_results, "mechanism_comparison.csv",
             ["mechanism", "identity_swap_only", "identity_overall",
              "feature_dependency_score", "trajectory_dependency_score"])

    # =========================================================================
    # Part 2: Confidence Calibration
    # =========================================================================
    print("\n=== Confidence Calibration ===")

    cal_results = []

    obj_file_mechs = [
        ("ObjectFile_v4", "obj_file_v4", obj_file_v4),
        ("ImprovedObjectFile_v4.1", "obj_file_v41", obj_file_v41),
        ("ConflictFirst_margin_v4.2", "obj_file_v42", obj_file_v42_margin),
        ("TrajRobust_aware_v4.3", "obj_file_v43", obj_file_v43_aware),
        ("TrajRobust_aware_aug_v4.3", "obj_file_v43", obj_file_v43_aware_aug),
        ("TrajRobust_veto_v4.3", "obj_file_v43", obj_file_v43_veto),
        ("TrajRobust_veto_aug_v4.3", "obj_file_v43", obj_file_v43_veto_aug),
    ]

    for mech_name, mech_type, mech in obj_file_mechs:
        if mech_type == "obj_file_v4":
            bd, asgn_acc = evaluate_obj_file_v4(mech, swap_test)
            pred_identity = mech.predict_identity(
                swap_test["observed_positions"], swap_test["object_features_obs"],
                swap_test["future_positions"], swap_test["object_features_fut"])
            total_confs = np.ones(len(swap_test["identity_labels"]))
            sources = []
            abstain_flags = np.zeros(len(swap_test["identity_labels"]), dtype=bool)
        else:
            bd, asgn_acc, confidences, sources, abstain_flags_raw = evaluate_obj_file_with_conflict(mech, swap_test)
            pred_identity = mech.predict_identity(
                swap_test["observed_positions"], swap_test["object_features_obs"],
                swap_test["future_positions"], swap_test["object_features_fut"],
                return_conflict_info=True)
            if isinstance(pred_identity, tuple):
                pred_identity = pred_identity[0]
            total_confs = extract_confidences(confidences, swap_test["identity_labels"])
            abstain_flags = np.array(abstain_flags_raw) if isinstance(abstain_flags_raw, list) else abstain_flags_raw

        true_identity = swap_test["identity_labels"]
        cal = compute_confidence_calibration(pred_identity, true_identity, total_confs)

        if sources:
            gate = compute_conflict_gate_stats(sources, pred_identity, true_identity)
        else:
            gate = {k: float('nan') for k in ['chosen_source_feature_rate', 'chosen_source_trajectory_rate',
                                                'chosen_source_uncertain_rate', 'chosen_source_agreement_rate']}

        abs_metrics = compute_abstention_metrics(pred_identity, true_identity, abstain_flags)

        cal_results.append({
            "mechanism": mech_name,
            "identity_swap_only": fmt(bd["identity_swap_only"]),
            "avg_confidence_correct": fmt(cal['avg_confidence_correct']),
            "avg_confidence_incorrect": fmt(cal['avg_confidence_incorrect']),
            "confidence_calibration_error": fmt(cal['confidence_calibration_error']),
            "chosen_source_feature_rate": fmt(gate['chosen_source_feature_rate']),
            "chosen_source_trajectory_rate": fmt(gate['chosen_source_trajectory_rate']),
            "chosen_source_uncertain_rate": fmt(gate['chosen_source_uncertain_rate']),
            "chosen_source_agreement_rate": fmt(gate['chosen_source_agreement_rate']),
            "abstention_rate": fmt(abs_metrics['abstention_rate']),
            "accuracy_when_not_abstaining": fmt(abs_metrics['accuracy_when_not_abstaining']),
        })

        print(f"  {mech_name}: swap={fmt(bd['identity_swap_only'])} cal_err={fmt(cal['confidence_calibration_error'])} "
              f"conf_corr={fmt(cal['avg_confidence_correct'])} conf_inc={fmt(cal['avg_confidence_incorrect'])} "
              f"feat={fmt(gate['chosen_source_feature_rate'])} traj={fmt(gate['chosen_source_trajectory_rate'])} "
              f"uncertain={fmt(gate['chosen_source_uncertain_rate'])} abstain={fmt(abs_metrics['abstention_rate'])}")

    save_csv(cal_results, "confidence_calibration.csv",
             ["mechanism", "identity_swap_only", "avg_confidence_correct",
              "avg_confidence_incorrect", "confidence_calibration_error",
              "chosen_source_feature_rate", "chosen_source_trajectory_rate",
              "chosen_source_uncertain_rate", "chosen_source_agreement_rate",
              "abstention_rate", "accuracy_when_not_abstaining"])

    # =========================================================================
    # Part 3: Feature-Trajectory Conflict
    # =========================================================================
    print("\n=== Feature-Trajectory Conflict ===")

    is_swap_clean = clean_test["is_swap"]
    no_swap_idx = np.where(~is_swap_clean)[0]
    swap_idx = np.where(is_swap_clean)[0]

    conflict_results = []

    if len(no_swap_idx) > 0:
        print(f"  Type A: feature_wrong (n={len(no_swap_idx)})")
        no_swap_data = {k: v[no_swap_idx] for k, v in clean_test.items()}
        fut_flipped = flip_future_features(no_swap_data["object_features_fut"])

        true_id = no_swap_data["identity_labels"]
        is_actually_swap = true_id[:, 0] != 0
        n_actual_no_swap = int((~is_actually_swap).sum())

        for mech_name, mech_type, mech in all_mechanisms:
            pred_identity = None
            sources_ep = []
            abstain_ep = []

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
            elif mech_type == "obj_file_v4":
                pred_identity = mech.predict_identity(
                    no_swap_data["observed_positions"],
                    no_swap_data["object_features_obs"],
                    no_swap_data["future_positions"], fut_flipped)
            elif mech_type in ("obj_file_v41", "obj_file_v42", "obj_file_v43"):
                result = mech.predict_identity(
                    no_swap_data["observed_positions"],
                    no_swap_data["object_features_obs"],
                    no_swap_data["future_positions"],
                    fut_flipped, return_conflict_info=True)
                if len(result) == 4:
                    pred_identity, conf, sources_ep, abstain_ep = result
                elif len(result) == 3:
                    pred_identity, conf, sources_ep = result
                    abstain_ep = []
                else:
                    pred_identity = result
                    sources_ep = []
                    abstain_ep = []

            correct = (pred_identity == true_id).all(axis=1)
            identity_acc = float(correct.mean())

            pred_is_swap = pred_identity[:, 0] != 0
            if n_actual_no_swap > 0:
                traj_correct_rate = float((~pred_is_swap[~is_actually_swap]).sum() / n_actual_no_swap)
                feat_wrong_rate = float(pred_is_swap[~is_actually_swap].sum() / n_actual_no_swap)
            else:
                traj_correct_rate = float("nan")
                feat_wrong_rate = float("nan")

            row = {
                "mechanism": mech_name,
                "conflict_type": "feature_wrong_trajectory_correct",
                "identity_under_conflict": fmt(identity_acc),
                "traj_correct_rate": fmt(traj_correct_rate),
                "feat_wrong_rate": fmt(feat_wrong_rate),
            }

            if sources_ep:
                gate = compute_conflict_gate_stats(sources_ep, pred_identity, true_id)
                row["chosen_source_feature_rate"] = fmt(gate['chosen_source_feature_rate'])
                row["chosen_source_trajectory_rate"] = fmt(gate['chosen_source_trajectory_rate'])
                row["chosen_source_uncertain_rate"] = fmt(gate['chosen_source_uncertain_rate'])
            else:
                row["chosen_source_feature_rate"] = "nan"
                row["chosen_source_trajectory_rate"] = "nan"
                row["chosen_source_uncertain_rate"] = "nan"

            if len(abstain_ep) > 0:
                abs_m = compute_abstention_metrics(pred_identity, true_id, np.array(abstain_ep))
                row["abstention_rate"] = fmt(abs_m['abstention_rate'])
                row["accuracy_when_not_abstaining"] = fmt(abs_m['accuracy_when_not_abstaining'])
            else:
                row["abstention_rate"] = "0.0000"
                row["accuracy_when_not_abstaining"] = fmt(identity_acc)

            conflict_results.append(row)
            print(f"    {mech_name}: acc={fmt(identity_acc)} traj_correct={fmt(traj_correct_rate)} "
                  f"feat_wrong={fmt(feat_wrong_rate)}")

    if len(swap_idx) > 0:
        print(f"  Type B: trajectory_noisy (n={len(swap_idx)})")
        swap_data = {k: v[swap_idx] for k, v in clean_test.items()}

        for mech_name, mech_type, mech in all_mechanisms:
            if mech_type == "learned":
                _, bd, _ = evaluate_learned_model(mech, swap_data,
                                                   fut_feat_override=swap_data["object_features_fut"])
            elif mech_type == "traj_only":
                _, bd, _ = evaluate_trajectory_only(mech, swap_data)
            elif mech_type == "obj_file_v4":
                bd, _ = evaluate_obj_file_v4(mech, swap_data,
                                              fut_feat_override=swap_data["object_features_fut"])
            elif mech_type in ("obj_file_v41", "obj_file_v42", "obj_file_v43"):
                bd, _, _, _, _ = evaluate_obj_file_with_conflict(mech, swap_data,
                                                                   fut_feat_override=swap_data["object_features_fut"])

            conflict_results.append({
                "mechanism": mech_name,
                "conflict_type": "trajectory_noisy",
                "identity_under_conflict": fmt(bd["identity_swap_only"]),
                "traj_correct_rate": fmt(1.0 - bd["identity_swap_only"]),
                "feat_wrong_rate": fmt(bd["identity_swap_only"]),
                "chosen_source_feature_rate": "nan",
                "chosen_source_trajectory_rate": "nan",
                "chosen_source_uncertain_rate": "nan",
                "abstention_rate": "0.0000",
                "accuracy_when_not_abstaining": fmt(bd["identity_swap_only"]),
            })

    save_csv(conflict_results, "conflict_gate_results.csv",
             ["mechanism", "conflict_type", "identity_under_conflict",
              "traj_correct_rate", "feat_wrong_rate",
              "chosen_source_feature_rate", "chosen_source_trajectory_rate",
              "chosen_source_uncertain_rate", "abstention_rate",
              "accuracy_when_not_abstaining"])

    # =========================================================================
    # Part 4: Occlusion Without Feature
    # =========================================================================
    print("\n=== Occlusion Without Feature ===")

    occ_results = []

    for occ_ratio in [0.0, 0.5, 0.75, 1.0]:
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

        for mech_name, mech_type, mech in all_mechanisms:
            if mech_type == "learned":
                _, bd, _ = evaluate_learned_model(mech, swap_test, fut_feat_override=fut_feat_occ)
            elif mech_type == "traj_only":
                _, bd, _ = evaluate_trajectory_only(mech, swap_test)
            elif mech_type == "obj_file_v4":
                bd, _ = evaluate_obj_file_v4(mech, swap_test,
                                              fut_feat_override=fut_feat_occ, occlusion_mask=occ_mask)
            elif mech_type in ("obj_file_v41", "obj_file_v42", "obj_file_v43"):
                bd, _, _, _, _ = evaluate_obj_file_with_conflict(mech, swap_test,
                                                                   fut_feat_override=fut_feat_occ, occlusion_mask=occ_mask)

            occ_results.append({
                "mechanism": mech_name,
                "occlusion_ratio": str(occ_ratio),
                "identity_under_occlusion_no_feature": fmt(bd["identity_swap_only"]),
            })

    save_csv(occ_results, "occlusion_no_feature_results.csv",
             ["mechanism", "occlusion_ratio", "identity_under_occlusion_no_feature"])

    # =========================================================================
    # Part 5: Approach Signal Analysis
    # =========================================================================
    print("\n=== Approach Signal Analysis ===")

    approach_results = []

    for dataset_name, dataset in [("swap_test", swap_test), ("clean_test", clean_test)]:
        approach_signals = obj_file_v43_aware._compute_approach_signal(dataset["observed_positions"])
        true_id = dataset["identity_labels"]
        is_swap = true_id[:, 0] != 0

        swap_approach = approach_signals[is_swap]
        no_swap_approach = approach_signals[~is_swap]

        approach_results.append({
            "dataset": dataset_name,
            "swap_mean_approach": fmt(float(np.mean(swap_approach))) if len(swap_approach) > 0 else "nan",
            "swap_std_approach": fmt(float(np.std(swap_approach))) if len(swap_approach) > 0 else "nan",
            "no_swap_mean_approach": fmt(float(np.mean(no_swap_approach))) if len(no_swap_approach) > 0 else "nan",
            "no_swap_std_approach": fmt(float(np.std(no_swap_approach))) if len(no_swap_approach) > 0 else "nan",
            "n_swap": int(is_swap.sum()),
            "n_no_swap": int((~is_swap).sum()),
        })

        print(f"  {dataset_name}: swap_approach={fmt(float(np.mean(swap_approach))) if len(swap_approach) > 0 else 'nan'} "
              f"no_swap_approach={fmt(float(np.mean(no_swap_approach))) if len(no_swap_approach) > 0 else 'nan'}")

    save_csv(approach_results, "approach_signal_analysis.csv",
             ["dataset", "swap_mean_approach", "swap_std_approach",
              "no_swap_mean_approach", "no_swap_std_approach",
              "n_swap", "n_no_swap"])

    # =========================================================================
    # README
    # =========================================================================
    v42_swap_row = [r for r in mech_results if r["mechanism"] == "ConflictFirst_margin_v4.2"]
    v43_aware_row = [r for r in mech_results if r["mechanism"] == "TrajRobust_aware_v4.3"]
    v43_aware_aug_row = [r for r in mech_results if r["mechanism"] == "TrajRobust_aware_aug_v4.3"]
    v43_veto_row = [r for r in mech_results if r["mechanism"] == "TrajRobust_veto_v4.3"]
    v43_veto_aug_row = [r for r in mech_results if r["mechanism"] == "TrajRobust_veto_aug_v4.3"]

    v42_swap = float(v42_swap_row[0]["identity_swap_only"]) if v42_swap_row else 0.0
    v43_aware_swap = float(v43_aware_row[0]["identity_swap_only"]) if v43_aware_row else 0.0
    v43_aware_aug_swap = float(v43_aware_aug_row[0]["identity_swap_only"]) if v43_aware_aug_row else 0.0
    v43_veto_swap = float(v43_veto_row[0]["identity_swap_only"]) if v43_veto_row else 0.0
    v43_veto_aug_swap = float(v43_veto_aug_row[0]["identity_swap_only"]) if v43_veto_aug_row else 0.0

    v43_aware_conflict = [r for r in conflict_results
                          if r["mechanism"] == "TrajRobust_aware_v4.3" and "feature_wrong" in r["conflict_type"]]
    v43_aware_aug_conflict = [r for r in conflict_results
                              if r["mechanism"] == "TrajRobust_aware_aug_v4.3" and "feature_wrong" in r["conflict_type"]]
    v43_veto_conflict = [r for r in conflict_results
                         if r["mechanism"] == "TrajRobust_veto_v4.3" and "feature_wrong" in r["conflict_type"]]
    v43_veto_aug_conflict = [r for r in conflict_results
                             if r["mechanism"] == "TrajRobust_veto_aug_v4.3" and "feature_wrong" in r["conflict_type"]]
    v42_conflict = [r for r in conflict_results
                    if r["mechanism"] == "ConflictFirst_margin_v4.2" and "feature_wrong" in r["conflict_type"]]

    v43_aware_conflict_res = float(v43_aware_conflict[0]["identity_under_conflict"]) if v43_aware_conflict else 0.0
    v43_aware_aug_conflict_res = float(v43_aware_aug_conflict[0]["identity_under_conflict"]) if v43_aware_aug_conflict else 0.0
    v43_veto_conflict_res = float(v43_veto_conflict[0]["identity_under_conflict"]) if v43_veto_conflict else 0.0
    v43_veto_aug_conflict_res = float(v43_veto_aug_conflict[0]["identity_under_conflict"]) if v43_veto_aug_conflict else 0.0
    v42_conflict_res = float(v42_conflict[0]["identity_under_conflict"]) if v42_conflict else 0.0

    v43_aware_cal = [r for r in cal_results if r["mechanism"] == "TrajRobust_aware_v4.3"]
    v43_aware_aug_cal = [r for r in cal_results if r["mechanism"] == "TrajRobust_aware_aug_v4.3"]
    v43_veto_cal = [r for r in cal_results if r["mechanism"] == "TrajRobust_veto_v4.3"]
    v43_veto_aug_cal = [r for r in cal_results if r["mechanism"] == "TrajRobust_veto_aug_v4.3"]

    best_swap = max(v43_aware_swap, v43_aware_aug_swap, v43_veto_swap, v43_veto_aug_swap)
    best_conflict = max(v43_aware_conflict_res, v43_aware_aug_conflict_res,
                        v43_veto_conflict_res, v43_veto_aug_conflict_res)

    pass_swap = best_swap > 0.55
    pass_conflict = best_conflict > 0.65

    if pass_swap and pass_conflict:
        recommendation = "proceed_to_paper_report"
    elif pass_conflict and not pass_swap:
        recommendation = "improve_trajectory_predictor"
    elif pass_swap and not pass_conflict:
        recommendation = "tune_conflict_thresholds"
    else:
        recommendation = "add_uncertainty_model"

    readme = f"""# SVT-v4.3: Trajectory-Robust ObjectFile

## 1. Purpose

v4.2 established conflict-first gate as the best balance mechanism. v4.3 addresses the root bottleneck: trajectory predictor weakness. Two innovations: (1) observed-period approach detection (are objects moving toward each other?), (2) multi-step trajectory voting with temporal decay, (3) trajectory training augmentation.

## 2. Mechanism Comparison

| Mechanism | Swap-Only | Feat Dep | Traj Dep |
|-----------|-----------|----------|----------|
"""
    for r in mech_results:
        readme += f"| {r['mechanism']} | {r['identity_swap_only']} | {r['feature_dependency_score']} | {r['trajectory_dependency_score']} |\n"

    readme += f"""
## 3. Confidence Calibration

| Mechanism | Swap | Conf Corr | Conf Inc | Cal Err | Feat Rate | Traj Rate | Uncertain | Abstain |
|-----------|------|-----------|----------|---------|-----------|-----------|-----------|---------|
"""
    for r in cal_results:
        readme += f"| {r['mechanism']} | {r['identity_swap_only']} | {r['avg_confidence_correct']} | {r['avg_confidence_incorrect']} | {r['confidence_calibration_error']} | {r['chosen_source_feature_rate']} | {r['chosen_source_trajectory_rate']} | {r['chosen_source_uncertain_rate']} | {r['abstention_rate']} |\n"

    readme += f"""
## 4. Conflict Gate Results

| Mechanism | Conflict Type | Identity | Traj Correct | Feat Wrong | Feat Rate | Traj Rate | Uncertain | Abstain |
|-----------|--------------|----------|-------------|-----------|-----------|-----------|-----------|---------|
"""
    for r in conflict_results:
        readme += f"| {r['mechanism']} | {r['conflict_type']} | {r['identity_under_conflict']} | {r['traj_correct_rate']} | {r['feat_wrong_rate']} | {r['chosen_source_feature_rate']} | {r['chosen_source_trajectory_rate']} | {r['chosen_source_uncertain_rate']} | {r['abstention_rate']} |\n"

    readme += f"""
## 5. Approach Signal Analysis

| Dataset | Swap Mean | Swap Std | No-Swap Mean | No-Swap Std | N Swap | N No-Swap |
|---------|-----------|----------|-------------|------------|--------|-----------|
"""
    for r in approach_results:
        readme += f"| {r['dataset']} | {r['swap_mean_approach']} | {r['swap_std_approach']} | {r['no_swap_mean_approach']} | {r['no_swap_std_approach']} | {r['n_swap']} | {r['n_no_swap']} |\n"

    readme += f"""
## 6. Key Comparisons

| Metric | v4.2 Margin | v4.3 Aware | v4.3 Aware+Aug | v4.3 Veto | v4.3 Veto+Aug |
|--------|------------|-----------|---------------|----------|--------------|
| Swap-Only | {v42_swap:.4f} | {v43_aware_swap:.4f} | {v43_aware_aug_swap:.4f} | {v43_veto_swap:.4f} | {v43_veto_aug_swap:.4f} |
| Conflict | {v42_conflict_res:.4f} | {v43_aware_conflict_res:.4f} | {v43_aware_aug_conflict_res:.4f} | {v43_veto_conflict_res:.4f} | {v43_veto_aug_conflict_res:.4f} |

## 7. Pass Criteria

| Criterion | Threshold | Best v4.3 | Pass |
|-----------|-----------|-----------|------|
| swap-only > 0.55 | > 0.55 | {best_swap:.4f} | {"YES" if pass_swap else "NO"} |
| conflict resolution > 0.65 | > 0.65 | {best_conflict:.4f} | {"YES" if pass_conflict else "NO"} |

## 8. Recommendation

**{recommendation}**
"""

    with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print("\n" + "=" * 60)
    print("SVT-v4.3 FINAL SUMMARY")
    print("=" * 60)
    for r in mech_results:
        print(f"  {r['mechanism']}: swap={r['identity_swap_only']} feat_dep={r['feature_dependency_score']} traj_dep={r['trajectory_dependency_score']}")
    print(f"  v4.2 conflict: {v42_conflict_res:.4f}")
    print(f"  v4.3 best conflict: {best_conflict:.4f}")
    print(f"  v4.3 best swap: {best_swap:.4f}")
    print(f"  Pass: swap={pass_swap} conflict={pass_conflict}")
    print(f"  Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
