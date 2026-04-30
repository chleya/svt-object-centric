"""
SVT-v4.2: Conflict-First ObjectFile
"""

import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v4_2_conflict_first_object_file"
SEED = 0
SWAP_RATIO = 0.3
EPOCHS = 20

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


def evaluate_object_file_v4(mechanism, test_data, obs_feat_override=None,
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

    return breakdown, asgn_acc


def evaluate_object_file_v41(mechanism, test_data, obs_feat_override=None,
                              fut_feat_override=None, occlusion_mask=None):
    obs_pos = test_data["observed_positions"]
    obs_feat = obs_feat_override if obs_feat_override is not None else test_data.get("object_features_obs")
    fut_feat = fut_feat_override if fut_feat_override is not None else test_data.get("object_features_fut")
    fut_pos = test_data["future_positions"]

    pred_identity, confidences, sources = mechanism.predict_identity(
        obs_pos, obs_feat, fut_pos, fut_feat,
        occlusion_mask=occlusion_mask, return_conflict_info=True)

    true_identity = test_data["identity_labels"]
    asgn_acc = compute_assignment_accuracy(pred_identity, true_identity)
    breakdown = compute_identity_breakdown(pred_identity, true_identity)

    return breakdown, asgn_acc, confidences, sources


def evaluate_object_file_v42(mechanism, test_data, obs_feat_override=None,
                              fut_feat_override=None, occlusion_mask=None):
    obs_pos = test_data["observed_positions"]
    obs_feat = obs_feat_override if obs_feat_override is not None else test_data.get("object_features_obs")
    fut_feat = fut_feat_override if fut_feat_override is not None else test_data.get("object_features_fut")
    fut_pos = test_data["future_positions"]

    pred_identity, confidences, sources, abstain_flags = mechanism.predict_identity(
        obs_pos, obs_feat, fut_pos, fut_feat,
        occlusion_mask=occlusion_mask, return_conflict_info=True)

    true_identity = test_data["identity_labels"]
    asgn_acc = compute_assignment_accuracy(pred_identity, true_identity)
    breakdown = compute_identity_breakdown(pred_identity, true_identity)

    return breakdown, asgn_acc, confidences, sources, abstain_flags


def extract_confidences_v41(confidences, true_identity):
    total_confs = []
    if confidences and confidences[0]:
        for ep_confs in confidences[0]:
            if isinstance(ep_confs, dict) and 'total_confs' in ep_confs:
                max_conf = max(ep_confs['total_confs'])
                total_confs.append(max_conf)
    if len(total_confs) != len(true_identity):
        total_confs = np.ones(len(true_identity))
    else:
        total_confs = np.array(total_confs)
    return total_confs


def extract_confidences_v42(confidences, true_identity):
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
    print("SVT-v4.2: Conflict-First ObjectFile")
    print("=" * 60)
    print(f"PyTorch: {'available' if TORCH_AVAILABLE else 'UNAVAILABLE'} (device={DEVICE})")

    if not TORCH_AVAILABLE:
        print("PyTorch unavailable. Exiting.")
        return

    from models.feature_similarity_models import FeatureOnlyAssignmentHead, HybridTrajectoryFeatureAssignmentHead
    from models.object_file_models import (
        TrajectoryOnlyAssignment, MinimalObjectFileMechanism,
        ImprovedObjectFile, ConflictFirstObjectFile,
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

    obj_file_v4 = MinimalObjectFileMechanism(feature_weight=1.0, traj_weight=1.0)
    obj_file_v41 = ImprovedObjectFile(traj_model=model_traj)

    obj_file_v42_traj = ConflictFirstObjectFile(
        traj_model=model_traj,
        strategy="prefer_trajectory_on_conflict")
    obj_file_v42_feat_low_traj = ConflictFirstObjectFile(
        traj_model=model_traj,
        strategy="prefer_feature_on_low_trajectory_confidence")
    obj_file_v42_abstain = ConflictFirstObjectFile(
        traj_model=model_traj,
        strategy="abstain_on_high_conflict",
        high_conflict_abstain_threshold=2.0)
    obj_file_v42_margin = ConflictFirstObjectFile(
        traj_model=model_traj,
        strategy="margin_gated",
        traj_margin_advantage=1.2)

    all_mechanisms = [
        ("FeatureOnly", "learned", model_fo),
        ("TrajectoryOnly", "traj_only", model_traj),
        ("Hybrid_b1.0", "learned", model_hybrid1),
        ("ObjectFile_v4", "obj_file_v4", obj_file_v4),
        ("ImprovedObjectFile_v4.1", "obj_file_v41", obj_file_v41),
        ("ConflictFirst_traj", "obj_file_v42", obj_file_v42_traj),
        ("ConflictFirst_feat_low_traj", "obj_file_v42", obj_file_v42_feat_low_traj),
        ("ConflictFirst_abstain", "obj_file_v42", obj_file_v42_abstain),
        ("ConflictFirst_margin", "obj_file_v42", obj_file_v42_margin),
    ]

    # =========================================================================
    # Part 1: Mechanism Comparison
    # =========================================================================
    print("\n=== Mechanism Comparison ===")

    mech_results = []

    for mech_name, mech_type, mech in all_mechanisms:
        if mech_type == "learned":
            _, bd, asgn_acc = evaluate_learned_model(mech, swap_test)
        elif mech_type == "traj_only":
            _, bd, asgn_acc = evaluate_trajectory_only(mech, swap_test)
        elif mech_type == "obj_file_v4":
            bd, asgn_acc = evaluate_object_file_v4(mech, swap_test)
        elif mech_type == "obj_file_v41":
            bd, asgn_acc, _, _ = evaluate_object_file_v41(mech, swap_test)
        elif mech_type == "obj_file_v42":
            bd, asgn_acc, _, _, _ = evaluate_object_file_v42(mech, swap_test)

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
            bd_shuf, _ = evaluate_object_file_v4(mech, swap_test,
                                                   obs_feat_override=obs_shuf, fut_feat_override=fut_shuf)
            bd_zero, _ = evaluate_object_file_v4(mech, swap_test,
                                                   obs_feat_override=obs_zero, fut_feat_override=fut_zero)
        elif mech_type == "obj_file_v41":
            bd_shuf, _, _, _ = evaluate_object_file_v41(mech, swap_test,
                                                          obs_feat_override=obs_shuf, fut_feat_override=fut_shuf)
            bd_zero, _, _, _ = evaluate_object_file_v41(mech, swap_test,
                                                          obs_feat_override=obs_zero, fut_feat_override=fut_zero)
        elif mech_type == "obj_file_v42":
            bd_shuf, _, _, _, _ = evaluate_object_file_v42(mech, swap_test,
                                                             obs_feat_override=obs_shuf, fut_feat_override=fut_shuf)
            bd_zero, _, _, _, _ = evaluate_object_file_v42(mech, swap_test,
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
        ("ConflictFirst_traj", "obj_file_v42", obj_file_v42_traj),
        ("ConflictFirst_feat_low_traj", "obj_file_v42", obj_file_v42_feat_low_traj),
        ("ConflictFirst_abstain", "obj_file_v42", obj_file_v42_abstain),
        ("ConflictFirst_margin", "obj_file_v42", obj_file_v42_margin),
    ]

    for mech_name, mech_type, mech in obj_file_mechs:
        if mech_type == "obj_file_v4":
            bd, asgn_acc = evaluate_object_file_v4(mech, swap_test)
            pred_identity = mech.predict_identity(
                swap_test["observed_positions"],
                swap_test["object_features_obs"],
                swap_test["future_positions"],
                swap_test["object_features_fut"])
            total_confs = np.ones(len(swap_test["identity_labels"]))
            sources = []
            abstain_flags = np.zeros(len(swap_test["identity_labels"]), dtype=bool)
        elif mech_type == "obj_file_v41":
            bd, asgn_acc, confidences, sources = evaluate_object_file_v41(mech, swap_test)
            pred_identity = mech.predict_identity(
                swap_test["observed_positions"],
                swap_test["object_features_obs"],
                swap_test["future_positions"],
                swap_test["object_features_fut"],
                return_conflict_info=True)
            if isinstance(pred_identity, tuple):
                pred_identity = pred_identity[0]
            total_confs = extract_confidences_v41(confidences, swap_test["identity_labels"])
            abstain_flags = np.zeros(len(swap_test["identity_labels"]), dtype=bool)
        elif mech_type == "obj_file_v42":
            bd, asgn_acc, confidences, sources, abstain_flags_raw = evaluate_object_file_v42(mech, swap_test)
            pred_identity = mech.predict_identity(
                swap_test["observed_positions"],
                swap_test["object_features_obs"],
                swap_test["future_positions"],
                swap_test["object_features_fut"],
                return_conflict_info=True)
            if isinstance(pred_identity, tuple):
                pred_identity = pred_identity[0]
            total_confs = extract_confidences_v42(confidences, swap_test["identity_labels"])
            if isinstance(abstain_flags_raw, list):
                abstain_flags = np.array(abstain_flags_raw)
            else:
                abstain_flags = abstain_flags_raw

        true_identity = swap_test["identity_labels"]
        cal = compute_confidence_calibration(pred_identity, true_identity, total_confs)

        if sources:
            gate = compute_conflict_gate_stats(sources, pred_identity, true_identity)
        else:
            gate = {
                'chosen_source_feature_rate': float('nan'),
                'chosen_source_trajectory_rate': float('nan'),
                'chosen_source_uncertain_rate': float('nan'),
                'chosen_source_agreement_rate': float('nan'),
            }

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

        print(f"  {mech_name}: cal_err={fmt(cal['confidence_calibration_error'])} "
              f"conf_correct={fmt(cal['avg_confidence_correct'])} conf_incorrect={fmt(cal['avg_confidence_incorrect'])} "
              f"feat_rate={fmt(gate['chosen_source_feature_rate'])} traj_rate={fmt(gate['chosen_source_trajectory_rate'])} "
              f"uncertain_rate={fmt(gate['chosen_source_uncertain_rate'])} "
              f"abstain_rate={fmt(abs_metrics['abstention_rate'])}")

    save_csv(cal_results, "confidence_calibration.csv",
             ["mechanism", "identity_swap_only", "avg_confidence_correct",
              "avg_confidence_incorrect", "confidence_calibration_error",
              "chosen_source_feature_rate", "chosen_source_trajectory_rate",
              "chosen_source_uncertain_rate", "chosen_source_agreement_rate",
              "abstention_rate", "accuracy_when_not_abstaining"])

    # =========================================================================
    # Part 3: Feature Noise Stress Test
    # =========================================================================
    print("\n=== Feature Noise Stress Test ===")

    noise_results = []

    for noise_std in [0.0, 0.1, 0.3, 0.5]:
        obs_noisy = add_feature_noise(swap_test["object_features_obs"], noise_std, seed=SEED)
        fut_noisy = add_feature_noise(swap_test["object_features_fut"], noise_std, seed=SEED + 1000)

        for mech_name, mech_type, mech in all_mechanisms:
            if mech_type == "learned":
                _, bd, _ = evaluate_learned_model(mech, swap_test,
                                                   obs_feat_override=obs_noisy, fut_feat_override=fut_noisy)
            elif mech_type == "traj_only":
                _, bd, _ = evaluate_trajectory_only(mech, swap_test)
            elif mech_type == "obj_file_v4":
                bd, _ = evaluate_object_file_v4(mech, swap_test,
                                                 obs_feat_override=obs_noisy, fut_feat_override=fut_noisy)
            elif mech_type == "obj_file_v41":
                bd, _, _, _ = evaluate_object_file_v41(mech, swap_test,
                                                        obs_feat_override=obs_noisy, fut_feat_override=fut_noisy)
            elif mech_type == "obj_file_v42":
                bd, _, _, _, _ = evaluate_object_file_v42(mech, swap_test,
                                                           obs_feat_override=obs_noisy, fut_feat_override=fut_noisy)

            noise_results.append({
                "mechanism": mech_name,
                "feature_noise_std": str(noise_std),
                "identity_under_feature_noise": fmt(bd["identity_swap_only"]),
            })

    save_csv(noise_results, "feature_noise_results.csv",
             ["mechanism", "feature_noise_std", "identity_under_feature_noise"])

    # =========================================================================
    # Part 4: Occlusion Without Feature
    # =========================================================================
    print("\n=== Occlusion Without Feature ===")

    occ_results = []

    for occ_ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
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
                bd, _ = evaluate_object_file_v4(mech, swap_test,
                                                 fut_feat_override=fut_feat_occ, occlusion_mask=occ_mask)
            elif mech_type == "obj_file_v41":
                bd, _, _, _ = evaluate_object_file_v41(mech, swap_test,
                                                        fut_feat_override=fut_feat_occ, occlusion_mask=occ_mask)
            elif mech_type == "obj_file_v42":
                bd, _, _, _, _ = evaluate_object_file_v42(mech, swap_test,
                                                           fut_feat_override=fut_feat_occ, occlusion_mask=occ_mask)

            occ_results.append({
                "mechanism": mech_name,
                "occlusion_ratio": str(occ_ratio),
                "identity_under_occlusion_no_feature": fmt(bd["identity_swap_only"]),
            })

    save_csv(occ_results, "occlusion_no_feature_results.csv",
             ["mechanism", "occlusion_ratio", "identity_under_occlusion_no_feature"])

    # =========================================================================
    # Part 5: Feature-Trajectory Conflict
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
                    no_swap_data["future_positions"],
                    fut_flipped)
            elif mech_type == "obj_file_v41":
                pred_identity, conf, sources_ep = mech.predict_identity(
                    no_swap_data["observed_positions"],
                    no_swap_data["object_features_obs"],
                    no_swap_data["future_positions"],
                    fut_flipped, return_conflict_info=True)
            elif mech_type == "obj_file_v42":
                pred_identity, conf, sources_ep, abstain_ep = mech.predict_identity(
                    no_swap_data["observed_positions"],
                    no_swap_data["object_features_obs"],
                    no_swap_data["future_positions"],
                    fut_flipped, return_conflict_info=True)

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
                "conflict_resolution_rate_trajectory_correct": fmt(traj_correct_rate),
                "conflict_resolution_rate_feature_correct": fmt(feat_wrong_rate),
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

            if len(sources_ep) > 0:
                true_conflict_flags = np.array([s != "agreement" for s in sources_ep])
                true_is_conflict_arr = np.ones(len(sources_ep), dtype=bool)
                cd = compute_conflict_detection_accuracy(sources_ep, true_is_conflict_arr)
                row["conflict_detection_accuracy"] = fmt(cd['conflict_detection_accuracy'])
            else:
                row["conflict_detection_accuracy"] = "nan"

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
                bd, _ = evaluate_object_file_v4(mech, swap_data,
                                                 fut_feat_override=swap_data["object_features_fut"])
            elif mech_type == "obj_file_v41":
                bd, _, _, _ = evaluate_object_file_v41(mech, swap_data,
                                                        fut_feat_override=swap_data["object_features_fut"])
            elif mech_type == "obj_file_v42":
                bd, _, _, _, _ = evaluate_object_file_v42(mech, swap_data,
                                                           fut_feat_override=swap_data["object_features_fut"])

            conflict_results.append({
                "mechanism": mech_name,
                "conflict_type": "trajectory_noisy",
                "identity_under_conflict": fmt(bd["identity_swap_only"]),
                "conflict_resolution_rate_trajectory_correct": fmt(1.0 - bd["identity_swap_only"]),
                "conflict_resolution_rate_feature_correct": fmt(bd["identity_swap_only"]),
                "chosen_source_feature_rate": "nan",
                "chosen_source_trajectory_rate": "nan",
                "chosen_source_uncertain_rate": "nan",
                "abstention_rate": "0.0000",
                "accuracy_when_not_abstaining": fmt(bd["identity_swap_only"]),
                "conflict_detection_accuracy": "nan",
            })

    save_csv(conflict_results, "conflict_gate_results.csv",
             ["mechanism", "conflict_type", "identity_under_conflict",
              "conflict_resolution_rate_trajectory_correct",
              "conflict_resolution_rate_feature_correct",
              "chosen_source_feature_rate", "chosen_source_trajectory_rate",
              "chosen_source_uncertain_rate", "abstention_rate",
              "accuracy_when_not_abstaining", "conflict_detection_accuracy"])

    # =========================================================================
    # Part 6: Abstention Analysis (ConflictFirst_abstain only)
    # =========================================================================
    print("\n=== Abstention Analysis ===")

    abstention_results = []

    for strategy_name, strategy_mech in [
        ("ConflictFirst_traj", obj_file_v42_traj),
        ("ConflictFirst_feat_low_traj", obj_file_v42_feat_low_traj),
        ("ConflictFirst_abstain", obj_file_v42_abstain),
        ("ConflictFirst_margin", obj_file_v42_margin),
    ]:
        pred_id, confs, srcs, abs_flags = strategy_mech.predict_identity(
            swap_test["observed_positions"],
            swap_test["object_features_obs"],
            swap_test["future_positions"],
            swap_test["object_features_fut"],
            return_conflict_info=True)

        true_id = swap_test["identity_labels"]
        abs_m = compute_abstention_metrics(pred_id, true_id, np.array(abs_flags))

        total_confs = extract_confidences_v42(confs, true_id)
        cal = compute_confidence_calibration(pred_id, true_id, total_confs)

        abstention_results.append({
            "mechanism": strategy_name,
            "abstention_rate": fmt(abs_m['abstention_rate']),
            "accuracy_when_not_abstaining": fmt(abs_m['accuracy_when_not_abstaining']),
            "accuracy_when_abstaining": fmt(abs_m['accuracy_when_abstaining']),
            "avg_confidence_correct": fmt(cal['avg_confidence_correct']),
            "avg_confidence_incorrect": fmt(cal['avg_confidence_incorrect']),
            "confidence_calibration_error": fmt(cal['confidence_calibration_error']),
        })

        print(f"  {strategy_name}: abstain_rate={fmt(abs_m['abstention_rate'])} "
              f"acc_not_abstain={fmt(abs_m['accuracy_when_not_abstaining'])} "
              f"conf_correct={fmt(cal['avg_confidence_correct'])} conf_incorrect={fmt(cal['avg_confidence_incorrect'])}")

    save_csv(abstention_results, "abstention_results.csv",
             ["mechanism", "abstention_rate", "accuracy_when_not_abstaining",
              "accuracy_when_abstaining", "avg_confidence_correct",
              "avg_confidence_incorrect", "confidence_calibration_error"])

    # =========================================================================
    # Part 7: Object File Trace Examples
    # =========================================================================
    print("\n=== Object File Trace Examples ===")

    n_examples = min(3, swap_test["observed_positions"].shape[0])
    trace_data = {k: v[:n_examples] for k, v in swap_test.items()}

    _, traces = obj_file_v42_traj.predict_identity(
        trace_data["observed_positions"],
        trace_data["object_features_obs"],
        trace_data["future_positions"],
        trace_data["object_features_fut"],
        return_trace=True)

    trace_rows = []
    for ep_traces in traces:
        for step in ep_traces:
            trace_rows.append({
                "episode_idx": step["episode_idx"],
                "timestep": step["timestep"],
                "object_idx": step["object_idx"],
                "identity_key_0": fmt(step["identity_key_0"]),
                "identity_key_1": fmt(step["identity_key_1"]),
                "last_pos_x": fmt(step["last_pos_x"]),
                "last_pos_y": fmt(step["last_pos_y"]),
                "occluded": str(step["occluded"]),
                "feature_confidence": fmt(step["feature_confidence"]),
                "trajectory_confidence": fmt(step["trajectory_confidence"]),
                "total_confidence": fmt(step["total_confidence"]),
            })

    save_csv(trace_rows, "object_file_trace_examples.csv",
             ["episode_idx", "timestep", "object_idx", "identity_key_0",
              "identity_key_1", "last_pos_x", "last_pos_y", "occluded",
              "feature_confidence", "trajectory_confidence", "total_confidence"])

    print(f"  Saved {len(trace_rows)} trace rows")

    # =========================================================================
    # README
    # =========================================================================
    v4_swap_row = [r for r in mech_results if r["mechanism"] == "ObjectFile_v4"]
    v41_swap_row = [r for r in mech_results if r["mechanism"] == "ImprovedObjectFile_v4.1"]
    v42_margin_row = [r for r in mech_results if r["mechanism"] == "ConflictFirst_margin"]
    v42_traj_row = [r for r in mech_results if r["mechanism"] == "ConflictFirst_traj"]
    v42_abstain_row = [r for r in mech_results if r["mechanism"] == "ConflictFirst_abstain"]

    v4_swap = float(v4_swap_row[0]["identity_swap_only"]) if v4_swap_row else 0.0
    v41_swap = float(v41_swap_row[0]["identity_swap_only"]) if v41_swap_row else 0.0
    v42_margin_swap = float(v42_margin_row[0]["identity_swap_only"]) if v42_margin_row else 0.0
    v42_traj_swap = float(v42_traj_row[0]["identity_swap_only"]) if v42_traj_row else 0.0
    v42_abstain_swap = float(v42_abstain_row[0]["identity_swap_only"]) if v42_abstain_row else 0.0

    v41_conflict = [r for r in conflict_results
                    if r["mechanism"] == "ImprovedObjectFile_v4.1" and "feature_wrong" in r["conflict_type"]]
    v42_margin_conflict = [r for r in conflict_results
                           if r["mechanism"] == "ConflictFirst_margin" and "feature_wrong" in r["conflict_type"]]
    v42_traj_conflict = [r for r in conflict_results
                         if r["mechanism"] == "ConflictFirst_traj" and "feature_wrong" in r["conflict_type"]]
    v42_abstain_conflict = [r for r in conflict_results
                            if r["mechanism"] == "ConflictFirst_abstain" and "feature_wrong" in r["conflict_type"]]

    v41_conflict_res = float(v41_conflict[0]["identity_under_conflict"]) if v41_conflict else 0.0
    v42_margin_conflict_res = float(v42_margin_conflict[0]["identity_under_conflict"]) if v42_margin_conflict else 0.0
    v42_traj_conflict_res = float(v42_traj_conflict[0]["identity_under_conflict"]) if v42_traj_conflict else 0.0

    v41_traj_correct = float(v41_conflict[0]["conflict_resolution_rate_trajectory_correct"]) if v41_conflict else 0.0
    v42_margin_traj_correct = float(v42_margin_conflict[0]["conflict_resolution_rate_trajectory_correct"]) if v42_margin_conflict else 0.0

    v42_margin_cal = [r for r in cal_results if r["mechanism"] == "ConflictFirst_margin"]
    v42_abstain_cal = [r for r in cal_results if r["mechanism"] == "ConflictFirst_abstain"]

    v42_margin_conf_corr = float(v42_margin_cal[0]["avg_confidence_correct"]) if v42_margin_cal else 0.0
    v42_margin_conf_inc = float(v42_margin_cal[0]["avg_confidence_incorrect"]) if v42_margin_cal else 0.0
    v42_margin_cal_err = float(v42_margin_cal[0]["confidence_calibration_error"]) if v42_margin_cal else 0.0

    v42_abstain_conf_corr = float(v42_abstain_cal[0]["avg_confidence_correct"]) if v42_abstain_cal else 0.0
    v42_abstain_conf_inc = float(v42_abstain_cal[0]["avg_confidence_incorrect"]) if v42_abstain_cal else 0.0

    pass_conflict = v42_margin_conflict_res > 0.610
    pass_swap = v42_margin_swap >= 0.558 - 0.05
    pass_confidence = v42_margin_conf_corr > v42_margin_conf_inc + 0.05

    v42_margin_uncertain = float(v42_margin_cal[0]["chosen_source_uncertain_rate"]) if v42_margin_cal else 0.0
    v42_abstain_uncertain = float(v42_abstain_cal[0]["chosen_source_uncertain_rate"]) if v42_abstain_cal else 0.0

    v42_margin_conflict_uncertain = float(v42_margin_conflict[0]["chosen_source_uncertain_rate"]) if v42_margin_conflict else 0.0
    v42_abstain_conflict_uncertain = float(v42_abstain_conflict[0]["chosen_source_uncertain_rate"]) if v42_abstain_conflict else 0.0
    pass_uncertain = v42_margin_conflict_uncertain > 0.0 or v42_abstain_conflict_uncertain > 0.0

    n_pass = sum([pass_conflict, pass_swap, pass_confidence, pass_uncertain])

    if n_pass >= 3:
        recommendation = "proceed_to_paper_report"
    elif pass_conflict and pass_swap and not pass_confidence:
        recommendation = "add_uncertainty_model"
    elif pass_conflict and not pass_swap:
        recommendation = "improve_trajectory_predictor"
    elif not pass_conflict:
        recommendation = "tune_conflict_thresholds"
    else:
        recommendation = "object_file_not_supported"

    readme = f"""# SVT-v4.2: Conflict-First ObjectFile

## 1. Purpose

v4.1 improved normal swap-only identity (9.6% -> 55.8%) but conflict resolution dropped (93.3% -> 61.0%). Confidence-based adaptive weighting failed because feature confidence is always high. v4.2 changes the gate from "weighted fusion" to "detect conflict first, then adjudicate".

## 2. Mechanism Comparison

| Mechanism | Swap-Only | Feat Dep | Traj Dep |
|-----------|-----------|----------|----------|
"""

    for r in mech_results:
        readme += f"| {r['mechanism']} | {r['identity_swap_only']} | {r['feature_dependency_score']} | {r['trajectory_dependency_score']} |\n"

    readme += f"""
## 3. Confidence Calibration

| Mechanism | Conf Correct | Conf Incorrect | Cal Error | Feat Rate | Traj Rate | Uncertain Rate | Abstain Rate | Acc Not Abstain |
|-----------|-------------|---------------|-----------|-----------|-----------|---------------|-------------|----------------|
"""

    for r in cal_results:
        readme += f"| {r['mechanism']} | {r['avg_confidence_correct']} | {r['avg_confidence_incorrect']} | {r['confidence_calibration_error']} | {r['chosen_source_feature_rate']} | {r['chosen_source_trajectory_rate']} | {r['chosen_source_uncertain_rate']} | {r['abstention_rate']} | {r['accuracy_when_not_abstaining']} |\n"

    readme += f"""
## 4. Conflict Gate Results

| Mechanism | Conflict Type | Identity | Traj Correct | Feat Wrong | Feat Rate | Traj Rate | Uncertain Rate | Abstain Rate | Acc Not Abstain | Conflict Det Acc |
|-----------|--------------|----------|-------------|-----------|-----------|-----------|---------------|-------------|----------------|----------------|
"""

    for r in conflict_results:
        readme += f"| {r['mechanism']} | {r['conflict_type']} | {r['identity_under_conflict']} | {r['conflict_resolution_rate_trajectory_correct']} | {r['conflict_resolution_rate_feature_correct']} | {r['chosen_source_feature_rate']} | {r['chosen_source_trajectory_rate']} | {r['chosen_source_uncertain_rate']} | {r['abstention_rate']} | {r['accuracy_when_not_abstaining']} | {r['conflict_detection_accuracy']} |\n"

    readme += f"""
## 5. Abstention Analysis

| Mechanism | Abstain Rate | Acc Not Abstain | Acc Abstain | Conf Correct | Conf Incorrect | Cal Error |
|-----------|-------------|----------------|------------|-------------|---------------|-----------|
"""

    for r in abstention_results:
        readme += f"| {r['mechanism']} | {r['abstention_rate']} | {r['accuracy_when_not_abstaining']} | {r['accuracy_when_abstaining']} | {r['avg_confidence_correct']} | {r['avg_confidence_incorrect']} | {r['confidence_calibration_error']} |\n"

    readme += f"""
## 6. Occlusion Without Feature

| Mechanism | Occ=0.0 | Occ=0.5 | Occ=0.75 | Occ=1.0 |
|-----------|---------|---------|----------|---------|
"""

    for mech_name in ["FeatureOnly", "TrajectoryOnly", "Hybrid_b1.0", "ObjectFile_v4",
                       "ImprovedObjectFile_v4.1", "ConflictFirst_traj",
                       "ConflictFirst_feat_low_traj", "ConflictFirst_abstain",
                       "ConflictFirst_margin"]:
        vals = []
        for occ in ["0.0", "0.5", "0.75", "1.0"]:
            rows = [r for r in occ_results if r["mechanism"] == mech_name and r["occlusion_ratio"] == occ]
            vals.append(rows[0]["identity_under_occlusion_no_feature"] if rows else "nan")
        readme += f"| {mech_name} | {' | '.join(vals)} |\n"

    readme += f"""
## 7. Pass Criteria

| Criterion | Threshold | Value | Pass |
|-----------|-----------|-------|------|
| conflict resolution > v4.1 | > 0.610 | {fmt(v42_margin_conflict_res)} | {"YES" if pass_conflict else "NO"} |
| swap-only not below v4.1 | >= 0.508 | {fmt(v42_margin_swap)} | {"YES" if pass_swap else "NO"} |
| confidence separation | correct > incorrect + 0.05 | {fmt(v42_margin_conf_corr)} vs {fmt(v42_margin_conf_inc)} | {"YES" if pass_confidence else "NO"} |
| uncertain on high-conflict | > 0 | margin={fmt(v42_margin_conflict_uncertain)} abstain={fmt(v42_abstain_conflict_uncertain)} | {"YES" if pass_uncertain else "NO"} |

## 8. Answers

### Q1: Does conflict-first improve conflict resolution over v4.1?

v4.1: {fmt(v41_conflict_res)}, ConflictFirst_margin: {fmt(v42_margin_conflict_res)}

{"YES - conflict-first gate improves conflict resolution." if pass_conflict else "NO - conflict-first gate does not improve conflict resolution."}

### Q2: Does it maintain normal swap-only identity?

v4.1: {fmt(v41_swap)}, ConflictFirst_margin: {fmt(v42_margin_swap)}

{"YES - swap-only identity maintained." if pass_swap else "NO - swap-only identity degraded."}

### Q3: Can confidence distinguish correct vs incorrect?

Correct: {fmt(v42_margin_conf_corr)}, Incorrect: {fmt(v42_margin_conf_inc)}, Cal Error: {fmt(v42_margin_cal_err)}

{"YES - confidence is calibrated." if pass_confidence else "NO - confidence is not well calibrated."}

### Q4: Does uncertain/abstain increase on high-conflict samples?

Margin-gated conflict uncertain rate: {fmt(v42_margin_conflict_uncertain)}, Abstain conflict uncertain rate: {fmt(v42_abstain_conflict_uncertain)}

{"YES - uncertain/abstain increases on high-conflict samples." if pass_uncertain else "NO - uncertain/abstain does not increase."}

### Q5: Final Recommendation

**{recommendation}**
"""

    with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print("\n" + "=" * 60)
    print("SVT-v4.2 FINAL SUMMARY")
    print("=" * 60)
    for r in mech_results:
        print(f"  {r['mechanism']}: swap={r['identity_swap_only']} feat_dep={r['feature_dependency_score']} traj_dep={r['trajectory_dependency_score']}")
    print(f"  v4.1 conflict: {fmt(v41_conflict_res)} -> v4.2 conflict: {fmt(v42_margin_conflict_res)}")
    print(f"  v4.1 traj_correct: {fmt(v41_traj_correct)} -> v4.2 traj_correct: {fmt(v42_margin_traj_correct)}")
    print(f"  Confidence: correct={fmt(v42_margin_conf_corr)} incorrect={fmt(v42_margin_conf_inc)} cal_err={fmt(v42_margin_cal_err)}")
    print(f"  Pass: conflict={pass_conflict} swap={pass_swap} conf={pass_confidence} uncertain={pass_uncertain}")
    print(f"  Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
