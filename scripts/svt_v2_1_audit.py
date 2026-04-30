"""
SVT-v2.1 Leakage and Identifiability Audit

This script performs comprehensive audits to understand why RawKNN v1
achieves identity=0.830 on identity_test, and whether this comes from:
- Data leakage (train/test overlap in initial positions, object order, etc.)
- Real structural cues (trajectory shape, occlusion patterns)
- Evaluation definition漏洞

Audits:
A. Object Order Randomization
B. Train/Test Disjoint Initial Position Split
C. Swap Time and Occlusion Location Randomization
D. Identity Label Permutation Sanity Check
E. Absolute Position Ablation
F. Nearest Neighbor Source Analysis
G. Identifiability Probe
"""

import sys
import os
import numpy as np
import csv
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.knn_retriever import KNN_REGISTRY
from baselines.knn_retriever_v2 import KNN_V2_REGISTRY
from metrics.prediction_metrics import compute_prediction_metrics
from metrics.identity_metrics import compute_identity_accuracy


def load_split(data_dir, split_name):
    path = os.path.join(data_dir, f"{split_name}.npz")
    data = np.load(path)
    return {
        "observed_positions": data["observed_positions"],
        "observed_velocities": data["observed_velocities"],
        "future_positions": data["future_positions"],
        "future_velocities": data["future_velocities"],
        "identity_labels": data["identity_labels"],
    }


def evaluate_knn_identity(model_class, train_data, test_data, k=5, weighting="inverse_distance"):
    """Evaluate a k-NN model on identity prediction."""
    model = model_class(k=k, weighting=weighting)
    model.fit(
        train_data["observed_positions"],
        train_data["future_positions"],
        train_data["identity_labels"],
    )

    pred_future = model.predict_future(test_data["observed_positions"])
    pred_identity = model.predict_identity(
        test_data["observed_positions"],
        test_future=test_data["future_positions"],
    )

    id_acc = compute_identity_accuracy(pred_identity, test_data["identity_labels"])
    pred_metrics = compute_prediction_metrics(pred_future, test_data["future_positions"])

    return {
        "identity_accuracy": id_acc,
        "skill_score": pred_metrics["skill_score"],
        "mse": pred_metrics["mse"],
    }


# =============================================================================
# Audit A: Object Order Randomization
# =============================================================================
def audit_object_order_randomization(data_dir, output_dir):
    """Randomly permute object order in train and test, then re-evaluate."""
    print("\n" + "=" * 60)
    print("AUDIT A: Object Order Randomization")
    print("=" * 60)

    train_data = load_split(data_dir, "train")
    test_data = load_split(data_dir, "identity_test")

    results = []
    rng = np.random.RandomState(42)

    for trial in range(5):
        # Random permutation for train
        train_perm = rng.permutation(2)
        train_obs = train_data["observed_positions"][:, :, train_perm, :]
        train_future = train_data["future_positions"][:, :, train_perm, :]
        train_ids = train_data["identity_labels"][:, train_perm]

        # Random permutation for test (independent)
        test_perm = rng.permutation(2)
        test_obs = test_data["observed_positions"][:, :, test_perm, :]
        test_future = test_data["future_positions"][:, :, test_perm, :]
        test_ids = test_data["identity_labels"][:, test_perm]

        perm_train = {
            "observed_positions": train_obs,
            "future_positions": train_future,
            "identity_labels": train_ids,
        }
        perm_test = {
            "observed_positions": test_obs,
            "future_positions": test_future,
            "identity_labels": test_ids,
        }

        for model_name, model_class in [
            ("RawTrajectoryKNN", KNN_REGISTRY["RawTrajectoryKNN"]),
            ("TranslationNormalizedKNN", KNN_REGISTRY["TranslationNormalizedKNN"]),
        ]:
            res = evaluate_knn_identity(model_class, perm_train, perm_test, k=5)
            results.append({
                "audit": "object_order_randomization",
                "trial": trial,
                "model": model_name,
                "identity_accuracy": res["identity_accuracy"],
                "skill_score": res["skill_score"],
            })
            print(f"  Trial {trial}, {model_name}: ID={res['identity_accuracy']:.3f}, Skill={res['skill_score']:.3f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "object_order_audit.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audit", "trial", "model", "identity_accuracy", "skill_score"])
        writer.writeheader()
        writer.writerows(results)

    return results


# =============================================================================
# Audit B: Train/Test Disjoint Initial Position Split
# =============================================================================
def audit_disjoint_init_positions(data_dir, output_dir):
    """Split train/test by initial position regions."""
    print("\n" + "=" * 60)
    print("AUDIT B: Disjoint Initial Position Split")
    print("=" * 60)

    train_data = load_split(data_dir, "train")
    test_data = load_split(data_dir, "identity_test")

    # Split train into left-half and right-half based on initial x-position mean
    train_init_x = train_data["observed_positions"][:, 0, :, 0].mean(axis=1)  # mean x of both objects
    train_left_mask = train_init_x < 32.0
    train_right_mask = ~train_left_mask

    # Split test similarly
    test_init_x = test_data["observed_positions"][:, 0, :, 0].mean(axis=1)
    test_left_mask = test_init_x < 32.0
    test_right_mask = ~test_left_mask

    results = []

    for split_name, train_mask, test_mask in [
        ("left_half", train_left_mask, test_left_mask),
        ("right_half", train_right_mask, test_right_mask),
        ("cross_left_train_right_test", train_left_mask, test_right_mask),
        ("cross_right_train_left_test", train_right_mask, test_left_mask),
    ]:
        if train_mask.sum() < 10 or test_mask.sum() < 10:
            continue

        split_train = {
            "observed_positions": train_data["observed_positions"][train_mask],
            "future_positions": train_data["future_positions"][train_mask],
            "identity_labels": train_data["identity_labels"][train_mask],
        }
        split_test = {
            "observed_positions": test_data["observed_positions"][test_mask],
            "future_positions": test_data["future_positions"][test_mask],
            "identity_labels": test_data["identity_labels"][test_mask],
        }

        for model_name, model_class in [
            ("RawTrajectoryKNN", KNN_REGISTRY["RawTrajectoryKNN"]),
        ]:
            res = evaluate_knn_identity(model_class, split_train, split_test, k=5)
            results.append({
                "audit": "disjoint_init",
                "split": split_name,
                "train_size": int(train_mask.sum()),
                "test_size": int(test_mask.sum()),
                "model": model_name,
                "identity_accuracy": res["identity_accuracy"],
                "skill_score": res["skill_score"],
            })
            print(f"  {split_name}: train={train_mask.sum()}, test={test_mask.sum()}, ID={res['identity_accuracy']:.3f}")

    with open(os.path.join(output_dir, "disjoint_init_audit.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audit", "split", "train_size", "test_size", "model", "identity_accuracy", "skill_score"])
        writer.writeheader()
        writer.writerows(results)

    return results


# =============================================================================
# Audit D: Identity Label Permutation Sanity Check
# =============================================================================
def audit_label_permutation(data_dir, output_dir):
    """Randomly permute identity labels in test set."""
    print("\n" + "=" * 60)
    print("AUDIT D: Identity Label Permutation")
    print("=" * 60)

    train_data = load_split(data_dir, "train")
    test_data = load_split(data_dir, "identity_test")

    results = []
    rng = np.random.RandomState(42)

    for trial in range(5):
        # Permute test identity labels
        perm_test_ids = test_data["identity_labels"].copy()
        for i in range(len(perm_test_ids)):
            if rng.random() < 0.5:
                perm_test_ids[i] = perm_test_ids[i][::-1]

        perm_test = {
            "observed_positions": test_data["observed_positions"],
            "future_positions": test_data["future_positions"],
            "identity_labels": perm_test_ids,
        }

        for model_name, model_class in [
            ("RawTrajectoryKNN", KNN_REGISTRY["RawTrajectoryKNN"]),
        ]:
            res = evaluate_knn_identity(model_class, train_data, perm_test, k=5)
            results.append({
                "audit": "label_permutation",
                "trial": trial,
                "model": model_name,
                "identity_accuracy": res["identity_accuracy"],
                "skill_score": res["skill_score"],
            })
            print(f"  Trial {trial}, {model_name}: ID={res['identity_accuracy']:.3f}")

    with open(os.path.join(output_dir, "label_permutation_audit.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audit", "trial", "model", "identity_accuracy", "skill_score"])
        writer.writeheader()
        writer.writerows(results)

    return results


# =============================================================================
# Audit E: Absolute Position Ablation
# =============================================================================
def audit_position_ablation(data_dir, output_dir):
    """Compare RawKNN vs TranslationNormalizedKNN vs VelocityOnlyKNN."""
    print("\n" + "=" * 60)
    print("AUDIT E: Absolute Position Ablation")
    print("=" * 60)

    train_data = load_split(data_dir, "train")
    test_data = load_split(data_dir, "identity_test")

    results = []

    for model_name, model_class in [
        ("RawTrajectoryKNN", KNN_REGISTRY["RawTrajectoryKNN"]),
        ("TranslationNormalizedKNN", KNN_REGISTRY["TranslationNormalizedKNN"]),
        ("VelocityOnlyKNN", KNN_REGISTRY["VelocityOnlyKNN"]),
    ]:
        for k in [1, 3, 5, 10]:
            res = evaluate_knn_identity(model_class, train_data, test_data, k=k)
            results.append({
                "audit": "position_ablation",
                "model": model_name,
                "k": k,
                "identity_accuracy": res["identity_accuracy"],
                "skill_score": res["skill_score"],
            })
            print(f"  {model_name} (k={k}): ID={res['identity_accuracy']:.3f}, Skill={res['skill_score']:.3f}")

    with open(os.path.join(output_dir, "position_ablation_audit.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audit", "model", "k", "identity_accuracy", "skill_score"])
        writer.writeheader()
        writer.writerows(results)

    return results


# =============================================================================
# Audit F: Nearest Neighbor Source Analysis
# =============================================================================
def audit_nn_source_analysis(data_dir, output_dir):
    """Analyze which train episodes are nearest neighbors for test episodes."""
    print("\n" + "=" * 60)
    print("AUDIT F: Nearest Neighbor Source Analysis")
    print("=" * 60)

    train_data = load_split(data_dir, "train")
    test_data = load_split(data_dir, "identity_test")

    from sklearn.neighbors import NearestNeighbors

    # Fit RawTrajectoryKNN
    train_flat = train_data["observed_positions"].reshape(train_data["observed_positions"].shape[0], -1)
    test_flat = test_data["observed_positions"].reshape(test_data["observed_positions"].shape[0], -1)

    nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn.fit(train_flat)
    distances, indices = nn.kneighbors(test_flat)

    results = []
    correct_mask = []

    for i in range(len(test_data["observed_positions"])):
        # Get the nearest neighbor
        nn_idx = indices[i, 0]
        nn_dist = distances[i, 0]

        # Check if test episode has swap
        test_swap = test_data["identity_labels"][i, 0] == 1

        # Check NN train episode initial position
        nn_init = train_data["observed_positions"][nn_idx, 0]
        test_init = test_data["observed_positions"][i, 0]
        init_pos_diff = np.linalg.norm(nn_init - test_init)

        # Predict identity using trajectory matching
        pred_future = train_data["future_positions"][nn_idx]
        mse_no_swap = np.mean((pred_future - test_data["future_positions"][i]) ** 2)
        swapped_pred = pred_future.copy()
        swapped_pred[:, [0, 1]] = swapped_pred[:, [1, 0]]
        mse_swap = np.mean((swapped_pred - test_data["future_positions"][i]) ** 2)
        pred_swap = mse_swap < mse_no_swap
        correct = pred_swap == test_swap

        correct_mask.append(correct)

        results.append({
            "audit": "nn_source",
            "test_idx": i,
            "nn_idx": int(nn_idx),
            "nn_distance": float(nn_dist),
            "init_pos_diff": float(init_pos_diff),
            "test_swap": int(test_swap),
            "pred_swap": int(pred_swap),
            "correct": int(correct),
        })

    correct_mask = np.array(correct_mask)
    correct_distances = distances[:, 0][correct_mask]
    wrong_distances = distances[:, 0][~correct_mask]

    print(f"  Correct predictions: {correct_mask.sum()}/{len(correct_mask)}")
    if len(correct_distances) > 0:
        print(f"  NN distance (correct): mean={correct_distances.mean():.3f}, std={correct_distances.std():.3f}")
    if len(wrong_distances) > 0:
        print(f"  NN distance (wrong): mean={wrong_distances.mean():.3f}, std={wrong_distances.std():.3f}")

    with open(os.path.join(output_dir, "nn_source_analysis.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audit", "test_idx", "nn_idx", "nn_distance", "init_pos_diff", "test_swap", "pred_swap", "correct"])
        writer.writeheader()
        writer.writerows(results)

    return results


# =============================================================================
# Audit G: Identifiability Probe
# =============================================================================
def audit_identifiability_probe(data_dir, output_dir):
    """Check if objects have distinguishable features."""
    print("\n" + "=" * 60)
    print("AUDIT G: Identifiability Probe")
    print("=" * 60)

    train_data = load_split(data_dir, "train")
    test_data = load_split(data_dir, "identity_test")

    results = []

    # Check 1: Are initial positions of object 0 and object 1 separable?
    train_obj0_init = train_data["observed_positions"][:, 0, 0, :]  # (B, 2)
    train_obj1_init = train_data["observed_positions"][:, 0, 1, :]

    obj0_mean = train_obj0_init.mean(axis=0)
    obj1_mean = train_obj1_init.mean(axis=0)
    obj0_std = train_obj0_init.std(axis=0)
    obj1_std = train_obj1_init.std(axis=0)

    print(f"  Object 0 init: mean={obj0_mean}, std={obj0_std}")
    print(f"  Object 1 init: mean={obj1_mean}, std={obj1_std}")

    # Check 2: Is there a systematic difference between obj0 and obj1 trajectories?
    # Compare trajectory centroids
    train_obj0_traj = train_data["observed_positions"][:, :, 0, :].mean(axis=1)  # (B, 2)
    train_obj1_traj = train_data["observed_positions"][:, :, 1, :].mean(axis=1)

    traj_diff = np.linalg.norm(train_obj0_traj - train_obj1_traj, axis=1).mean()
    print(f"  Mean trajectory centroid distance: {traj_diff:.3f}")

    # Check 3: Can we classify object identity from trajectory alone?
    # Simple classifier: centroid x-position
    from sklearn.linear_model import LogisticRegression

    # Use train data: predict which object (0 or 1) based on trajectory
    X = np.concatenate([
        train_data["observed_positions"][:, :, 0, :].reshape(len(train_data["observed_positions"]), -1),
        train_data["observed_positions"][:, :, 1, :].reshape(len(train_data["observed_positions"]), -1),
    ])
    y = np.concatenate([np.zeros(len(train_data["observed_positions"])), np.ones(len(train_data["observed_positions"]))])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    # Test on identity_test: can we tell which is "object 0" vs "object 1"?
    test_obj0 = test_data["observed_positions"][:, :, 0, :].reshape(len(test_data["observed_positions"]), -1)
    test_obj1 = test_data["observed_positions"][:, :, 1, :].reshape(len(test_data["observed_positions"]), -1)

    pred_obj0 = clf.predict(test_obj0)
    pred_obj1 = clf.predict(test_obj1)

    # If classifier says obj0 is class 0 and obj1 is class 1, that's consistent
    consistency = ((pred_obj0 == 0) & (pred_obj1 == 1)).mean()
    print(f"  Trajectory-based identity consistency: {consistency:.3f}")

    results.append({
        "audit": "identifiability",
        "check": "trajectory_consistency",
        "value": float(consistency),
        "description": "Can distinguish obj0 from obj1 by trajectory shape",
    })

    # Check 4: Is there correlation between initial position and swap?
    test_init = test_data["observed_positions"][:, 0, :, :]
    swap_mask = test_data["identity_labels"][:, 0] == 1

    # Distance from center
    center = np.array([32.0, 32.0])
    dist_to_center = np.linalg.norm(test_init.mean(axis=1) - center, axis=1)
    swap_dist = dist_to_center[swap_mask].mean()
    noswap_dist = dist_to_center[~swap_mask].mean()

    print(f"  Mean dist to center: swap={swap_dist:.3f}, no-swap={noswap_dist:.3f}")

    results.append({
        "audit": "identifiability",
        "check": "center_distance_correlation",
        "swap_mean": float(swap_dist),
        "noswap_mean": float(noswap_dist),
        "description": "Correlation between initial position and swap",
    })

    with open(os.path.join(output_dir, "identifiability_probe.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audit", "check", "value", "swap_mean", "noswap_mean", "description"])
        writer.writeheader()
        writer.writerows(results)

    return results


# =============================================================================
# Main
# =============================================================================
def main():
    data_dir = "data_hard"
    output_dir = "results/svt_v2_1_audit"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("SVT-v2.1 Leakage and Identifiability Audit")
    print("=" * 60)

    # Run all audits
    audit_object_order_randomization(data_dir, output_dir)
    audit_disjoint_init_positions(data_dir, output_dir)
    audit_label_permutation(data_dir, output_dir)
    audit_position_ablation(data_dir, output_dir)
    audit_nn_source_analysis(data_dir, output_dir)
    audit_identifiability_probe(data_dir, output_dir)

    print("\n" + "=" * 60)
    print("All audits complete. Results saved to:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
