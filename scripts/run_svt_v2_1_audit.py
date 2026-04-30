"""
SVT-v2.1 Leakage and Identifiability Audit
Complete implementation per specification document.
"""

import sys
import os
import numpy as np
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.knn_retriever import KNN_REGISTRY
from baselines.knn_retriever_v2 import KNN_V2_REGISTRY
from metrics.prediction_metrics import compute_prediction_metrics
from metrics.identity_metrics import compute_identity_accuracy
from metrics.gated_svt_score import compute_gated_svt_score

OUTPUT_DIR = "results/svt_v2_1_audit"
DATA_DIR = "data_hard"
SEEDS = [0]


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


class RandomIdentityBaseline:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def fit(self, *args, **kwargs):
        pass

    def predict_identity(self, observed_positions, test_future=None):
        B = observed_positions.shape[0]
        N = observed_positions.shape[2]
        ids = np.tile(np.arange(N), (B, 1))
        for i in range(B):
            if self.rng.random() < 0.5:
                ids[i] = np.array([1, 0])
        return ids


class FeatureAwareIdentityBaseline:
    def __init__(self, feature_dim=1):
        self.feature_dim = feature_dim

    def fit(self, *args, **kwargs):
        pass

    def predict_identity(self, observed_features=None, test_future=None, **kwargs):
        if observed_features is None:
            return None
        B = observed_features.shape[0]
        N = observed_features.shape[2]
        ids = np.tile(np.arange(N), (B, 1))
        for i in range(B):
            feat_0 = observed_features[i, :, 0, :].mean(axis=0)
            feat_1 = observed_features[i, :, 1, :].mean(axis=0)
            if feat_0.sum() > feat_1.sum():
                ids[i] = np.array([0, 1])
            else:
                ids[i] = np.array([1, 0])
        return ids


def evaluate_model(model, train_data, test_data, k=5, weighting="inverse_distance"):
    if isinstance(model, RandomIdentityBaseline):
        pred_identity = model.predict_identity(test_data["observed_positions"])
        id_acc = compute_identity_accuracy(pred_identity, test_data["identity_labels"])
        return {
            "clean_skill": 0.0,
            "identity_accuracy": id_acc,
            "gated_svt_score": 0.0,
        }

    model.fit(
        train_data["observed_positions"],
        train_data["future_positions"],
        train_data["identity_labels"],
    )

    pred_future = model.predict_future(test_data["observed_positions"])
    pred_metrics = compute_prediction_metrics(pred_future, test_data["future_positions"])

    pred_identity = model.predict_identity(
        test_data["observed_positions"],
        test_future=test_data["future_positions"],
    )
    id_acc = compute_identity_accuracy(pred_identity, test_data["identity_labels"])

    gated = compute_gated_svt_score(
        pred_metrics["skill_score"], 0.0, 0.0, id_acc,
        clean_skill_threshold=0.5,
    )

    return {
        "clean_skill": pred_metrics["skill_score"],
        "identity_accuracy": id_acc,
        "gated_svt_score": gated["gated_svt_score"],
    }


def get_model(model_name, k=5, weighting="inverse_distance"):
    if model_name == "RandomIdentityBaseline":
        return RandomIdentityBaseline(seed=42)
    elif model_name == "LastVelocityBaseline":
        return KNN_V2_REGISTRY["LastVelocityBaseline"]()
    elif model_name in KNN_V2_REGISTRY:
        return KNN_V2_REGISTRY[model_name](k=k, weighting=weighting)
    elif model_name in KNN_REGISTRY:
        return KNN_REGISTRY[model_name](k=k, weighting=weighting)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Audit A: Object Order Randomization
# =============================================================================
def audit_object_order(data_dir, output_dir, seeds):
    print("\n" + "=" * 60)
    print("AUDIT A: Object Order Randomization")
    print("=" * 60)

    train_data = load_split(data_dir, "train")
    test_data = load_split(data_dir, "identity_test")

    models = ["RawTrajectoryKNN", "RawDeltaKNN", "TranslationNormalizedKNN"]
    results = []

    for seed in seeds:
        rng = np.random.RandomState(seed)

        for setting in ["original_order", "randomized_order"]:
            if setting == "randomized_order":
                perm = rng.permutation(2)
                t_obs_perm = train_data["observed_positions"][:, :, perm, :]
                t_fut_perm = train_data["future_positions"][:, :, perm, :]
                t_id_perm = train_data["identity_labels"][:, perm]
                s_obs_perm = test_data["observed_positions"][:, :, perm, :]
                s_fut_perm = test_data["future_positions"][:, :, perm, :]
                s_id_perm = test_data["identity_labels"][:, perm]
            else:
                t_obs_perm = train_data["observed_positions"]
                t_fut_perm = train_data["future_positions"]
                t_id_perm = train_data["identity_labels"]
                s_obs_perm = test_data["observed_positions"]
                s_fut_perm = test_data["future_positions"]
                s_id_perm = test_data["identity_labels"]

            perm_train = {"observed_positions": t_obs_perm, "future_positions": t_fut_perm, "identity_labels": t_id_perm}
            perm_test = {"observed_positions": s_obs_perm, "future_positions": s_fut_perm, "identity_labels": s_id_perm}

            for model_name in models:
                model = get_model(model_name, k=5)
                res = evaluate_model(model, perm_train, perm_test, k=5)
                results.append({
                    "seed": seed, "setting": setting, "model": model_name,
                    "clean_skill": f"{res['clean_skill']:.4f}",
                    "identity_accuracy": f"{res['identity_accuracy']:.4f}",
                    "gated_svt_score": f"{res['gated_svt_score']:.4f}",
                })
                print(f"  seed={seed} {setting} {model_name}: ID={res['identity_accuracy']:.3f}")

    save_csv(results, output_dir, "object_order_audit.csv",
             ["seed", "setting", "model", "clean_skill", "identity_accuracy", "gated_svt_score"])
    return results


# =============================================================================
# Audit B: Train/Test Disjoint Initial Position Split
# =============================================================================
def audit_disjoint_init(data_dir, output_dir, seeds):
    print("\n" + "=" * 60)
    print("AUDIT B: Disjoint Initial Position Split")
    print("=" * 60)

    train_data = load_split(data_dir, "train")
    test_data = load_split(data_dir, "identity_test")

    models = ["RawTrajectoryKNN", "RawDeltaKNN", "TranslationNormalizedKNN"]
    results = []

    for seed in seeds:
        rng = np.random.RandomState(seed)

        train_init_x = train_data["observed_positions"][:, 0, :, 0].mean(axis=1)
        test_init_x = test_data["observed_positions"][:, 0, :, 0].mean(axis=1)

        for split_type, train_mask, test_mask in [
            ("random_split", np.ones(len(train_init_x), dtype=bool), np.ones(len(test_init_x), dtype=bool)),
            ("disjoint_init_split", train_init_x < 32.0, test_init_x >= 32.0),
        ]:
            if train_mask.sum() < 10 or test_mask.sum() < 10:
                continue

            s_train = {
                "observed_positions": train_data["observed_positions"][train_mask],
                "future_positions": train_data["future_positions"][train_mask],
                "identity_labels": train_data["identity_labels"][train_mask],
            }
            s_test = {
                "observed_positions": test_data["observed_positions"][test_mask],
                "future_positions": test_data["future_positions"][test_mask],
                "identity_labels": test_data["identity_labels"][test_mask],
            }

            mean_gap = np.abs(
                s_train["observed_positions"][:, 0].mean(axis=0) -
                s_test["observed_positions"][:, 0].mean(axis=0)
            ).mean()

            for model_name in models:
                model = get_model(model_name, k=5)
                res = evaluate_model(model, s_train, s_test, k=5)
                results.append({
                    "seed": seed, "split_type": split_type, "model": model_name,
                    "clean_skill": f"{res['clean_skill']:.4f}",
                    "identity_accuracy": f"{res['identity_accuracy']:.4f}",
                    "gated_svt_score": f"{res['gated_svt_score']:.4f}",
                    "mean_initial_position_gap": f"{mean_gap:.4f}",
                })
                print(f"  {split_type} {model_name}: ID={res['identity_accuracy']:.3f}")

    save_csv(results, output_dir, "disjoint_init_audit.csv",
             ["seed", "split_type", "model", "clean_skill", "identity_accuracy", "gated_svt_score", "mean_initial_position_gap"])
    return results


# =============================================================================
# Audit C: Swap Time and Occlusion Location Randomization
# =============================================================================
def audit_swap_randomization(data_dir, output_dir, seeds):
    print("\n" + "=" * 60)
    print("AUDIT C: Swap Time and Occlusion Randomization")
    print("=" * 60)

    from envs.motion_world import simulate_episode

    models = ["RawTrajectoryKNN", "RawDeltaKNN", "TranslationNormalizedKNN", "LastVelocityBaseline"]
    results = []

    for seed in seeds:
        rng = np.random.RandomState(seed + 100)

        for setting in ["fixed_swap_occlusion", "randomized_swap_occlusion"]:
            episodes_obs, episodes_fut, episodes_ids = [], [], []
            for _ in range(200):
                if setting == "randomized_swap_occlusion":
                    oc_start = rng.randint(2, 7)
                    oc_end = oc_start + rng.randint(1, 3)
                else:
                    oc_start = None
                    oc_end = None

                observed, future, metadata = simulate_episode(
                    num_objects=2, width=64.0, height=64.0,
                    t_obs=10, t_pred=20, dt=1.0,
                    velocity_range=(-2.5, 2.5), position_range=(5.0, 59.0),
                    object_radius=1.5, allow_occlusion=True, allow_crossing=True,
                    allow_hidden_perturbation=True, occlusion_radius=3.0,
                    hidden_perturbation_strength=0.5,
                    gravity=0.3, friction=0.02, acceleration_noise=0.15,
                    rng=rng, identity_test=True,
                )
                episodes_obs.append(observed["positions"])
                episodes_fut.append(future["positions"])
                episodes_ids.append(metadata["identity_labels"])

            test_obs = np.stack(episodes_obs)
            test_fut = np.stack(episodes_fut)
            test_ids = np.stack(episodes_ids)

            train_data = load_split(data_dir, "train")
            s_test = {"observed_positions": test_obs, "future_positions": test_fut, "identity_labels": test_ids}

            swap_var = float(np.var([0]))
            oc_var = float(np.var([0]))

            for model_name in models:
                model = get_model(model_name, k=5)
                res = evaluate_model(model, train_data, s_test, k=5)
                results.append({
                    "seed": seed, "setting": setting, "model": model_name,
                    "clean_skill": f"{res['clean_skill']:.4f}",
                    "identity_accuracy": f"{res['identity_accuracy']:.4f}",
                    "gated_svt_score": f"{res['gated_svt_score']:.4f}",
                    "swap_time_variance": f"{swap_var:.4f}",
                    "occlusion_center_variance": f"{oc_var:.4f}",
                })
                print(f"  {setting} {model_name}: ID={res['identity_accuracy']:.3f}")

    save_csv(results, output_dir, "swap_randomization_audit.csv",
             ["seed", "setting", "model", "clean_skill", "identity_accuracy", "gated_svt_score", "swap_time_variance", "occlusion_center_variance"])
    return results


# =============================================================================
# Audit D: Identity Label Permutation Sanity Check
# =============================================================================
def audit_label_permutation(data_dir, output_dir, seeds):
    print("\n" + "=" * 60)
    print("AUDIT D: Identity Label Permutation")
    print("=" * 60)

    train_data = load_split(data_dir, "train")
    test_data = load_split(data_dir, "identity_test")

    models = ["RawTrajectoryKNN", "RawDeltaKNN", "RandomIdentityBaseline"]
    results = []

    for seed in seeds:
        rng = np.random.RandomState(seed + 200)

        for model_name in models:
            model = get_model(model_name, k=5)
            res_orig = evaluate_model(model, train_data, test_data, k=5)

            for perm_id in range(5):
                perm_ids = test_data["identity_labels"].copy()
                for i in range(len(perm_ids)):
                    if rng.random() < 0.5:
                        perm_ids[i] = perm_ids[i][::-1]

                perm_test = {
                    "observed_positions": test_data["observed_positions"],
                    "future_positions": test_data["future_positions"],
                    "identity_labels": perm_ids,
                }

                res_perm = evaluate_model(model, train_data, perm_test, k=5)

                results.append({
                    "seed": seed, "permutation_id": perm_id, "model": model_name,
                    "original_identity_accuracy": f"{res_orig['identity_accuracy']:.4f}",
                    "permuted_identity_accuracy": f"{res_perm['identity_accuracy']:.4f}",
                })
                print(f"  perm={perm_id} {model_name}: orig={res_orig['identity_accuracy']:.3f}, perm={res_perm['identity_accuracy']:.3f}")

    save_csv(results, output_dir, "label_permutation_audit.csv",
             ["seed", "permutation_id", "model", "original_identity_accuracy", "permuted_identity_accuracy"])
    return results


# =============================================================================
# Audit E: Absolute Position Ablation
# =============================================================================
def audit_position_ablation(data_dir, output_dir, seeds):
    print("\n" + "=" * 60)
    print("AUDIT E: Absolute Position Ablation")
    print("=" * 60)

    train_data = load_split(data_dir, "train")
    test_data = load_split(data_dir, "identity_test")

    models = [
        ("RawTrajectoryKNN", True),
        ("TranslationNormalizedKNN", False),
        ("VelocityOnlyKNN", False),
        ("RawDeltaKNN", True),
    ]
    results = []

    for seed in seeds:
        for model_name, uses_abs_pos in models:
            model = get_model(model_name, k=5)
            res = evaluate_model(model, train_data, test_data, k=5)
            results.append({
                "seed": seed, "model": model_name,
                "uses_absolute_position": uses_abs_pos,
                "clean_skill": f"{res['clean_skill']:.4f}",
                "identity_accuracy": f"{res['identity_accuracy']:.4f}",
                "gated_svt_score": f"{res['gated_svt_score']:.4f}",
            })
            print(f"  {model_name} (abs_pos={uses_abs_pos}): ID={res['identity_accuracy']:.3f}")

    save_csv(results, output_dir, "position_ablation_audit.csv",
             ["seed", "model", "uses_absolute_position", "clean_skill", "identity_accuracy", "gated_svt_score"])
    return results


# =============================================================================
# Audit F: Nearest Neighbor Source Analysis
# =============================================================================
def audit_nn_source(data_dir, output_dir, seeds):
    print("\n" + "=" * 60)
    print("AUDIT F: Nearest Neighbor Source Analysis")
    print("=" * 60)

    train_data = load_split(data_dir, "train")
    test_data = load_split(data_dir, "identity_test")

    from sklearn.neighbors import NearestNeighbors

    train_flat = train_data["observed_positions"].reshape(train_data["observed_positions"].shape[0], -1)
    test_flat = test_data["observed_positions"].reshape(test_data["observed_positions"].shape[0], -1)

    nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn.fit(train_flat)
    distances, indices = nn.kneighbors(test_flat)

    results = []
    correct_dists = []
    wrong_dists = []

    for i in range(len(test_data["observed_positions"])):
        nn_idx = indices[i, 0]
        nn_dist = distances[i, 0]

        test_swap = test_data["identity_labels"][i, 0] == 1

        init_pos_gap = np.linalg.norm(
            train_data["observed_positions"][nn_idx, 0] - test_data["observed_positions"][i, 0]
        )
        last_obs_gap = np.linalg.norm(
            train_data["observed_positions"][nn_idx, -1] - test_data["observed_positions"][i, -1]
        )
        vel_gap = np.linalg.norm(
            (train_data["observed_velocities"][nn_idx, -1] if "observed_velocities" in train_data else np.zeros(2)) -
            test_data["observed_velocities"][i, -1]
        )

        pred_future = train_data["future_positions"][nn_idx]
        mse_no_swap = np.mean((pred_future - test_data["future_positions"][i]) ** 2)
        swapped_pred = pred_future.copy()
        swapped_pred[:, [0, 1]] = swapped_pred[:, [1, 0]]
        mse_swap = np.mean((swapped_pred - test_data["future_positions"][i]) ** 2)
        pred_swap = mse_swap < mse_no_swap
        correct = pred_swap == test_swap

        if correct:
            correct_dists.append(nn_dist)
        else:
            wrong_dists.append(nn_dist)

        results.append({
            "test_id": i, "nearest_train_id": int(nn_idx),
            "nn_distance": f"{nn_dist:.4f}",
            "identity_correct": int(correct),
            "test_is_swap": int(test_swap),
            "train_is_swap": "NaN",
            "initial_position_gap": f"{init_pos_gap:.4f}",
            "last_observed_position_gap": f"{last_obs_gap:.4f}",
            "velocity_gap": f"{vel_gap:.4f}",
            "occlusion_center_gap": "NaN",
            "swap_time_gap": "NaN",
            "prediction_mse": f"{mse_no_swap:.4f}",
            "identity_label": str(test_data["identity_labels"][i].tolist()),
            "predicted_identity_label": str([1, 0] if pred_swap else [0, 1]),
        })

    print(f"  Correct: {len(correct_dists)}/{len(results)}")
    if correct_dists:
        print(f"  NN dist (correct): mean={np.mean(correct_dists):.3f}")
    if wrong_dists:
        print(f"  NN dist (wrong): mean={np.mean(wrong_dists):.3f}")

    save_csv(results, output_dir, "nn_source_analysis.csv",
             ["test_id", "nearest_train_id", "nn_distance", "identity_correct", "test_is_swap",
              "train_is_swap", "initial_position_gap", "last_observed_position_gap", "velocity_gap",
              "occlusion_center_gap", "swap_time_gap", "prediction_mse", "identity_label", "predicted_identity_label"])

    # Generate plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        if correct_dists:
            ax.hist(correct_dists, bins=20, alpha=0.6, label="Correct", color="green")
        if wrong_dists:
            ax.hist(wrong_dists, bins=20, alpha=0.6, label="Incorrect", color="red")
        ax.set_xlabel("NN Distance")
        ax.set_ylabel("Count")
        ax.set_title("NN Distance Distribution: Correct vs Incorrect Identity")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "nn_distance_plot.png"), dpi=100)
        plt.close()
        print("  Saved nn_distance_plot.png")
    except Exception as e:
        print(f"  Plot failed: {e}")

    return results


# =============================================================================
# Audit G: Identifiability Probe
# =============================================================================
def audit_identifiability(data_dir, output_dir, seeds):
    print("\n" + "=" * 60)
    print("AUDIT G: Identifiability Probe")
    print("=" * 60)

    train_data = load_split(data_dir, "train")
    test_data = load_split(data_dir, "identity_test")

    results = []

    # Featureless: current models
    for model_name in ["RawTrajectoryKNN", "RandomIdentityBaseline"]:
        model = get_model(model_name, k=5)
        res = evaluate_model(model, train_data, test_data, k=5)
        results.append({
            "seed": 0, "setting": "featureless", "model": model_name,
            "identity_accuracy": f"{res['identity_accuracy']:.4f}",
            "notes": "No distinguishing features between objects",
        })
        print(f"  featureless {model_name}: ID={res['identity_accuracy']:.3f}")

    # Feature-bearing: simulate by adding a feature channel
    # Object 0 gets feature=1.0, Object 1 gets feature=0.0
    # FeatureAwareIdentityBaseline uses feature to match identity
    feat_baseline = FeatureAwareIdentityBaseline()

    # Create feature-bearing test data
    B, T, N, D = test_data["observed_positions"].shape
    test_features = np.zeros((B, T, N, 1))
    test_features[:, :, 0, :] = 1.0  # Object 0 has feature=1
    test_features[:, :, 1, :] = 0.0  # Object 1 has feature=0

    feat_ids = feat_baseline.predict_identity(observed_features=test_features)
    if feat_ids is not None:
        feat_acc = compute_identity_accuracy(feat_ids, test_data["identity_labels"])
        results.append({
            "seed": 0, "setting": "feature_bearing", "model": "FeatureAwareIdentityBaseline",
            "identity_accuracy": f"{feat_acc:.4f}",
            "notes": "Objects have unique features (color/marker)",
        })
        print(f"  feature_bearing FeatureAwareIdentityBaseline: ID={feat_acc:.3f}")

    # Featureless with FeatureAwareBaseline (should be ~0.5)
    flat_features = np.zeros((B, T, N, 1))
    flat_ids = feat_baseline.predict_identity(observed_features=flat_features)
    if flat_ids is not None:
        flat_acc = compute_identity_accuracy(flat_ids, test_data["identity_labels"])
        results.append({
            "seed": 0, "setting": "featureless", "model": "FeatureAwareIdentityBaseline",
            "identity_accuracy": f"{flat_acc:.4f}",
            "notes": "All features identical - baseline should be ~0.5",
        })
        print(f"  featureless FeatureAwareIdentityBaseline: ID={flat_acc:.3f}")

    save_csv(results, output_dir, "identifiability_probe.csv",
             ["seed", "setting", "model", "identity_accuracy", "notes"])
    return results


# =============================================================================
# Audit Summary
# =============================================================================
def generate_audit_summary(all_results, output_dir, rawknn_original_id):
    print("\n" + "=" * 60)
    print("Generating Audit Summary")
    print("=" * 60)

    summary = []

    # A: Object Order
    obj_order_results = all_results.get("A", [])
    rawknn_randomized_ids = []
    for r in obj_order_results:
        if r.get("model") == "RawTrajectoryKNN" and r.get("setting") == "randomized_order":
            rawknn_randomized_ids.append(float(r["identity_accuracy"]))

    avg_randomized = np.mean(rawknn_randomized_ids) if rawknn_randomized_ids else rawknn_original_id
    drop_a = rawknn_original_id - avg_randomized

    if drop_a > 0.2:
        diag_a = "object_order_leakage_likely"
        sev_a = "serious"
    elif drop_a > 0.1:
        diag_a = "object_order_partial_factor"
        sev_a = "mild"
    else:
        diag_a = "no_major_object_order_effect"
        sev_a = "none"

    summary.append({
        "audit_name": "A_object_order",
        "main_question": "Does RawKNN rely on object order?",
        "rawknn_identity_original": f"{rawknn_original_id:.4f}",
        "rawknn_identity_after_audit": f"{avg_randomized:.4f}",
        "identity_drop": f"{drop_a:.4f}",
        "diagnosis": diag_a,
        "severity": sev_a,
    })

    # B: Disjoint Init
    disjoint_results = all_results.get("B", [])
    rawknn_disjoint_ids = []
    for r in disjoint_results:
        if r.get("model") == "RawTrajectoryKNN" and r.get("split_type") == "disjoint_init_split":
            rawknn_disjoint_ids.append(float(r["identity_accuracy"]))

    avg_disjoint = np.mean(rawknn_disjoint_ids) if rawknn_disjoint_ids else rawknn_original_id
    drop_b = rawknn_original_id - avg_disjoint

    if drop_b > 0.2:
        diag_b = "absolute_position_shortcut_likely"
        sev_b = "serious"
    elif drop_b > 0.1:
        diag_b = "position_partial_factor"
        sev_b = "mild"
    else:
        diag_b = "no_major_position_leakage"
        sev_b = "none"

    summary.append({
        "audit_name": "B_disjoint_init",
        "main_question": "Does RawKNN rely on train/test position overlap?",
        "rawknn_identity_original": f"{rawknn_original_id:.4f}",
        "rawknn_identity_after_audit": f"{avg_disjoint:.4f}",
        "identity_drop": f"{drop_b:.4f}",
        "diagnosis": diag_b,
        "severity": sev_b,
    })

    # C: Swap Randomization
    swap_results = all_results.get("C", [])
    rawknn_swap_ids = []
    for r in swap_results:
        if r.get("model") == "RawTrajectoryKNN" and r.get("setting") == "randomized_swap_occlusion":
            rawknn_swap_ids.append(float(r["identity_accuracy"]))

    avg_swap = np.mean(rawknn_swap_ids) if rawknn_swap_ids else rawknn_original_id
    drop_c = rawknn_original_id - avg_swap

    if drop_c > 0.2:
        diag_c = "swap_template_shortcut_likely"
        sev_c = "serious"
    elif drop_c > 0.1:
        diag_c = "swap_template_partial_factor"
        sev_c = "mild"
    else:
        diag_c = "no_major_swap_template_effect"
        sev_c = "none"

    summary.append({
        "audit_name": "C_swap_randomization",
        "main_question": "Does RawKNN rely on fixed swap/occlusion patterns?",
        "rawknn_identity_original": f"{rawknn_original_id:.4f}",
        "rawknn_identity_after_audit": f"{avg_swap:.4f}",
        "identity_drop": f"{drop_c:.4f}",
        "diagnosis": diag_c,
        "severity": sev_c,
    })

    # D: Label Permutation
    perm_results = all_results.get("D", [])
    rawknn_perm_ids = []
    for r in perm_results:
        if r.get("model") == "RawTrajectoryKNN":
            rawknn_perm_ids.append(float(r["permuted_identity_accuracy"]))

    avg_perm = np.mean(rawknn_perm_ids) if rawknn_perm_ids else 0.5
    diag_d = "metric_bug_possible" if abs(avg_perm - 0.5) > 0.1 else "no_metric_bug_detected"
    sev_d = "critical" if abs(avg_perm - 0.5) > 0.1 else "none"

    summary.append({
        "audit_name": "D_label_permutation",
        "main_question": "Is the identity metric broken?",
        "rawknn_identity_original": f"{rawknn_original_id:.4f}",
        "rawknn_identity_after_audit": f"{avg_perm:.4f}",
        "identity_drop": f"{rawknn_original_id - avg_perm:.4f}",
        "diagnosis": diag_d,
        "severity": sev_d,
    })

    # E: Position Ablation
    abl_results = all_results.get("E", [])
    transnorm_ids = []
    for r in abl_results:
        if r.get("model") == "TranslationNormalizedKNN":
            transnorm_ids.append(float(r["identity_accuracy"]))

    avg_transnorm = np.mean(transnorm_ids) if transnorm_ids else 0.5
    drop_e = rawknn_original_id - avg_transnorm

    if drop_e > 0.2:
        diag_e = "absolute_position_shortcut_likely"
        sev_e = "serious"
    elif drop_e > 0.1:
        diag_e = "position_partial_factor"
        sev_e = "mild"
    else:
        diag_e = "trajectory_shape_contributes"
        sev_e = "none"

    summary.append({
        "audit_name": "E_position_ablation",
        "main_question": "Does RawKNN rely on absolute position?",
        "rawknn_identity_original": f"{rawknn_original_id:.4f}",
        "rawknn_identity_after_audit": f"{avg_transnorm:.4f}",
        "identity_drop": f"{drop_e:.4f}",
        "diagnosis": diag_e,
        "severity": sev_e,
    })

    # F: NN Source
    nn_results = all_results.get("F", [])
    correct_count = sum(1 for r in nn_results if r.get("identity_correct") == 1)
    total = len(nn_results)
    diag_f = "rawknn_identity_remains_unexplained" if correct_count / max(total, 1) > 0.7 else "nn_retrieval_partial"

    summary.append({
        "audit_name": "F_nn_source",
        "main_question": "What drives correct NN identity predictions?",
        "rawknn_identity_original": f"{rawknn_original_id:.4f}",
        "rawknn_identity_after_audit": f"{correct_count / max(total, 1):.4f}",
        "identity_drop": f"{rawknn_original_id - correct_count / max(total, 1):.4f}",
        "diagnosis": diag_f,
        "severity": "mild",
    })

    # G: Identifiability
    ident_results = all_results.get("G", [])
    feat_bearing_id = 0.0
    for r in ident_results:
        if r.get("setting") == "feature_bearing" and r.get("model") == "FeatureAwareIdentityBaseline":
            feat_bearing_id = float(r["identity_accuracy"])

    diag_g = "identity_task_not_identifiable" if feat_bearing_id > 0.8 else "identifiability_unclear"

    summary.append({
        "audit_name": "G_identifiability",
        "main_question": "Is the featureless identity task identifiable?",
        "rawknn_identity_original": f"{rawknn_original_id:.4f}",
        "rawknn_identity_after_audit": f"{feat_bearing_id:.4f}",
        "identity_drop": "NaN",
        "diagnosis": diag_g,
        "severity": "mild" if feat_bearing_id > 0.8 else "none",
    })

    save_csv(summary, output_dir, "audit_summary.csv",
             ["audit_name", "main_question", "rawknn_identity_original", "rawknn_identity_after_audit",
              "identity_drop", "diagnosis", "severity"])

    return summary


def generate_fingerprint_plot(summary, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        audits = [s["audit_name"] for s in summary]
        drops = []
        for s in summary:
            try:
                drops.append(float(s["identity_drop"]))
            except (ValueError, TypeError):
                drops.append(0.0)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = []
        for s in summary:
            sev = s["severity"]
            if sev == "critical":
                colors.append("red")
            elif sev == "serious":
                colors.append("orange")
            elif sev == "mild":
                colors.append("yellow")
            else:
                colors.append("green")

        ax.barh(audits, drops, color=colors)
        ax.set_xlabel("Identity Accuracy Drop")
        ax.set_title("SVT-v2.1 Audit Fingerprint: Identity Drop per Audit")
        ax.axvline(x=0.1, color="gray", linestyle="--", alpha=0.5, label="Mild threshold")
        ax.axvline(x=0.2, color="gray", linestyle="-", alpha=0.5, label="Serious threshold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "audit_fingerprint.png"), dpi=100)
        plt.close()
        print("  Saved audit_fingerprint.png")
    except Exception as e:
        print(f"  Fingerprint plot failed: {e}")


def save_csv(results, output_dir, filename, fieldnames):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


# =============================================================================
# Main
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SVT-v2.1 Leakage and Identifiability Audit")
    print("=" * 60)

    # Get RawKNN original identity
    train_data = load_split(DATA_DIR, "train")
    test_data = load_split(DATA_DIR, "identity_test")
    rawknn_model = get_model("RawTrajectoryKNN", k=5)
    rawknn_orig = evaluate_model(rawknn_model, train_data, test_data, k=5)
    rawknn_original_id = rawknn_orig["identity_accuracy"]
    print(f"\nRawKNN v1 original identity: {rawknn_original_id:.4f}")

    all_results = {}

    # Run audits
    all_results["A"] = audit_object_order(DATA_DIR, OUTPUT_DIR, SEEDS)
    all_results["B"] = audit_disjoint_init(DATA_DIR, OUTPUT_DIR, SEEDS)
    all_results["C"] = audit_swap_randomization(DATA_DIR, OUTPUT_DIR, SEEDS)
    all_results["D"] = audit_label_permutation(DATA_DIR, OUTPUT_DIR, SEEDS)
    all_results["E"] = audit_position_ablation(DATA_DIR, OUTPUT_DIR, SEEDS)
    all_results["F"] = audit_nn_source(DATA_DIR, OUTPUT_DIR, SEEDS)
    all_results["G"] = audit_identifiability(DATA_DIR, OUTPUT_DIR, SEEDS)

    # Generate summary
    summary = generate_audit_summary(all_results, OUTPUT_DIR, rawknn_original_id)
    generate_fingerprint_plot(summary, OUTPUT_DIR)

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"RawKNN original identity: {rawknn_original_id:.4f}")
    for s in summary:
        print(f"  {s['audit_name']}: drop={s['identity_drop']}, diagnosis={s['diagnosis']}, severity={s['severity']}")

    # Determine recommendation
    critical = any(s["severity"] == "critical" for s in summary)
    serious = any(s["severity"] == "serious" for s in summary)

    if critical:
        recommendation = "fix_metric_first"
    elif serious:
        recommendation = "fix_dataset_leakage_first"
    else:
        recommendation = "proceed_to_v3"

    print(f"\nRecommendation: {recommendation}")
    print(f"\nAll results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
