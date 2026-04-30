"""
SVT-v2.3 Swap-Only Identity Stress Test
"""

import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_2d_motion_v22 import generate_dataset_v22
from baselines.knn_retriever import KNN_REGISTRY
from baselines.knn_retriever_v2 import KNN_V2_REGISTRY
from baselines.feature_aware_baseline import FeatureAwareIdentityBaseline
from metrics.prediction_metrics import compute_prediction_metrics
from metrics.identity_breakdown import compute_identity_breakdown
from metrics.gated_svt_score import compute_gated_svt_score

OUTPUT_DIR = "results/svt_v2_3_swap_only_identity"
SEED = 0


def save_csv(results, filename, fieldnames):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, filename), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


class RandomIdentityBaseline:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def predict_identity(self, observed_positions, test_future=None):
        B = observed_positions.shape[0]
        N = observed_positions.shape[2]
        ids = np.tile(np.arange(N), (B, 1))
        for i in range(B):
            if self.rng.random() < 0.5:
                ids[i] = np.array([1, 0])
        return ids


def evaluate_knn_with_breakdown(model_class, train_data, test_data, k=5):
    model = model_class(k=k, weighting="inverse_distance")
    model.fit(train_data["observed_positions"], train_data["future_positions"],
              train_data.get("identity_labels"))
    pred_future = model.predict_future(test_data["observed_positions"])
    pred_metrics = compute_prediction_metrics(pred_future, test_data["future_positions"])
    pred_identity = model.predict_identity(test_data["observed_positions"],
                                            test_future=test_data["future_positions"])
    breakdown = compute_identity_breakdown(pred_identity, test_data["identity_labels"])
    return {
        "clean_skill": pred_metrics["skill_score"],
        "breakdown": breakdown,
    }


def evaluate_feature_aware_with_breakdown(train_data, test_data):
    fab = FeatureAwareIdentityBaseline()
    obs_feat = test_data.get("object_features_obs")
    fut_feat = test_data.get("object_features_fut")
    if obs_feat is not None and fut_feat is not None:
        pred_ids = fab.predict_identity(obs_feat, fut_feat)
        breakdown = compute_identity_breakdown(pred_ids, test_data["identity_labels"])
    else:
        pred_ids = np.tile(np.arange(2), (test_data["identity_labels"].shape[0], 1))
        rng_rand = np.random.RandomState(42)
        for i in range(len(pred_ids)):
            if rng_rand.random() < 0.5:
                pred_ids[i] = np.array([1, 0])
        breakdown = compute_identity_breakdown(pred_ids, test_data["identity_labels"])
    return {"clean_skill": 0.0, "breakdown": breakdown}


def evaluate_random_with_breakdown(test_data, seed=42):
    rand_model = RandomIdentityBaseline(seed=seed)
    pred_ids = rand_model.predict_identity(test_data["observed_positions"])
    breakdown = compute_identity_breakdown(pred_ids, test_data["identity_labels"])
    return {"clean_skill": 0.0, "breakdown": breakdown}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SVT-v2.3 Swap-Only Identity Stress Test")
    print("=" * 60)

    # =========================================================================
    # Step 1: Generate datasets
    # =========================================================================
    print("\n--- Generating v2.3 datasets ---")

    datasets = {}
    for feature_mode in ["featureless", "feature_bearing"]:
        ds = generate_dataset_v22(
            feature_mode=feature_mode,
            randomize_object_order=True,
            disjoint_init_split=True,
            n_identity_test=200,
            n_identity_test_swap_only=200,
            n_identity_test_no_swap_only=200,
            seed=SEED,
        )
        datasets[feature_mode] = ds

        for split_name in ds:
            n = len(ds[split_name]["observed_positions"])
            labels = ds[split_name]["identity_labels"]
            n_swap = np.sum(labels[:, 0] != 0)
            print(f"  {feature_mode} {split_name}: {n} episodes, {n_swap} swap")

    # =========================================================================
    # Step 2: Identity Breakdown across all models and splits
    # =========================================================================
    print("\n--- Identity Breakdown ---")
    breakdown_results = []

    test_splits = ["identity_test_mixed", "identity_test_swap_only", "identity_test_no_swap_only"]

    for feature_mode in ["featureless", "feature_bearing"]:
        ds = datasets[feature_mode]
        train_data = ds["train"]

        for split_name in test_splits:
            test_data = ds[split_name]

            # RawTrajectoryKNN
            res = evaluate_knn_with_breakdown(KNN_REGISTRY["RawTrajectoryKNN"],
                                               train_data, test_data, k=5)
            bd = res["breakdown"]
            breakdown_results.append({
                "seed": SEED, "feature_mode": feature_mode,
                "split_name": split_name, "model": "RawTrajectoryKNN",
                "identity_overall": f"{bd['identity_overall']:.4f}",
                "identity_no_swap": f"{bd['identity_no_swap']:.4f}" if not np.isnan(bd['identity_no_swap']) else "nan",
                "identity_swap_only": f"{bd['identity_swap_only']:.4f}" if not np.isnan(bd['identity_swap_only']) else "nan",
                "swap_detect_recall": f"{bd['swap_detect_recall']:.4f}" if not np.isnan(bd['swap_detect_recall']) else "nan",
                "swap_false_positive_rate": f"{bd['swap_false_positive_rate']:.4f}" if not np.isnan(bd['swap_false_positive_rate']) else "nan",
                "balanced_identity": f"{bd['balanced_identity']:.4f}" if not np.isnan(bd['balanced_identity']) else "nan",
                "sample_count": len(test_data["observed_positions"]),
            })
            print(f"  {feature_mode} {split_name} RawKNN: overall={bd['identity_overall']:.3f} swap={bd['identity_swap_only']:.3f} noswap={bd['identity_no_swap']:.3f}")

            # RawDeltaKNN
            res = evaluate_knn_with_breakdown(KNN_V2_REGISTRY["RawDeltaKNN"],
                                               train_data, test_data, k=5)
            bd = res["breakdown"]
            breakdown_results.append({
                "seed": SEED, "feature_mode": feature_mode,
                "split_name": split_name, "model": "RawDeltaKNN",
                "identity_overall": f"{bd['identity_overall']:.4f}",
                "identity_no_swap": f"{bd['identity_no_swap']:.4f}" if not np.isnan(bd['identity_no_swap']) else "nan",
                "identity_swap_only": f"{bd['identity_swap_only']:.4f}" if not np.isnan(bd['identity_swap_only']) else "nan",
                "swap_detect_recall": f"{bd['swap_detect_recall']:.4f}" if not np.isnan(bd['swap_detect_recall']) else "nan",
                "swap_false_positive_rate": f"{bd['swap_false_positive_rate']:.4f}" if not np.isnan(bd['swap_false_positive_rate']) else "nan",
                "balanced_identity": f"{bd['balanced_identity']:.4f}" if not np.isnan(bd['balanced_identity']) else "nan",
                "sample_count": len(test_data["observed_positions"]),
            })
            print(f"  {feature_mode} {split_name} RawDeltaKNN: overall={bd['identity_overall']:.3f} swap={bd['identity_swap_only']:.3f}")

            # TranslationNormalizedKNN
            res = evaluate_knn_with_breakdown(KNN_REGISTRY["TranslationNormalizedKNN"],
                                               train_data, test_data, k=5)
            bd = res["breakdown"]
            breakdown_results.append({
                "seed": SEED, "feature_mode": feature_mode,
                "split_name": split_name, "model": "TranslationNormalizedKNN",
                "identity_overall": f"{bd['identity_overall']:.4f}",
                "identity_no_swap": f"{bd['identity_no_swap']:.4f}" if not np.isnan(bd['identity_no_swap']) else "nan",
                "identity_swap_only": f"{bd['identity_swap_only']:.4f}" if not np.isnan(bd['identity_swap_only']) else "nan",
                "swap_detect_recall": f"{bd['swap_detect_recall']:.4f}" if not np.isnan(bd['swap_detect_recall']) else "nan",
                "swap_false_positive_rate": f"{bd['swap_false_positive_rate']:.4f}" if not np.isnan(bd['swap_false_positive_rate']) else "nan",
                "balanced_identity": f"{bd['balanced_identity']:.4f}" if not np.isnan(bd['balanced_identity']) else "nan",
                "sample_count": len(test_data["observed_positions"]),
            })
            print(f"  {feature_mode} {split_name} TransNormKNN: overall={bd['identity_overall']:.3f} swap={bd['identity_swap_only']:.3f}")

            # RandomIdentityBaseline
            res = evaluate_random_with_breakdown(test_data)
            bd = res["breakdown"]
            breakdown_results.append({
                "seed": SEED, "feature_mode": feature_mode,
                "split_name": split_name, "model": "RandomIdentityBaseline",
                "identity_overall": f"{bd['identity_overall']:.4f}",
                "identity_no_swap": f"{bd['identity_no_swap']:.4f}" if not np.isnan(bd['identity_no_swap']) else "nan",
                "identity_swap_only": f"{bd['identity_swap_only']:.4f}" if not np.isnan(bd['identity_swap_only']) else "nan",
                "swap_detect_recall": f"{bd['swap_detect_recall']:.4f}" if not np.isnan(bd['swap_detect_recall']) else "nan",
                "swap_false_positive_rate": f"{bd['swap_false_positive_rate']:.4f}" if not np.isnan(bd['swap_false_positive_rate']) else "nan",
                "balanced_identity": f"{bd['balanced_identity']:.4f}" if not np.isnan(bd['balanced_identity']) else "nan",
                "sample_count": len(test_data["observed_positions"]),
            })
            print(f"  {feature_mode} {split_name} Random: overall={bd['identity_overall']:.3f} swap={bd['identity_swap_only']:.3f}")

            # FeatureAwareIdentityBaseline
            res = evaluate_feature_aware_with_breakdown(train_data, test_data)
            bd = res["breakdown"]
            breakdown_results.append({
                "seed": SEED, "feature_mode": feature_mode,
                "split_name": split_name, "model": "FeatureAwareIdentityBaseline",
                "identity_overall": f"{bd['identity_overall']:.4f}",
                "identity_no_swap": f"{bd['identity_no_swap']:.4f}" if not np.isnan(bd['identity_no_swap']) else "nan",
                "identity_swap_only": f"{bd['identity_swap_only']:.4f}" if not np.isnan(bd['identity_swap_only']) else "nan",
                "swap_detect_recall": f"{bd['swap_detect_recall']:.4f}" if not np.isnan(bd['swap_detect_recall']) else "nan",
                "swap_false_positive_rate": f"{bd['swap_false_positive_rate']:.4f}" if not np.isnan(bd['swap_false_positive_rate']) else "nan",
                "balanced_identity": f"{bd['balanced_identity']:.4f}" if not np.isnan(bd['balanced_identity']) else "nan",
                "sample_count": len(test_data["observed_positions"]),
            })
            print(f"  {feature_mode} {split_name} FAB: overall={bd['identity_overall']:.3f} swap={bd['identity_swap_only']:.3f}")

    save_csv(breakdown_results, "identity_breakdown.csv",
             ["seed", "feature_mode", "split_name", "model",
              "identity_overall", "identity_no_swap", "identity_swap_only",
              "swap_detect_recall", "swap_false_positive_rate",
              "balanced_identity", "sample_count"])

    # =========================================================================
    # Step 3: Swap-Only Model Health Check
    # =========================================================================
    print("\n--- Swap-Only Model Health Check ---")
    health_results = []

    for feature_mode in ["featureless", "feature_bearing"]:
        ds = datasets[feature_mode]
        train_data = ds["train"]
        swap_only_data = ds["identity_test_swap_only"]

        for model_name, eval_fn in [
            ("RawTrajectoryKNN", lambda: evaluate_knn_with_breakdown(KNN_REGISTRY["RawTrajectoryKNN"], train_data, swap_only_data, k=5)),
            ("RawDeltaKNN", lambda: evaluate_knn_with_breakdown(KNN_V2_REGISTRY["RawDeltaKNN"], train_data, swap_only_data, k=5)),
            ("TranslationNormalizedKNN", lambda: evaluate_knn_with_breakdown(KNN_REGISTRY["TranslationNormalizedKNN"], train_data, swap_only_data, k=5)),
            ("RandomIdentityBaseline", lambda: evaluate_random_with_breakdown(swap_only_data)),
            ("FeatureAwareIdentityBaseline", lambda: evaluate_feature_aware_with_breakdown(train_data, swap_only_data)),
        ]:
            res = eval_fn()
            bd = res["breakdown"]
            swap_only_id = bd["identity_swap_only"] if not np.isnan(bd["identity_swap_only"]) else bd["identity_overall"]

            if model_name == "FeatureAwareIdentityBaseline" and feature_mode == "feature_bearing":
                status = "PASS" if swap_only_id >= 0.95 else "FAIL"
            elif model_name == "RandomIdentityBaseline":
                status = "PASS" if abs(swap_only_id - 0.5) < 0.15 else "FAIL"
            elif model_name in ["RawTrajectoryKNN", "TranslationNormalizedKNN"]:
                mixed_bd = None
                for r in breakdown_results:
                    if r["model"] == model_name and r["feature_mode"] == feature_mode and r["split_name"] == "identity_test_mixed":
                        mixed_bd = r
                        break
                if mixed_bd is not None:
                    noswap_val = float(mixed_bd["identity_no_swap"]) if mixed_bd["identity_no_swap"] != "nan" else 0.5
                    if noswap_val > 0.9 and swap_only_id < 0.7:
                        status = "NO_SWAP_BIAS"
                    else:
                        status = "PASS"
                else:
                    status = "UNKNOWN"
            else:
                status = "PASS"

            health_results.append({
                "seed": SEED, "feature_mode": feature_mode,
                "model": model_name,
                "swap_only_identity": f"{swap_only_id:.4f}",
                "status": status,
            })
            print(f"  {feature_mode} {model_name}: swap_only_id={swap_only_id:.3f} -> {status}")

    save_csv(health_results, "swap_only_model_health.csv",
             ["seed", "feature_mode", "model", "swap_only_identity", "status"])

    # =========================================================================
    # Step 4: Gated SVT Score v2.3
    # =========================================================================
    print("\n--- Gated SVT Score v2.3 ---")
    gated_results = []

    for feature_mode in ["featureless", "feature_bearing"]:
        ds = datasets[feature_mode]
        train_data = ds["train"]
        mixed_data = ds["identity_test_mixed"]

        for model_name, eval_fn in [
            ("RawTrajectoryKNN", lambda: evaluate_knn_with_breakdown(KNN_REGISTRY["RawTrajectoryKNN"], train_data, mixed_data, k=5)),
            ("RawDeltaKNN", lambda: evaluate_knn_with_breakdown(KNN_V2_REGISTRY["RawDeltaKNN"], train_data, mixed_data, k=5)),
            ("TranslationNormalizedKNN", lambda: evaluate_knn_with_breakdown(KNN_REGISTRY["TranslationNormalizedKNN"], train_data, mixed_data, k=5)),
        ]:
            res = eval_fn()
            bd = res["breakdown"]
            clean_skill = res["clean_skill"]

            gated_overall = compute_gated_svt_score(clean_skill, 0.0, 0.0, bd["identity_overall"])
            gated_swap = compute_gated_svt_score(clean_skill, 0.0, 0.0, bd["identity_swap_only"])

            gated_results.append({
                "seed": SEED, "feature_mode": feature_mode,
                "model": model_name,
                "clean_skill": f"{clean_skill:.4f}",
                "cf_skill": "0.0000",
                "comp_skill": "0.0000",
                "identity_overall": f"{bd['identity_overall']:.4f}",
                "identity_swap_only": f"{bd['identity_swap_only']:.4f}" if not np.isnan(bd['identity_swap_only']) else "nan",
                "gated_score_overall_id": f"{gated_overall['gated_svt_score']:.4f}",
                "gated_score_swap_only_id": f"{gated_swap['gated_svt_score']:.4f}",
            })
            print(f"  {feature_mode} {model_name}: gated_overall={gated_overall['gated_svt_score']:.4f} gated_swap={gated_swap['gated_svt_score']:.4f}")

    save_csv(gated_results, "gated_score_v23.csv",
             ["seed", "feature_mode", "model", "clean_skill", "cf_skill", "comp_skill",
              "identity_overall", "identity_swap_only",
              "gated_score_overall_id", "gated_score_swap_only_id"])

    # =========================================================================
    # Step 5: Plot
    # =========================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        models_plot = ["RawTrajectoryKNN", "RawDeltaKNN", "TranslationNormalizedKNN",
                       "RandomIdentityBaseline", "FeatureAwareIdentityBaseline"]
        splits_plot = ["identity_test_mixed", "identity_test_swap_only", "identity_test_no_swap_only"]

        # Panel 1: Overall identity by split (featureless)
        ax = axes[0, 0]
        x = np.arange(len(models_plot))
        width = 0.25
        for j, split_name in enumerate(splits_plot):
            vals = []
            for m in models_plot:
                found = [r for r in breakdown_results
                         if r["model"] == m and r["feature_mode"] == "featureless" and r["split_name"] == split_name]
                vals.append(float(found[0]["identity_overall"]) if found else 0.0)
            ax.bar(x + j * width, vals, width, label=split_name.replace("identity_test_", ""))
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.replace("Trajectory", "Traj.").replace("Identity", "ID") for m in models_plot],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title("Featureless: Overall Identity by Split")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
        ax.legend(fontsize=7)

        # Panel 2: Overall identity by split (feature_bearing)
        ax = axes[0, 1]
        for j, split_name in enumerate(splits_plot):
            vals = []
            for m in models_plot:
                found = [r for r in breakdown_results
                         if r["model"] == m and r["feature_mode"] == "feature_bearing" and r["split_name"] == split_name]
                vals.append(float(found[0]["identity_overall"]) if found else 0.0)
            ax.bar(x + j * width, vals, width, label=split_name.replace("identity_test_", ""))
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.replace("Trajectory", "Traj.").replace("Identity", "ID") for m in models_plot],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title("Feature-Bearing: Overall Identity by Split")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
        ax.legend(fontsize=7)

        # Panel 3: Swap-only identity comparison
        ax = axes[1, 0]
        for j, fm in enumerate(["featureless", "feature_bearing"]):
            vals = []
            for m in models_plot:
                found = [r for r in breakdown_results
                         if r["model"] == m and r["feature_mode"] == fm and r["split_name"] == "identity_test_swap_only"]
                swap_val = found[0]["identity_swap_only"] if found else "nan"
                vals.append(float(swap_val) if swap_val != "nan" else 0.0)
            ax.bar(x + j * width, vals, width, label=fm)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([m.replace("Trajectory", "Traj.").replace("Identity", "ID") for m in models_plot],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title("Swap-Only Identity (swap_only split)")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
        ax.axhline(y=0.95, color="blue", linestyle="--", alpha=0.5)
        ax.legend(fontsize=7)

        # Panel 4: Balanced identity on mixed split
        ax = axes[1, 1]
        for j, fm in enumerate(["featureless", "feature_bearing"]):
            vals = []
            for m in models_plot:
                found = [r for r in breakdown_results
                         if r["model"] == m and r["feature_mode"] == fm and r["split_name"] == "identity_test_mixed"]
                bal_val = found[0]["balanced_identity"] if found else "nan"
                vals.append(float(bal_val) if bal_val != "nan" else 0.0)
            ax.bar(x + j * width, vals, width, label=fm)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([m.replace("Trajectory", "Traj.").replace("Identity", "ID") for m in models_plot],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title("Balanced Identity (mixed split)")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
        ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "identity_breakdown_plot.png"), dpi=100)
        plt.close()
        print("\n  Saved identity_breakdown_plot.png")
    except Exception as e:
        print(f"\n  Plot failed: {e}")

    # =========================================================================
    # Final Summary
    # =========================================================================
    fab_fb_swap = [r for r in breakdown_results
                   if r["model"] == "FeatureAwareIdentityBaseline"
                   and r["feature_mode"] == "feature_bearing"
                   and r["split_name"] == "identity_test_swap_only"]
    fab_fb_swap_id = float(fab_fb_swap[0]["identity_swap_only"]) if fab_fb_swap else 0.0

    rand_swap = [r for r in breakdown_results
                 if r["model"] == "RandomIdentityBaseline"
                 and r["split_name"] == "identity_test_swap_only"]
    rand_swap_id = np.mean([float(r["identity_swap_only"]) for r in rand_swap]) if rand_swap else 0.0

    rawknn_mixed = [r for r in breakdown_results
                    if r["model"] == "RawTrajectoryKNN"
                    and r["feature_mode"] == "featureless"
                    and r["split_name"] == "identity_test_mixed"]
    rawknn_swap = [r for r in breakdown_results
                   if r["model"] == "RawTrajectoryKNN"
                   and r["feature_mode"] == "featureless"
                   and r["split_name"] == "identity_test_swap_only"]

    rawknn_mixed_overall = float(rawknn_mixed[0]["identity_overall"]) if rawknn_mixed else 0.0
    rawknn_swap_only_id = float(rawknn_swap[0]["identity_swap_only"]) if rawknn_swap else 0.0

    no_swap_bias = rawknn_mixed_overall > 0.7 and rawknn_swap_only_id < 0.7

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"FeatureAwareBaseline feature_bearing swap_only: {fab_fb_swap_id:.4f}")
    print(f"RandomIdentityBaseline swap_only avg: {rand_swap_id:.4f}")
    print(f"RawKNN featureless mixed overall: {rawknn_mixed_overall:.4f}")
    print(f"RawKNN featureless swap_only: {rawknn_swap_only_id:.4f}")
    print(f"NO-SWAP BIAS detected: {no_swap_bias}")

    if fab_fb_swap_id >= 0.95 and abs(rand_swap_id - 0.5) < 0.15:
        if no_swap_bias:
            print("Recommendation: proceed_to_v3_with_swap_only_metric")
        else:
            print("Recommendation: proceed_to_v3")
    else:
        print("Recommendation: fix_identity_pipeline_first")


if __name__ == "__main__":
    main()
