"""
SVT-v3.1: Minimal Learned Models
"""

import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v3_1_learned_models"
SEED = 0
FORCE_TRAIN = "attractor"
FORCE_TEST = "vortex"

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

from data.generate_nonlinear_feature_ood import generate_v3_dataset
from baselines.knn_retriever import KNN_REGISTRY
from baselines.knn_retriever_v2 import KNN_V2_REGISTRY, LastVelocityBaseline
from baselines.feature_aware_baseline import FeatureAwareIdentityBaseline
from metrics.prediction_metrics import compute_prediction_metrics
from metrics.identity_breakdown import compute_identity_breakdown
from metrics.ood_metrics import compute_ood_skill, compute_crossing_occlusion_skill
from metrics.svt_v3_score import compute_gated_svt_score_v3


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


def evaluate_knn_model(model_class, train_data, test_data, k=5, **kwargs):
    model = model_class(k=k, weighting="inverse_distance", **kwargs)
    model.fit(train_data["observed_positions"], train_data["future_positions"],
              train_data.get("identity_labels"))
    pred_future = model.predict_future(test_data["observed_positions"])
    pred_metrics = compute_prediction_metrics(pred_future, test_data["future_positions"])
    pred_identity = model.predict_identity(test_data["observed_positions"],
                                            test_future=test_data["future_positions"])
    breakdown = compute_identity_breakdown(pred_identity, test_data["identity_labels"])
    return pred_future, pred_metrics, pred_identity, breakdown


def evaluate_learned_model(model, test_data, uses_features=False):
    obs_pos = test_data["observed_positions"]
    obs_feat = test_data.get("object_features_obs") if uses_features else None

    pred_future = model.predict_future(obs_pos, obs_feat)
    if isinstance(pred_future, torch.Tensor):
        pred_future = pred_future.cpu().numpy()

    pred_metrics = compute_prediction_metrics(pred_future, test_data["future_positions"])

    pred_identity = model.predict_identity(obs_pos, obs_feat)
    if isinstance(pred_identity, torch.Tensor):
        pred_identity = pred_identity.cpu().numpy()

    breakdown = compute_identity_breakdown(pred_identity, test_data["identity_labels"])
    return pred_future, pred_metrics, pred_identity, breakdown


def evaluate_feature_aware(test_data):
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
    return breakdown


def evaluate_random(test_data, seed=42):
    rand_model = RandomIdentityBaseline(seed=seed)
    pred_ids = rand_model.predict_identity(test_data["observed_positions"])
    breakdown = compute_identity_breakdown(pred_ids, test_data["identity_labels"])
    return breakdown


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SVT-v3.1: Minimal Learned Models")
    print("=" * 60)
    print(f"PyTorch: {'available' if TORCH_AVAILABLE else 'UNAVAILABLE'} (device={DEVICE})")

    # =========================================================================
    # Step 1: Generate v3 dataset
    # =========================================================================
    print("\n--- Generating v3 dataset ---")

    ds = generate_v3_dataset(
        n_train=1000, n_test=200,
        feature_mode="feature_bearing",
        force_train_type=FORCE_TRAIN,
        force_test_type=FORCE_TEST,
        randomize_object_order=True,
        disjoint_init_split=True,
        seed=SEED,
    )

    train_data = ds["train_id"]
    for split_name, split_data in ds.items():
        n = len(split_data["observed_positions"])
        n_swap = int(split_data["is_swap"].sum())
        print(f"  {split_name}: {n} episodes, {n_swap} swap")

    # =========================================================================
    # Step 2: Train learned models
    # =========================================================================
    trained_models = {}
    training_curves = []

    if not TORCH_AVAILABLE:
        print("\n--- PyTorch unavailable, skipping learned models ---")
    else:
        from models.mlp_position_only import MLPPositionOnly
        from models.mlp_position_feature import MLPPositionFeature
        from models.object_centric_feature import ObjectCentricFeatureModel
        from utils.torch_training import train_model

        learned_configs = [
            ("MLPPositionOnly", MLPPositionOnly, False, {}),
            ("MLPPositionFeature", MLPPositionFeature, True, {"identity_weight": 1.0}),
            ("ObjectCentricFeatureModel", ObjectCentricFeatureModel, True, {"identity_weight": 1.0}),
        ]

        for model_name, model_class, uses_feat, extra_kwargs in learned_configs:
            print(f"\n--- Training {model_name} ---")
            try:
                model = model_class(**extra_kwargs)
                log = train_model(
                    model, train_data, val_data=ds["clean_test_id"],
                    epochs=20, batch_size=64, lr=1e-3, device=DEVICE,
                    uses_features=uses_feat, verbose=True,
                )
                trained_models[model_name] = (model, uses_feat)

                for entry in log:
                    training_curves.append({
                        "seed": SEED, "model": model_name,
                        "epoch": entry["epoch"],
                        "train_loss": fmt(entry["train_loss"]),
                        "mse_loss": fmt(entry["mse_loss"]),
                        "identity_loss": fmt(entry["identity_loss"]),
                        "val_clean_skill": fmt(entry["val_clean_skill"]),
                    })
                print(f"  {model_name} trained: final val_skill={fmt(log[-1]['val_clean_skill'])}")
            except Exception as e:
                print(f"  {model_name} FAILED: {e}")
                trained_models[model_name] = (None, uses_feat)

    save_csv(training_curves, "learned_training_curves.csv",
             ["seed", "model", "epoch", "train_loss", "mse_loss", "identity_loss", "val_clean_skill"])

    # =========================================================================
    # Step 3: Evaluate all models on all splits
    # =========================================================================
    print("\n--- Evaluating all models ---")

    comparison_results = []
    identity_results = []
    ood_results = []
    gated_results = []

    # Baseline configs
    baseline_configs = [
        ("LastVelocityBaseline", "baseline", True, False, False),
        ("RawTrajectoryKNN", "baseline", True, False, False),
        ("RawDeltaKNN", "baseline", True, False, False),
        ("TranslationNormalizedKNN", "baseline", True, False, False),
        ("FeatureAwareIdentityBaseline", "baseline", False, True, False),
        ("RandomIdentityBaseline", "baseline", False, False, False),
    ]

    # Learned model configs
    learned_configs_eval = [
        ("MLPPositionOnly", "learned", True, False, False),
        ("MLPPositionFeature", "learned", True, True, False),
        ("ObjectCentricFeatureModel", "learned", True, True, True),
    ]

    all_configs = baseline_configs + learned_configs_eval

    # --- Evaluate on identity splits ---
    for split_name in ["identity_test_mixed", "identity_test_swap_only", "identity_test_no_swap_only"]:
        test_data = ds[split_name]

        for model_name, model_type, uses_pos, uses_feat, is_obj_centric in all_configs:
            breakdown = None
            clean_skill = float("nan")

            if model_name == "FeatureAwareIdentityBaseline":
                breakdown = evaluate_feature_aware(test_data)
            elif model_name == "RandomIdentityBaseline":
                breakdown = evaluate_random(test_data)
            elif model_name == "LastVelocityBaseline":
                model = LastVelocityBaseline()
                model.fit(train_data["observed_positions"], train_data["future_positions"])
                pred_future = model.predict_future(test_data["observed_positions"])
                pred_metrics = compute_prediction_metrics(pred_future, test_data["future_positions"])
                pred_identity = model.predict_identity(test_data["observed_positions"],
                                                        test_future=test_data["future_positions"])
                breakdown = compute_identity_breakdown(pred_identity, test_data["identity_labels"])
                clean_skill = pred_metrics["skill_score"]
            elif model_name in ["RawTrajectoryKNN", "TranslationNormalizedKNN"]:
                registry = KNN_REGISTRY
                pred_fut, pred_met, pred_id, bd = evaluate_knn_model(
                    registry[model_name], train_data, test_data)
                breakdown = bd
                clean_skill = pred_met["skill_score"]
            elif model_name == "RawDeltaKNN":
                pred_fut, pred_met, pred_id, bd = evaluate_knn_model(
                    KNN_V2_REGISTRY["RawDeltaKNN"], train_data, test_data)
                breakdown = bd
                clean_skill = pred_met["skill_score"]
            elif model_name in trained_models:
                model_obj, model_uses_feat = trained_models[model_name]
                if model_obj is None:
                    continue
                pred_fut, pred_met, pred_id, bd = evaluate_learned_model(
                    model_obj, test_data, uses_features=model_uses_feat)
                breakdown = bd
                clean_skill = pred_met["skill_score"]
            else:
                continue

            if breakdown is None:
                continue

            id_overall = breakdown["identity_overall"]
            id_swap = breakdown["identity_swap_only"]
            id_noswap = breakdown["identity_no_swap"]
            gap = id_overall - id_swap if not np.isnan(id_swap) else float("nan")
            bias_flag = gap > 0.1 if not np.isnan(gap) else False

            identity_results.append({
                "seed": SEED, "model": model_name,
                "split_name": split_name, "feature_mode": "feature_bearing",
                "identity_overall": fmt(id_overall),
                "identity_no_swap": fmt(id_noswap),
                "identity_swap_only": fmt(id_swap),
                "swap_detect_recall": fmt(breakdown["swap_detect_recall"]),
                "swap_false_positive_rate": fmt(breakdown["swap_false_positive_rate"]),
                "balanced_identity": fmt(breakdown["balanced_identity"]),
                "sample_count": len(test_data["observed_positions"]),
            })

            if split_name == "identity_test_mixed":
                comparison_results.append({
                    "seed": SEED, "model": model_name, "model_type": model_type,
                    "uses_position": str(uses_pos), "uses_feature": str(uses_feat),
                    "is_object_centric": str(is_obj_centric),
                    "clean_skill": fmt(clean_skill),
                    "crossing_occlusion_skill": "nan",
                    "ood_skill": "nan",
                    "identity_overall": fmt(id_overall),
                    "identity_no_swap": fmt(id_noswap),
                    "identity_swap_only": fmt(id_swap),
                    "balanced_identity": fmt(breakdown["balanced_identity"]),
                    "no_swap_bias_gap": fmt(gap),
                    "no_swap_bias_flag": str(bias_flag),
                })

    # --- Evaluate crossing/occlusion ---
    co_test = ds["crossing_occlusion_test"]
    for model_name, model_type, uses_pos, uses_feat, is_obj_centric in all_configs:
        co_skill_val = float("nan")
        id_cross = float("nan")
        id_occl = float("nan")
        id_both = float("nan")

        if model_name in ["FeatureAwareIdentityBaseline", "RandomIdentityBaseline"]:
            pass
        elif model_name == "LastVelocityBaseline":
            model = LastVelocityBaseline()
            model.fit(train_data["observed_positions"], train_data["future_positions"])
            pred = model.predict_future(co_test["observed_positions"])
            co_skill_data = compute_crossing_occlusion_skill(
                pred, co_test["future_positions"],
                co_test["has_crossing"], co_test["has_occlusion"])
            co_skill_val = co_skill_data.get("overall_skill", float("nan"))
        elif model_name in ["RawTrajectoryKNN", "TranslationNormalizedKNN"]:
            registry_model = KNN_REGISTRY.get(model_name)
            if registry_model:
                m = registry_model(k=5, weighting="inverse_distance")
                m.fit(train_data["observed_positions"], train_data["future_positions"])
                pred = m.predict_future(co_test["observed_positions"])
                co_skill_data = compute_crossing_occlusion_skill(
                    pred, co_test["future_positions"],
                    co_test["has_crossing"], co_test["has_occlusion"])
                co_skill_val = co_skill_data.get("overall_skill", float("nan"))
        elif model_name == "RawDeltaKNN":
            m = KNN_V2_REGISTRY["RawDeltaKNN"](k=5, weighting="inverse_distance")
            m.fit(train_data["observed_positions"], train_data["future_positions"])
            pred = m.predict_future(co_test["observed_positions"])
            co_skill_data = compute_crossing_occlusion_skill(
                pred, co_test["future_positions"],
                co_test["has_crossing"], co_test["has_occlusion"])
            co_skill_val = co_skill_data.get("overall_skill", float("nan"))
        elif model_name in trained_models:
            model_obj, model_uses_feat = trained_models[model_name]
            if model_obj is not None:
                obs_feat = co_test.get("object_features_obs") if model_uses_feat else None
                pred = model_obj.predict_future(co_test["observed_positions"], obs_feat)
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
                co_skill_data = compute_crossing_occlusion_skill(
                    pred, co_test["future_positions"],
                    co_test["has_crossing"], co_test["has_occlusion"])
                co_skill_val = co_skill_data.get("overall_skill", float("nan"))

        for r in comparison_results:
            if r["model"] == model_name:
                r["crossing_occlusion_skill"] = fmt(co_skill_val)

    # --- Evaluate OOD ---
    ood_test = ds["ood_force_test"]
    clean_test = ds["clean_test_id"]

    for model_name, model_type, uses_pos, uses_feat, is_obj_centric in all_configs:
        if model_name in ["FeatureAwareIdentityBaseline", "RandomIdentityBaseline"]:
            continue

        id_pred = None
        ood_pred = None

        if model_name == "LastVelocityBaseline":
            model = LastVelocityBaseline()
            model.fit(train_data["observed_positions"], train_data["future_positions"])
            id_pred = model.predict_future(clean_test["observed_positions"])
            ood_pred = model.predict_future(ood_test["observed_positions"])
        elif model_name in ["RawTrajectoryKNN", "TranslationNormalizedKNN"]:
            registry_model = KNN_REGISTRY.get(model_name)
            if registry_model:
                m = registry_model(k=5, weighting="inverse_distance")
                m.fit(train_data["observed_positions"], train_data["future_positions"])
                id_pred = m.predict_future(clean_test["observed_positions"])
                ood_pred = m.predict_future(ood_test["observed_positions"])
        elif model_name == "RawDeltaKNN":
            m = KNN_V2_REGISTRY["RawDeltaKNN"](k=5, weighting="inverse_distance")
            m.fit(train_data["observed_positions"], train_data["future_positions"])
            id_pred = m.predict_future(clean_test["observed_positions"])
            ood_pred = m.predict_future(ood_test["observed_positions"])
        elif model_name in trained_models:
            model_obj, model_uses_feat = trained_models[model_name]
            if model_obj is not None:
                obs_feat_id = clean_test.get("object_features_obs") if model_uses_feat else None
                obs_feat_ood = ood_test.get("object_features_obs") if model_uses_feat else None
                id_pred = model_obj.predict_future(clean_test["observed_positions"], obs_feat_id)
                ood_pred = model_obj.predict_future(ood_test["observed_positions"], obs_feat_ood)
                if isinstance(id_pred, torch.Tensor):
                    id_pred = id_pred.cpu().numpy()
                if isinstance(ood_pred, torch.Tensor):
                    ood_pred = ood_pred.cpu().numpy()

        if id_pred is None or ood_pred is None:
            continue

        ood_metrics = compute_ood_skill(id_pred, clean_test["future_positions"],
                                         ood_pred, ood_test["future_positions"])

        ood_identity_swap = float("nan")
        ood_gated = 0.0
        if model_name in trained_models:
            model_obj, model_uses_feat = trained_models[model_name]
            if model_obj is not None:
                obs_feat_ood = ood_test.get("object_features_obs") if model_uses_feat else None
                ood_pred_ids = model_obj.predict_identity(ood_test["observed_positions"], obs_feat_ood)
                if isinstance(ood_pred_ids, torch.Tensor):
                    ood_pred_ids = ood_pred_ids.cpu().numpy()
                ood_bd = compute_identity_breakdown(ood_pred_ids, ood_test["identity_labels"])
                ood_identity_swap = ood_bd["identity_swap_only"]
                ood_gated = compute_gated_svt_score_v3(
                    ood_metrics["id_skill"], ood_metrics["ood_skill"], 0.0,
                    ood_identity_swap)["gated_score_swap_only"]

        ood_results.append({
            "seed": SEED, "model": model_name,
            "train_force_type": FORCE_TRAIN, "test_force_type": FORCE_TEST,
            "clean_skill": fmt(ood_metrics["id_skill"]),
            "ood_skill": fmt(ood_metrics["ood_skill"]),
            "ood_skill_drop": fmt(ood_metrics["ood_skill_drop"]),
            "ood_identity_swap_only": fmt(ood_identity_swap),
            "ood_gated_score": fmt(ood_gated),
        })

        for r in comparison_results:
            if r["model"] == model_name:
                r["ood_skill"] = fmt(ood_metrics["ood_skill"])

    # --- Gated scores ---
    for r in comparison_results:
        model_name = r["model"]
        clean_sk = float(r["clean_skill"]) if r["clean_skill"] != "nan" else 0.0
        id_overall = float(r["identity_overall"]) if r["identity_overall"] != "nan" else 0.0
        id_swap = float(r["identity_swap_only"]) if r["identity_swap_only"] != "nan" else 0.0

        ood_sk = 0.0
        for ood_r in ood_results:
            if ood_r["model"] == model_name:
                ood_sk = float(ood_r["ood_skill"]) if ood_r["ood_skill"] != "nan" else 0.0
                break

        co_sk = float(r["crossing_occlusion_skill"]) if r["crossing_occlusion_skill"] != "nan" else 0.0

        gated = compute_gated_svt_score_v3(clean_sk, ood_sk, co_sk, id_swap, id_overall)

        gated_results.append({
            "seed": SEED, "model": model_name,
            "clean_skill": fmt(clean_sk),
            "cf_skill": fmt(ood_sk),
            "comp_skill": fmt(co_sk),
            "identity_overall": fmt(id_overall),
            "identity_swap_only": fmt(id_swap),
            "gated_score_overall_id": fmt(gated["gated_score_overall_id"]),
            "gated_score_swap_only": fmt(gated["gated_score_swap_only"]),
            "score_drop": fmt(gated["score_drop"]),
            "no_swap_bias_flag": str(gated["no_swap_bias_flag"]),
        })

    # =========================================================================
    # Save all CSVs
    # =========================================================================
    save_csv(comparison_results, "learned_model_comparison.csv",
             ["seed", "model", "model_type", "uses_position", "uses_feature",
              "is_object_centric", "clean_skill", "crossing_occlusion_skill",
              "ood_skill", "identity_overall", "identity_no_swap",
              "identity_swap_only", "balanced_identity",
              "no_swap_bias_gap", "no_swap_bias_flag"])

    save_csv(identity_results, "learned_model_identity.csv",
             ["seed", "model", "split_name", "feature_mode",
              "identity_overall", "identity_no_swap", "identity_swap_only",
              "swap_detect_recall", "swap_false_positive_rate",
              "balanced_identity", "sample_count"])

    save_csv(ood_results, "learned_model_ood.csv",
             ["seed", "model", "train_force_type", "test_force_type",
              "clean_skill", "ood_skill", "ood_skill_drop",
              "ood_identity_swap_only", "ood_gated_score"])

    save_csv(gated_results, "learned_model_gated_scores.csv",
             ["seed", "model", "clean_skill", "cf_skill", "comp_skill",
              "identity_overall", "identity_swap_only",
              "gated_score_overall_id", "gated_score_swap_only",
              "score_drop", "no_swap_bias_flag"])

    # =========================================================================
    # Plots
    # =========================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Fingerprint: clean_skill vs identity_swap_only
        fig, ax = plt.subplots(figsize=(8, 6))
        for r in comparison_results:
            cs = float(r["clean_skill"]) if r["clean_skill"] != "nan" else 0.0
            iso = float(r["identity_swap_only"]) if r["identity_swap_only"] != "nan" else 0.0
            mt = r["model_type"]
            color = "blue" if mt == "baseline" else "red"
            marker = "o" if mt == "baseline" else "s"
            ax.scatter(cs, iso, c=color, marker=marker, s=100, zorder=5)
            ax.annotate(r["model"], (cs, iso), fontsize=7, ha="left", va="bottom")
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline")
        ax.axhline(y=0.95, color="blue", linestyle="--", alpha=0.5, label="Feature-aware target")
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.3, label="Skill gate")
        ax.set_xlabel("Clean Skill")
        ax.set_ylabel("Identity Swap-Only")
        ax.set_title("Prediction vs Identity Tradeoff")
        ax.legend(fontsize=7)
        ax.set_xlim(-0.5, 1.0)
        ax.set_ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "learned_fingerprint.png"), dpi=100)
        plt.close()
        print("\n  Saved learned_fingerprint.png")

        # 2. Identity-Prediction tradeoff bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        models_list = [r["model"] for r in comparison_results]
        x = np.arange(len(models_list))
        clean_skills = [float(r["clean_skill"]) if r["clean_skill"] != "nan" else 0.0 for r in comparison_results]
        swap_ids = [float(r["identity_swap_only"]) if r["identity_swap_only"] != "nan" else 0.0 for r in comparison_results]
        width = 0.35
        ax.bar(x - width/2, clean_skills, width, label="Clean Skill", color="steelblue")
        ax.bar(x + width/2, swap_ids, width, label="Identity Swap-Only", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(models_list, rotation=30, ha="right", fontsize=8)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
        ax.set_title("Prediction Skill vs Swap-Only Identity")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "identity_prediction_tradeoff.png"), dpi=100)
        plt.close()
        print("  Saved identity_prediction_tradeoff.png")

        # 3. OOD transfer
        if ood_results:
            fig, ax = plt.subplots(figsize=(8, 5))
            ood_models = [r["model"] for r in ood_results]
            id_skills = [float(r["clean_skill"]) for r in ood_results]
            ood_skills = [float(r["ood_skill"]) for r in ood_results]
            x = np.arange(len(ood_models))
            width = 0.35
            ax.bar(x - width/2, id_skills, width, label="ID Skill")
            ax.bar(x + width/2, ood_skills, width, label="OOD Skill")
            ax.set_xticks(x)
            ax.set_xticklabels(ood_models, rotation=30, ha="right")
            ax.set_title(f"ID vs OOD Skill ({FORCE_TRAIN} -> {FORCE_TEST})")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "ood_transfer_learned.png"), dpi=100)
            plt.close()
            print("  Saved ood_transfer_learned.png")

    except Exception as e:
        print(f"\n  Plot failed: {e}")

    # =========================================================================
    # README
    # =========================================================================
    pytorch_status = "available" if TORCH_AVAILABLE else "unavailable"

    learned_summary = []
    for r in comparison_results:
        if r["model_type"] == "learned":
            learned_summary.append(r)

    best_learned_swap = max([float(r["identity_swap_only"]) for r in learned_summary if r["identity_swap_only"] != "nan"], default=0.0)
    best_learned_skill = max([float(r["clean_skill"]) for r in learned_summary if r["clean_skill"] != "nan"], default=0.0)

    mlp_pos_swap = [float(r["identity_swap_only"]) for r in comparison_results if r["model"] == "MLPPositionOnly" and r["identity_swap_only"] != "nan"]
    mlp_feat_swap = [float(r["identity_swap_only"]) for r in comparison_results if r["model"] == "MLPPositionFeature" and r["identity_swap_only"] != "nan"]
    obj_centric_swap = [float(r["identity_swap_only"]) for r in comparison_results if r["model"] == "ObjectCentricFeatureModel" and r["identity_swap_only"] != "nan"]

    feature_helps = False
    if mlp_pos_swap and mlp_feat_swap:
        feature_helps = mlp_feat_swap[0] > mlp_pos_swap[0] + 0.05

    obj_centric_helps = False
    if mlp_feat_swap and obj_centric_swap:
        obj_centric_helps = obj_centric_swap[0] > mlp_feat_swap[0] + 0.05

    raw_delta_skill = [float(r["clean_skill"]) for r in comparison_results if r["model"] == "RawDeltaKNN" and r["clean_skill"] != "nan"]
    raw_delta_swap = [float(r["identity_swap_only"]) for r in comparison_results if r["model"] == "RawDeltaKNN" and r["identity_swap_only"] != "nan"]

    if not TORCH_AVAILABLE:
        recommendation = "fix_pytorch_dependency"
    elif best_learned_swap < 0.55:
        recommendation = "add_stronger_object_centric_bias"
    elif feature_helps and obj_centric_helps:
        recommendation = "proceed_to_object_file_models"
    elif feature_helps:
        recommendation = "increase_training_budget"
    else:
        recommendation = "fix_feature_training"

    readme = f"""# SVT-v3.1: Minimal Learned Models

## 1. Purpose

v3.1 is a **minimal learned model stress test**, not a model competition. The goal is to test whether three simple learned models can simultaneously achieve:
1. Prediction skill
2. Swap-only identity tracking
3. OOD dynamics transfer
4. Crossing/occlusion robustness

No modifications to the v3 benchmark. No complex architectures.

## 2. Models

| Model | Type | Uses Position | Uses Feature | Object-Centric |
|-------|------|--------------|-------------|----------------|
| LastVelocityBaseline | baseline | yes | no | no |
| RawTrajectoryKNN | baseline | yes | no | no |
| RawDeltaKNN | baseline | yes | no | no |
| TranslationNormalizedKNN | baseline | yes | no | no |
| FeatureAwareIdentityBaseline | baseline | no | yes | no |
| RandomIdentityBaseline | baseline | no | no | no |
| MLPPositionOnly | learned | yes | no | no |
| MLPPositionFeature | learned | yes | yes | no |
| ObjectCentricFeatureModel | learned | yes | yes | yes |

PyTorch status: {pytorch_status}

## 3. Identity Policy

Per v2.4/v3 policy:
- **identity_swap_only** is the primary identity metric
- identity_overall is diagnostic only (inflated by no-swap episodes)
- no-swap accuracy ≠ identity tracking
- All no_swap_bias_flag=True entries must be flagged

## 4. Main Results

### Prediction Skill vs Identity Swap-Only

| Model | Clean Skill | Identity Swap-Only | No-Swap Bias Gap | Bias Flag |
|-------|------------|-------------------|-----------------|-----------|
"""

    for r in comparison_results:
        readme += f"| {r['model']} | {r['clean_skill']} | {r['identity_swap_only']} | {r['no_swap_bias_gap']} | {r['no_swap_bias_flag']} |\n"

    readme += f"""
### OOD Transfer

| Model | ID Skill | OOD Skill | OOD Drop | OOD Identity Swap-Only |
|-------|---------|----------|---------|----------------------|
"""

    for r in ood_results:
        readme += f"| {r['model']} | {r['clean_skill']} | {r['ood_skill']} | {r['ood_skill_drop']} | {r['ood_identity_swap_only']} |\n"

    readme += f"""
## 5. Interpretation

### Position + Feature vs Position Only
- MLPPositionFeature identity_swap_only vs MLPPositionOnly: {"Feature helps" if feature_helps else "Feature does NOT help"}
- {"Feature input improves identity tracking" if feature_helps else "Feature input alone does not improve identity tracking — model may not be utilizing features effectively"}

### Object-Centric vs Plain MLP
- ObjectCentricFeatureModel vs MLPPositionFeature: {"Object-centric helps" if obj_centric_helps else "Object-centric does NOT help"}
- {"Object-centric inductive bias improves identity tracking" if obj_centric_helps else "Object-centric inductive bias does not significantly improve identity tracking at this scale"}

### RawDeltaKNN Prediction/Identity Tradeoff
- RawDeltaKNN clean_skill: {fmt(raw_delta_skill[0]) if raw_delta_skill else 'N/A'}, identity_swap_only: {fmt(raw_delta_swap[0]) if raw_delta_swap else 'N/A'}
- {"Delta-Output Identity Paradox continues: high prediction, low identity" if raw_delta_skill and raw_delta_swap and raw_delta_skill[0] > 0.3 and raw_delta_swap[0] < 0.4 else "Delta-Output Identity Paradox pattern varies"}

### No-Swap Bias in Learned Models
- {"Some learned models show no-swap bias" if any(r['no_swap_bias_flag'] == 'True' for r in comparison_results if r['model_type'] == 'learned') else "Learned models do not show significant no-swap bias"}

### OOD Skill Drop
- {"OOD skill drop is significant — force field change is detectable" if any(float(r['ood_skill_drop']) > 0.05 for r in ood_results) else "OOD skill drop is minimal"}

## 6. Failure Cases
"""

    if not TORCH_AVAILABLE:
        readme += "- Learned models skipped: PyTorch unavailable\n"
    elif best_learned_swap < 0.55:
        readme += "- Minimal learned models did not solve swap-only identity under v3 benchmark.\n"
        readme += "- Best learned identity_swap_only = {:.4f}, barely above random (0.5)\n".format(best_learned_swap)
    else:
        readme += "- Learned models show limited identity tracking capability.\n"

    readme += f"""
## 7. Final Recommendation

**{recommendation}**
"""

    with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SVT-v3.1 FINAL SUMMARY")
    print("=" * 60)
    print(f"PyTorch: {pytorch_status}")
    print(f"Trained models: {list(trained_models.keys())}")
    print(f"Best learned swap_only identity: {fmt(best_learned_swap)}")
    print(f"Best learned clean_skill: {fmt(best_learned_skill)}")
    print(f"Feature helps identity: {feature_helps}")
    print(f"Object-centric helps identity: {obj_centric_helps}")
    print(f"Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
