"""
SVT-v3.2: Swap-Augmented Training Test

Tests whether increasing swap_ratio in training data improves identity_swap_only.
"""

import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v3_2_swap_augmented_training"
SEED = 0
SWAP_RATIOS = [0.0, 0.1, 0.3, 0.5]
FORCE_TRAIN = "attractor"
FORCE_TEST = "vortex"
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
from baselines.knn_retriever_v2 import KNN_V2_REGISTRY
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


def generate_swap_augmented_train(n_train=1000, swap_ratio=0.5, seed=0,
                                   force_type="attractor", feature_mode="feature_bearing",
                                   arena_size=64.0, t_obs=10, t_pred=20, num_objects=2,
                                   field_strength=0.5, damping=0.95, noise_std=0.1):
    rng = np.random.RandomState(seed)
    episodes = []
    for _ in range(n_train):
        ep = _generate_single_episode(
            t_obs=t_obs, t_pred=t_pred,
            num_objects=num_objects, arena_size=arena_size,
            feature_mode=feature_mode,
            randomize_object_order=True,
            identity_test=True,
            swap_probability=swap_ratio,
            force_type=force_type,
            field_strength=field_strength,
            damping=damping,
            noise_std=noise_std,
            rng=rng,
        )
        episodes.append(ep)
    return _stack_episodes(episodes, feature_mode)


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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SVT-v3.2: Swap-Augmented Training Test")
    print("=" * 60)
    print(f"PyTorch: {'available' if TORCH_AVAILABLE else 'UNAVAILABLE'} (device={DEVICE})")
    print(f"Swap ratios: {SWAP_RATIOS}")

    # =========================================================================
    # Step 1: Generate fixed evaluation splits (once)
    # =========================================================================
    print("\n--- Generating evaluation splits ---")

    eval_ds = generate_v3_dataset(
        n_train=1000, n_test=200,
        feature_mode="feature_bearing",
        force_train_type=FORCE_TRAIN,
        force_test_type=FORCE_TEST,
        randomize_object_order=True,
        disjoint_init_split=True,
        seed=SEED,
    )

    for split_name in ["identity_test_swap_only", "identity_test_no_swap_only",
                        "identity_test_mixed", "crossing_occlusion_test", "ood_force_test",
                        "clean_test_id"]:
        n = len(eval_ds[split_name]["observed_positions"])
        n_swap = int(eval_ds[split_name]["is_swap"].sum())
        print(f"  {split_name}: {n} episodes, {n_swap} swap")

    # =========================================================================
    # Step 2: Baseline evaluation (once, no training needed)
    # =========================================================================
    print("\n--- Baseline evaluation ---")

    baseline_results = []
    train_data_noswap = eval_ds["train_id"]

    # RawDeltaKNN
    knn_model = KNN_V2_REGISTRY["RawDeltaKNN"](k=5, weighting="inverse_distance")
    knn_model.fit(train_data_noswap["observed_positions"], train_data_noswap["future_positions"])

    for split_name in ["identity_test_mixed", "identity_test_swap_only"]:
        test_data = eval_ds[split_name]
        pred_fut = knn_model.predict_future(test_data["observed_positions"])
        pred_met = compute_prediction_metrics(pred_fut, test_data["future_positions"])
        pred_id = knn_model.predict_identity(test_data["observed_positions"],
                                              test_future=test_data["future_positions"])
        bd = compute_identity_breakdown(pred_id, test_data["identity_labels"])

        baseline_results.append({
            "swap_ratio": "N/A", "model": "RawDeltaKNN", "split_name": split_name,
            "clean_skill": fmt(pred_met["skill_score"]),
            "identity_swap_only": fmt(bd["identity_swap_only"]),
            "identity_overall": fmt(bd["identity_overall"]),
            "identity_no_swap": fmt(bd["identity_no_swap"]),
            "balanced_identity": fmt(bd["balanced_identity"]),
            "no_swap_bias_gap": fmt(bd["identity_overall"] - bd["identity_swap_only"]
                                     if not np.isnan(bd["identity_swap_only"]) else float("nan")),
        })
        print(f"  RawDeltaKNN {split_name}: swap_only={fmt(bd['identity_swap_only'])}")

    # FeatureAwareBaseline
    for split_name in ["identity_test_mixed", "identity_test_swap_only"]:
        test_data = eval_ds[split_name]
        fab = FeatureAwareIdentityBaseline()
        obs_feat = test_data.get("object_features_obs")
        fut_feat = test_data.get("object_features_fut")
        if obs_feat is not None and fut_feat is not None:
            pred_ids = fab.predict_identity(obs_feat, fut_feat)
            bd = compute_identity_breakdown(pred_ids, test_data["identity_labels"])
        else:
            bd = {"identity_swap_only": float("nan"), "identity_overall": float("nan"),
                  "identity_no_swap": float("nan"), "balanced_identity": float("nan")}

        baseline_results.append({
            "swap_ratio": "N/A", "model": "FeatureAwareBaseline", "split_name": split_name,
            "clean_skill": "nan",
            "identity_swap_only": fmt(bd["identity_swap_only"]),
            "identity_overall": fmt(bd["identity_overall"]),
            "identity_no_swap": fmt(bd["identity_no_swap"]),
            "balanced_identity": fmt(bd["balanced_identity"]),
            "no_swap_bias_gap": "nan",
        })
        print(f"  FAB {split_name}: swap_only={fmt(bd['identity_swap_only'])}")

    # RandomBaseline
    for split_name in ["identity_test_swap_only"]:
        test_data = eval_ds[split_name]
        rand_model = RandomIdentityBaseline(seed=42)
        pred_ids = rand_model.predict_identity(test_data["observed_positions"])
        bd = compute_identity_breakdown(pred_ids, test_data["identity_labels"])
        baseline_results.append({
            "swap_ratio": "N/A", "model": "RandomBaseline", "split_name": split_name,
            "clean_skill": "0.0000",
            "identity_swap_only": fmt(bd["identity_swap_only"]),
            "identity_overall": fmt(bd["identity_overall"]),
            "identity_no_swap": fmt(bd["identity_no_swap"]),
            "balanced_identity": fmt(bd["balanced_identity"]),
            "no_swap_bias_gap": "nan",
        })

    # =========================================================================
    # Step 3: Train and evaluate at each swap_ratio
    # =========================================================================
    comparison_results = []
    identity_results = []
    ood_results = []
    gated_results = []
    training_curves = []
    sample_efficiency = []

    if not TORCH_AVAILABLE:
        print("\n--- PyTorch unavailable, skipping learned models ---")
    else:
        from models.mlp_position_feature import MLPPositionFeature
        from models.object_centric_feature import ObjectCentricFeatureModel
        from utils.torch_training import train_model

        for swap_ratio in SWAP_RATIOS:
            print(f"\n{'='*50}")
            print(f"SWAP RATIO = {swap_ratio}")
            print(f"{'='*50}")

            # Generate training data with this swap ratio
            print(f"  Generating train data (swap_ratio={swap_ratio})...")
            train_data = generate_swap_augmented_train(
                n_train=1000, swap_ratio=swap_ratio, seed=SEED,
                force_type=FORCE_TRAIN, feature_mode="feature_bearing",
            )
            n_swap_train = int(train_data["is_swap"].sum())
            print(f"  Train: {len(train_data['observed_positions'])} episodes, {n_swap_train} swap ({n_swap_train/len(train_data['observed_positions']):.1%})")

            sample_efficiency.append({
                "swap_ratio": swap_ratio,
                "n_train": len(train_data["observed_positions"]),
                "n_swap_episodes": n_swap_train,
                "swap_fraction": fmt(n_swap_train / len(train_data["observed_positions"])),
            })

            for model_name, model_class, uses_feat in [
                ("MLPPositionFeature", MLPPositionFeature, True),
                ("ObjectCentricFeatureModel", ObjectCentricFeatureModel, True),
            ]:
                print(f"\n  Training {model_name} (swap_ratio={swap_ratio})...")

                try:
                    model = model_class(identity_weight=1.0)
                    log = train_model(
                        model, train_data, val_data=eval_ds["clean_test_id"],
                        epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
                        uses_features=uses_feat, verbose=False,
                    )

                    for entry in log:
                        training_curves.append({
                            "swap_ratio": swap_ratio, "model": model_name,
                            "epoch": entry["epoch"],
                            "train_loss": fmt(entry["train_loss"]),
                            "mse_loss": fmt(entry["mse_loss"]),
                            "identity_loss": fmt(entry["identity_loss"]),
                            "val_clean_skill": fmt(entry["val_clean_skill"]),
                        })

                    final_skill = log[-1]["val_clean_skill"]
                    print(f"    Trained: val_skill={fmt(final_skill)}")

                    # Evaluate on all test splits
                    for split_name in ["identity_test_mixed", "identity_test_swap_only",
                                        "identity_test_no_swap_only"]:
                        test_data = eval_ds[split_name]
                        pred_fut, pred_met, pred_id, bd = evaluate_learned_model(
                            model, test_data, uses_features=uses_feat)

                        id_overall = bd["identity_overall"]
                        id_swap = bd["identity_swap_only"]
                        id_noswap = bd["identity_no_swap"]
                        gap = id_overall - id_swap if not np.isnan(id_swap) else float("nan")

                        identity_results.append({
                            "swap_ratio": swap_ratio, "model": model_name,
                            "split_name": split_name,
                            "identity_overall": fmt(id_overall),
                            "identity_no_swap": fmt(id_noswap),
                            "identity_swap_only": fmt(id_swap),
                            "swap_detect_recall": fmt(bd["swap_detect_recall"]),
                            "swap_false_positive_rate": fmt(bd["swap_false_positive_rate"]),
                            "balanced_identity": fmt(bd["balanced_identity"]),
                            "sample_count": len(test_data["observed_positions"]),
                        })

                        if split_name == "identity_test_mixed":
                            comparison_results.append({
                                "swap_ratio": swap_ratio, "model": model_name,
                                "clean_skill": fmt(pred_met["skill_score"]),
                                "identity_overall": fmt(id_overall),
                                "identity_no_swap": fmt(id_noswap),
                                "identity_swap_only": fmt(id_swap),
                                "balanced_identity": fmt(bd["balanced_identity"]),
                                "no_swap_bias_gap": fmt(gap),
                                "no_swap_bias_flag": str(gap > 0.1 if not np.isnan(gap) else False),
                            })

                        print(f"    {split_name}: swap_only={fmt(id_swap)} overall={fmt(id_overall)}")

                    # OOD evaluation
                    ood_test = eval_ds["ood_force_test"]
                    clean_test = eval_ds["clean_test_id"]
                    obs_feat_ood = ood_test.get("object_features_obs") if uses_feat else None
                    obs_feat_id = clean_test.get("object_features_obs") if uses_feat else None

                    id_pred = model.predict_future(clean_test["observed_positions"], obs_feat_id)
                    ood_pred = model.predict_future(ood_test["observed_positions"], obs_feat_ood)
                    if isinstance(id_pred, torch.Tensor):
                        id_pred = id_pred.cpu().numpy()
                    if isinstance(ood_pred, torch.Tensor):
                        ood_pred = ood_pred.cpu().numpy()

                    ood_met = compute_ood_skill(id_pred, clean_test["future_positions"],
                                                 ood_pred, ood_test["future_positions"])

                    ood_pred_ids = model.predict_identity(ood_test["observed_positions"], obs_feat_ood)
                    if isinstance(ood_pred_ids, torch.Tensor):
                        ood_pred_ids = ood_pred_ids.cpu().numpy()
                    ood_bd = compute_identity_breakdown(ood_pred_ids, ood_test["identity_labels"])

                    ood_results.append({
                        "swap_ratio": swap_ratio, "model": model_name,
                        "clean_skill": fmt(ood_met["id_skill"]),
                        "ood_skill": fmt(ood_met["ood_skill"]),
                        "ood_skill_drop": fmt(ood_met["ood_skill_drop"]),
                        "ood_identity_swap_only": fmt(ood_bd["identity_swap_only"]),
                    })
                    print(f"    OOD: id_skill={fmt(ood_met['id_skill'])} ood_skill={fmt(ood_met['ood_skill'])} ood_swap_only={fmt(ood_bd['identity_swap_only'])}")

                    # Crossing/occlusion
                    co_test = eval_ds["crossing_occlusion_test"]
                    obs_feat_co = co_test.get("object_features_obs") if uses_feat else None
                    co_pred = model.predict_future(co_test["observed_positions"], obs_feat_co)
                    if isinstance(co_pred, torch.Tensor):
                        co_pred = co_pred.cpu().numpy()
                    co_skill = compute_crossing_occlusion_skill(
                        co_pred, co_test["future_positions"],
                        co_test["has_crossing"], co_test["has_occlusion"])

                    # Gated score
                    mixed_comp = [r for r in comparison_results
                                  if r["swap_ratio"] == swap_ratio and r["model"] == model_name]
                    mixed_bd = [r for r in identity_results
                                if r["swap_ratio"] == swap_ratio and r["model"] == model_name
                                and r["split_name"] == "identity_test_mixed"][-1]

                    cs = float(mixed_comp[0]["clean_skill"]) if mixed_comp and mixed_comp[0]["clean_skill"] != "nan" else 0.0
                    gated = compute_gated_svt_score_v3(
                        cs,
                        ood_met["ood_skill"],
                        co_skill.get("overall_skill", 0.0),
                        float(mixed_bd["identity_swap_only"]) if mixed_bd["identity_swap_only"] != "nan" else 0.0,
                        float(mixed_bd["identity_overall"]) if mixed_bd["identity_overall"] != "nan" else 0.0,
                    )

                    gated_results.append({
                        "swap_ratio": swap_ratio, "model": model_name,
                        "clean_skill": fmt(cs),
                        "cf_skill": fmt(ood_met["ood_skill"]),
                        "comp_skill": fmt(co_skill.get("overall_skill", float("nan"))),
                        "identity_overall": mixed_bd["identity_overall"],
                        "identity_swap_only": mixed_bd["identity_swap_only"],
                        "gated_score_overall_id": fmt(gated["gated_score_overall_id"]),
                        "gated_score_swap_only": fmt(gated["gated_score_swap_only"]),
                        "score_drop": fmt(gated["score_drop"]),
                        "no_swap_bias_flag": str(gated["no_swap_bias_flag"]),
                    })

                except Exception as e:
                    print(f"    {model_name} FAILED at swap_ratio={swap_ratio}: {e}")
                    import traceback
                    traceback.print_exc()

    # =========================================================================
    # Save all CSVs
    # =========================================================================
    # Add baselines to comparison
    for br in baseline_results:
        if br["split_name"] == "identity_test_mixed":
            comparison_results.append({
                "swap_ratio": br["swap_ratio"], "model": br["model"],
                "clean_skill": br["clean_skill"],
                "identity_overall": br["identity_overall"],
                "identity_no_swap": br["identity_no_swap"],
                "identity_swap_only": br["identity_swap_only"],
                "balanced_identity": br["balanced_identity"],
                "no_swap_bias_gap": br["no_swap_bias_gap"],
                "no_swap_bias_flag": str(float(br["no_swap_bias_gap"]) > 0.1 if br["no_swap_bias_gap"] != "nan" else False),
            })

    save_csv(comparison_results, "swap_ratio_comparison.csv",
             ["swap_ratio", "model", "clean_skill", "identity_overall",
              "identity_no_swap", "identity_swap_only", "balanced_identity",
              "no_swap_bias_gap", "no_swap_bias_flag"])

    save_csv(identity_results, "swap_ratio_identity.csv",
             ["swap_ratio", "model", "split_name",
              "identity_overall", "identity_no_swap", "identity_swap_only",
              "swap_detect_recall", "swap_false_positive_rate",
              "balanced_identity", "sample_count"])

    save_csv(ood_results, "swap_ratio_ood.csv",
             ["swap_ratio", "model", "clean_skill", "ood_skill",
              "ood_skill_drop", "ood_identity_swap_only"])

    save_csv(gated_results, "swap_ratio_gated_scores.csv",
             ["swap_ratio", "model", "clean_skill", "cf_skill", "comp_skill",
              "identity_overall", "identity_swap_only",
              "gated_score_overall_id", "gated_score_swap_only",
              "score_drop", "no_swap_bias_flag"])

    save_csv(training_curves, "swap_training_curves.csv",
             ["swap_ratio", "model", "epoch", "train_loss",
              "mse_loss", "identity_loss", "val_clean_skill"])

    save_csv(sample_efficiency, "sample_efficiency.csv",
             ["swap_ratio", "n_train", "n_swap_episodes", "swap_fraction"])

    # =========================================================================
    # Plots
    # =========================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Swap ratio vs identity_swap_only
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_name in ["MLPPositionFeature", "ObjectCentricFeatureModel"]:
            swap_only_rows = [r for r in identity_results
                              if r["model"] == model_name and r["split_name"] == "identity_test_swap_only"]
            if swap_only_rows:
                ratios = [float(r["swap_ratio"]) for r in swap_only_rows]
                vals = [float(r["identity_swap_only"]) for r in swap_only_rows]
                ax.plot(ratios, vals, 'o-', label=model_name)

        # Baseline references
        fab_swap = [r for r in baseline_results if r["model"] == "FeatureAwareBaseline" and r["split_name"] == "identity_test_swap_only"]
        if fab_swap:
            ax.axhline(y=float(fab_swap[0]["identity_swap_only"]), color="blue", linestyle="--",
                        alpha=0.5, label="FeatureAwareBaseline")
        rand_swap = [r for r in baseline_results if r["model"] == "RandomBaseline"]
        if rand_swap:
            ax.axhline(y=float(rand_swap[0]["identity_swap_only"]), color="gray", linestyle="--",
                        alpha=0.5, label="RandomBaseline")
        ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.3)
        ax.set_xlabel("Training Swap Ratio")
        ax.set_ylabel("Identity Swap-Only")
        ax.set_title("Swap Ratio vs Identity Swap-Only")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "swap_ratio_identity_plot.png"), dpi=100)
        plt.close()
        print("\n  Saved swap_ratio_identity_plot.png")

        # 2. Prediction-Identity tradeoff
        fig, ax = plt.subplots(figsize=(8, 6))
        for model_name in ["MLPPositionFeature", "ObjectCentricFeatureModel"]:
            mixed_rows = [r for r in comparison_results if r["model"] == model_name]
            if mixed_rows:
                skills = [float(r["clean_skill"]) for r in mixed_rows if r["clean_skill"] != "nan"]
                swap_ids = [float(r["identity_swap_only"]) for r in mixed_rows if r["identity_swap_only"] != "nan"]
                ratios = [float(r["swap_ratio"]) for r in mixed_rows if r["swap_ratio"] != "N/A"]
                if len(skills) == len(swap_ids) == len(ratios):
                    ax.scatter(skills, swap_ids, s=80, zorder=5)
                    for s, si, r in zip(skills, swap_ids, ratios):
                        ax.annotate(f"{model_name[:8]}\nr={r}", (s, si), fontsize=6, ha="left")

        # KNN reference
        knn_mixed = [r for r in baseline_results if r["model"] == "RawDeltaKNN" and r["split_name"] == "identity_test_mixed"]
        if knn_mixed:
            ax.scatter([0.7], [float(knn_mixed[0]["identity_swap_only"])], marker="x", c="red", s=100, zorder=5, label="RawDeltaKNN")

        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3)
        ax.set_xlabel("Clean Skill")
        ax.set_ylabel("Identity Swap-Only")
        ax.set_title("Prediction vs Identity Tradeoff by Swap Ratio")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "prediction_identity_tradeoff.png"), dpi=100)
        plt.close()
        print("  Saved prediction_identity_tradeoff.png")

    except Exception as e:
        print(f"\n  Plot failed: {e}")

    # =========================================================================
    # README
    # =========================================================================
    # Determine recommendation
    learned_swap_only = [float(r["identity_swap_only"]) for r in identity_results
                          if r["split_name"] == "identity_test_swap_only"
                          and r["model"] in ["MLPPositionFeature", "ObjectCentricFeatureModel"]
                          and r["identity_swap_only"] != "nan"]

    best_learned_swap = max(learned_swap_only) if learned_swap_only else 0.0

    # Check if swap_ratio increase improves identity
    obj_centric_at_05 = [float(r["identity_swap_only"]) for r in identity_results
                          if r["model"] == "ObjectCentricFeatureModel"
                          and r["split_name"] == "identity_test_swap_only"
                          and str(r["swap_ratio"]) == "0.5"
                          and r["identity_swap_only"] != "nan"]
    obj_centric_at_00 = [float(r["identity_swap_only"]) for r in identity_results
                          if r["model"] == "ObjectCentricFeatureModel"
                          and r["split_name"] == "identity_test_swap_only"
                          and str(r["swap_ratio"]) == "0.0"
                          and r["identity_swap_only"] != "nan"]

    swap_helps = False
    if obj_centric_at_05 and obj_centric_at_00:
        swap_helps = obj_centric_at_05[0] > obj_centric_at_00[0] + 0.05

    obj_better_than_mlp = False
    obj_at_05 = [float(r["identity_swap_only"]) for r in identity_results
                  if r["model"] == "ObjectCentricFeatureModel"
                  and r["split_name"] == "identity_test_swap_only"
                  and str(r["swap_ratio"]) == "0.5"
                  and r["identity_swap_only"] != "nan"]
    mlp_at_05 = [float(r["identity_swap_only"]) for r in identity_results
                  if r["model"] == "MLPPositionFeature"
                  and r["split_name"] == "identity_test_swap_only"
                  and str(r["swap_ratio"]) == "0.5"
                  and r["identity_swap_only"] != "nan"]
    if obj_at_05 and mlp_at_05:
        obj_better_than_mlp = obj_at_05[0] > mlp_at_05[0] + 0.05

    if not TORCH_AVAILABLE:
        recommendation = "fix_pytorch_dependency"
    elif best_learned_swap < 0.55:
        recommendation = "add_contrastive_identity_loss"
    elif swap_helps and obj_better_than_mlp:
        recommendation = "proceed_to_object_file_models"
    elif swap_helps:
        recommendation = "increase_training_budget"
    elif best_learned_swap >= 0.55:
        recommendation = "fix_feature_training"
    else:
        recommendation = "benchmark_too_hard"

    readme = f"""# SVT-v3.2: Swap-Augmented Training Test

## 1. Purpose

v3.2 tests whether adding swap episodes to training data improves identity_swap_only in learned models.

v3.1 found that all learned models collapse to "always no-swap" (identity_swap_only=0.0) when trained without swap data. This experiment asks: **does swap-augmented training fix the identity head?**

## 2. Method

- Training swap ratios: {SWAP_RATIOS}
- Evaluation splits: fixed v3 benchmark (generated once, same for all ratios)
- Models: MLPPositionFeature, ObjectCentricFeatureModel
- Baselines: RawDeltaKNN, FeatureAwareBaseline, RandomBaseline
- Primary metric: identity_swap_only

## 3. Main Results

### Swap Ratio vs Identity Swap-Only

| Swap Ratio | Model | Clean Skill | Swap-Only ID | Overall ID | No-Swap ID | Bias Gap |
|------------|-------|------------|-------------|-----------|-----------|---------|
"""

    for r in sorted(comparison_results, key=lambda x: (str(x["swap_ratio"]), x["model"])):
        readme += f"| {r['swap_ratio']} | {r['model']} | {r['clean_skill']} | {r['identity_swap_only']} | {r['identity_overall']} | {r['identity_no_swap']} | {r['no_swap_bias_gap']} |\n"

    readme += f"""
### Baseline References

| Model | Swap-Only ID |
|-------|-------------|
| FeatureAwareBaseline | {fab_swap[0]['identity_swap_only'] if fab_swap else 'N/A'} |
| RawDeltaKNN | {knn_mixed[0]['identity_swap_only'] if knn_mixed else 'N/A'} |
| RandomBaseline | {rand_swap[0]['identity_swap_only'] if rand_swap else 'N/A'} |

## 4. Key Questions

### Does swap_ratio increase improve identity_swap_only?
{"Yes — swap-augmented training improves identity tracking" if swap_helps else "No — swap-augmented training does NOT significantly improve identity tracking"}

### Is ObjectCentricFeature better than MLPPositionFeature?
{"Yes — object-centric bias helps identity tracking" if obj_better_than_mlp else "No — object-centric bias does NOT significantly help at this scale"}

### Is prediction skill separated from identity?
See prediction_identity_tradeoff.png for the tradeoff landscape.

### If swap_ratio=0.5 still fails, what's wrong?
If identity_swap_only remains near 0.5 at swap_ratio=0.5, the issue is likely:
- Identity head architecture (single swap logit may be insufficient)
- Feature not being utilized effectively by the model
- Need for contrastive or self-supervised identity loss

## 5. Best Learned Identity Swap-Only

**{fmt(best_learned_swap)}**

{"This is above random (0.5) — swap-augmented training partially works" if best_learned_swap > 0.55 else "This is at or below random — swap-augmented training alone is insufficient"}

## 6. Failure Cases
"""

    if not TORCH_AVAILABLE:
        readme += "- Learned models skipped: PyTorch unavailable\n"
    elif best_learned_swap < 0.55:
        readme += "- Swap-augmented training did NOT solve identity tracking\n"
        readme += f"- Best identity_swap_only = {fmt(best_learned_swap)}, at or below random\n"
    else:
        readme += "- Swap-augmented training partially improved identity tracking\n"

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
    print("SVT-v3.2 FINAL SUMMARY")
    print("=" * 60)
    print(f"Best learned identity_swap_only: {fmt(best_learned_swap)}")
    print(f"Swap ratio helps identity: {swap_helps}")
    print(f"Object-centric better than MLP: {obj_better_than_mlp}")
    print(f"Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
