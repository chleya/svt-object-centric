"""
SVT-v3.2.1: Stability Audit

Multi-seed verification of v3.2 results + feature ablation.
"""

import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v3_2_1_stability_audit"
SEEDS = [0, 1, 2]
SWAP_RATIOS = [0.0, 0.1, 0.3, 0.5]
EPOCHS = 20
FORCE_TRAIN = "attractor"
FORCE_TEST = "vortex"

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
from metrics.ood_metrics import compute_ood_skill


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


def shuffle_features(features_obs, seed=42):
    if features_obs is None:
        return None
    rng = np.random.RandomState(seed)
    shuffled = features_obs.copy()
    B, T, N, F = shuffled.shape
    for b in range(B):
        for t in range(T):
            perm = rng.permutation(N)
            shuffled[b, t] = shuffled[b, t, perm]
    return shuffled


def zero_features(features_obs):
    if features_obs is None:
        return None
    return np.zeros_like(features_obs)


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
    return pred_metrics, breakdown


def evaluate_ood(model, eval_ds, uses_features):
    clean_test = eval_ds["clean_test_id"]
    ood_test = eval_ds["ood_force_test"]
    obs_feat_id = clean_test.get("object_features_obs") if uses_features else None
    obs_feat_ood = ood_test.get("object_features_obs") if uses_features else None

    id_pred = model.predict_future(clean_test["observed_positions"], obs_feat_id)
    ood_pred = model.predict_future(ood_test["observed_positions"], obs_feat_ood)
    if isinstance(id_pred, torch.Tensor):
        id_pred = id_pred.cpu().numpy()
    if isinstance(ood_pred, torch.Tensor):
        ood_pred = ood_pred.cpu().numpy()

    ood_met = compute_ood_skill(id_pred, clean_test["future_positions"],
                                 ood_pred, ood_test["future_positions"])
    return ood_met


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SVT-v3.2.1: Stability Audit")
    print("=" * 60)
    print(f"PyTorch: {'available' if TORCH_AVAILABLE else 'UNAVAILABLE'} (device={DEVICE})")

    if not TORCH_AVAILABLE:
        print("PyTorch unavailable. Exiting.")
        return

    from models.mlp_position_feature import MLPPositionFeature
    from models.object_centric_feature import ObjectCentricFeatureModel
    from utils.torch_training import train_model

    # =========================================================================
    # Part 1: Multi-seed stability
    # =========================================================================
    print("\n=== Part 1: Multi-Seed Stability ===")

    multi_seed_results = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        eval_ds = generate_v3_dataset(
            n_train=1000, n_test=200,
            feature_mode="feature_bearing",
            force_train_type=FORCE_TRAIN,
            force_test_type=FORCE_TEST,
            randomize_object_order=True,
            disjoint_init_split=True,
            seed=seed,
        )

        for swap_ratio in SWAP_RATIOS:
            print(f"  swap_ratio={swap_ratio}")

            train_data = generate_swap_augmented_train(
                n_train=1000, swap_ratio=swap_ratio, seed=seed,
                force_type=FORCE_TRAIN,
            )

            for model_name, model_class in [
                ("MLPPositionFeature", MLPPositionFeature),
                ("ObjectCentricFeatureModel", ObjectCentricFeatureModel),
            ]:
                try:
                    model = model_class(identity_weight=1.0)
                    log = train_model(
                        model, train_data, val_data=eval_ds["clean_test_id"],
                        epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
                        uses_features=True, verbose=False,
                    )

                    # Evaluate on swap_only split
                    swap_test = eval_ds["identity_test_swap_only"]
                    pred_met, bd = evaluate_learned_model(model, swap_test, uses_features=True)

                    # Evaluate on mixed split for bias
                    mixed_test = eval_ds["identity_test_mixed"]
                    _, bd_mixed = evaluate_learned_model(model, mixed_test, uses_features=True)

                    # OOD
                    ood_met = evaluate_ood(model, eval_ds, uses_features=True)

                    gap = bd_mixed["identity_overall"] - bd_mixed["identity_swap_only"] \
                        if not np.isnan(bd_mixed["identity_swap_only"]) else float("nan")

                    multi_seed_results.append({
                        "seed": seed, "swap_ratio": swap_ratio, "model": model_name,
                        "clean_skill": fmt(pred_met["skill_score"]),
                        "identity_swap_only": fmt(bd["identity_swap_only"]),
                        "identity_overall_mixed": fmt(bd_mixed["identity_overall"]),
                        "no_swap_bias_gap": fmt(gap),
                        "ood_skill": fmt(ood_met["ood_skill"]),
                    })

                    print(f"    {model_name}: swap_only_id={fmt(bd['identity_swap_only'])} clean_skill={fmt(pred_met['skill_score'])}")

                except Exception as e:
                    print(f"    {model_name} FAILED: {e}")
                    multi_seed_results.append({
                        "seed": seed, "swap_ratio": swap_ratio, "model": model_name,
                        "clean_skill": "nan", "identity_swap_only": "nan",
                        "identity_overall_mixed": "nan", "no_swap_bias_gap": "nan",
                        "ood_skill": "nan",
                    })

    save_csv(multi_seed_results, "multi_seed_summary.csv",
             ["seed", "swap_ratio", "model", "clean_skill", "identity_swap_only",
              "identity_overall_mixed", "no_swap_bias_gap", "ood_skill"])

    # =========================================================================
    # Part 2: Feature Ablation
    # =========================================================================
    print("\n=== Part 2: Feature Ablation ===")

    ablation_results = []
    seed = 0
    swap_ratio = 0.3

    eval_ds = generate_v3_dataset(
        n_train=1000, n_test=200,
        feature_mode="feature_bearing",
        force_train_type=FORCE_TRAIN,
        force_test_type=FORCE_TEST,
        randomize_object_order=True,
        disjoint_init_split=True,
        seed=seed,
    )

    train_data = generate_swap_augmented_train(
        n_train=1000, swap_ratio=swap_ratio, seed=seed,
        force_type=FORCE_TRAIN,
    )

    for feat_mode in ["normal", "shuffled", "zero"]:
        print(f"\n  Feature mode: {feat_mode}")

        if feat_mode == "shuffled":
            train_feat = shuffle_features(train_data["object_features_obs"], seed=seed)
            train_data_abl = dict(train_data)
            train_data_abl["object_features_obs"] = train_feat
        elif feat_mode == "zero":
            train_feat = zero_features(train_data["object_features_obs"])
            train_data_abl = dict(train_data)
            train_data_abl["object_features_obs"] = train_feat
        else:
            train_data_abl = train_data

        for model_name, model_class in [
            ("MLPPositionFeature", MLPPositionFeature),
            ("ObjectCentricFeatureModel", ObjectCentricFeatureModel),
        ]:
            try:
                model = model_class(identity_weight=1.0)
                log = train_model(
                    model, train_data_abl, val_data=eval_ds["clean_test_id"],
                    epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
                    uses_features=True, verbose=False,
                )

                # Evaluate with NORMAL features at test time
                swap_test = eval_ds["identity_test_swap_only"]
                pred_met, bd = evaluate_learned_model(model, swap_test, uses_features=True)

                mixed_test = eval_ds["identity_test_mixed"]
                _, bd_mixed = evaluate_learned_model(model, mixed_test, uses_features=True)

                ood_met = evaluate_ood(model, eval_ds, uses_features=True)

                ablation_results.append({
                    "feature_mode": feat_mode, "model": model_name,
                    "swap_ratio": swap_ratio,
                    "clean_skill": fmt(pred_met["skill_score"]),
                    "identity_swap_only": fmt(bd["identity_swap_only"]),
                    "identity_overall_mixed": fmt(bd_mixed["identity_overall"]),
                    "no_swap_bias_gap": fmt(bd_mixed["identity_overall"] - bd_mixed["identity_swap_only"]
                                             if not np.isnan(bd_mixed["identity_swap_only"]) else float("nan")),
                    "ood_skill": fmt(ood_met["ood_skill"]),
                })

                print(f"    {model_name}: swap_only_id={fmt(bd['identity_swap_only'])}")

            except Exception as e:
                print(f"    {model_name} FAILED: {e}")
                ablation_results.append({
                    "feature_mode": feat_mode, "model": model_name,
                    "swap_ratio": swap_ratio,
                    "clean_skill": "nan", "identity_swap_only": "nan",
                    "identity_overall_mixed": "nan", "no_swap_bias_gap": "nan",
                    "ood_skill": "nan",
                })

    save_csv(ablation_results, "feature_ablation.csv",
             ["feature_mode", "model", "swap_ratio", "clean_skill",
              "identity_swap_only", "identity_overall_mixed",
              "no_swap_bias_gap", "ood_skill"])

    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n=== Analysis ===")

    # Q1: Is ObjectCentricFeature stably better than MLPPositionFeature?
    obj_swap_ids = [float(r["identity_swap_only"]) for r in multi_seed_results
                     if r["model"] == "ObjectCentricFeatureModel"
                     and r["identity_swap_only"] != "nan"
                     and r["swap_ratio"] in ["0.1", "0.3", "0.5"]]
    mlp_swap_ids = [float(r["identity_swap_only"]) for r in multi_seed_results
                     if r["model"] == "MLPPositionFeature"
                     and r["identity_swap_only"] != "nan"
                     and r["swap_ratio"] in ["0.1", "0.3", "0.5"]]

    obj_mean = np.mean(obj_swap_ids) if obj_swap_ids else 0.0
    mlp_mean = np.mean(mlp_swap_ids) if mlp_swap_ids else 0.0
    obj_stably_better = obj_mean > mlp_mean + 0.1

    print(f"  ObjectCentric mean swap_only: {fmt(obj_mean)}")
    print(f"  MLPPosition mean swap_only: {fmt(mlp_mean)}")
    print(f"  ObjectCentric stably better: {obj_stably_better}")

    # Q2: Is swap_ratio=0.3 still an effective point?
    sr03_obj = [float(r["identity_swap_only"]) for r in multi_seed_results
                 if r["model"] == "ObjectCentricFeatureModel"
                 and str(r["swap_ratio"]) == "0.3"
                 and r["identity_swap_only"] != "nan"]
    sr03_mlp = [float(r["identity_swap_only"]) for r in multi_seed_results
                 if r["model"] == "MLPPositionFeature"
                 and str(r["swap_ratio"]) == "0.3"
                 and r["identity_swap_only"] != "nan"]

    sr03_obj_mean = np.mean(sr03_obj) if sr03_obj else 0.0
    sr03_effective = sr03_obj_mean > 0.6

    print(f"  swap_ratio=0.3 ObjectCentric mean: {fmt(sr03_obj_mean)}")
    print(f"  swap_ratio=0.3 effective: {sr03_effective}")

    # Q3: Is normal feature better than shuffled/zero?
    normal_swap = [float(r["identity_swap_only"]) for r in ablation_results
                    if r["feature_mode"] == "normal" and r["identity_swap_only"] != "nan"]
    shuffled_swap = [float(r["identity_swap_only"]) for r in ablation_results
                      if r["feature_mode"] == "shuffled" and r["identity_swap_only"] != "nan"]
    zero_swap = [float(r["identity_swap_only"]) for r in ablation_results
                  if r["feature_mode"] == "zero" and r["identity_swap_only"] != "nan"]

    normal_mean = np.mean(normal_swap) if normal_swap else 0.0
    shuffled_mean = np.mean(shuffled_swap) if shuffled_swap else 0.0
    zero_mean = np.mean(zero_swap) if zero_swap else 0.0
    feature_matters = normal_mean > max(shuffled_mean, zero_mean) + 0.05

    print(f"  Normal feature mean: {fmt(normal_mean)}")
    print(f"  Shuffled feature mean: {fmt(shuffled_mean)}")
    print(f"  Zero feature mean: {fmt(zero_mean)}")
    print(f"  Feature matters: {feature_matters}")

    # =========================================================================
    # Plots
    # =========================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Swap ratio vs identity_swap_only (multi-seed)
        ax = axes[0]
        for model_name, color, marker in [
            ("MLPPositionFeature", "coral", "o"),
            ("ObjectCentricFeatureModel", "steelblue", "s"),
        ]:
            for seed in SEEDS:
                rows = [r for r in multi_seed_results
                        if r["model"] == model_name and r["seed"] == seed
                        and r["identity_swap_only"] != "nan"]
                if rows:
                    ratios = [float(r["swap_ratio"]) for r in rows]
                    vals = [float(r["identity_swap_only"]) for r in rows]
                    label = model_name if seed == 0 else None
                    ax.plot(ratios, vals, marker, color=color, alpha=0.6, label=label)

        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3)
        ax.set_xlabel("Swap Ratio")
        ax.set_ylabel("Identity Swap-Only")
        ax.set_title("Multi-Seed Stability")
        ax.legend(fontsize=7)
        ax.set_ylim(-0.05, 1.05)

        # Panel 2: Feature ablation bar chart
        ax = axes[1]
        models_abl = ["MLPPositionFeature", "ObjectCentricFeatureModel"]
        x = np.arange(len(models_abl))
        width = 0.25
        for i, (fm, color) in enumerate([("normal", "steelblue"), ("shuffled", "orange"), ("zero", "gray")]):
            vals = []
            for m in models_abl:
                found = [r for r in ablation_results if r["feature_mode"] == fm and r["model"] == m]
                vals.append(float(found[0]["identity_swap_only"]) if found and found[0]["identity_swap_only"] != "nan" else 0.0)
            ax.bar(x + i * width, vals, width, label=fm, color=color)
        ax.set_xticks(x + width)
        ax.set_xticklabels(["MLP+Feat", "ObjCentric"], fontsize=9)
        ax.set_ylabel("Identity Swap-Only")
        ax.set_title("Feature Ablation (swap_ratio=0.3)")
        ax.legend(fontsize=7)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3)

        # Panel 3: Clean skill vs identity swap_only
        ax = axes[2]
        for model_name, color, marker in [
            ("MLPPositionFeature", "coral", "o"),
            ("ObjectCentricFeatureModel", "steelblue", "s"),
        ]:
            rows = [r for r in multi_seed_results
                    if r["model"] == model_name
                    and r["identity_swap_only"] != "nan"
                    and r["clean_skill"] != "nan"]
            if rows:
                skills = [float(r["clean_skill"]) for r in rows]
                swap_ids = [float(r["identity_swap_only"]) for r in rows]
                ax.scatter(skills, swap_ids, c=color, marker=marker, alpha=0.6, label=model_name)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3)
        ax.set_xlabel("Clean Skill")
        ax.set_ylabel("Identity Swap-Only")
        ax.set_title("Prediction vs Identity")
        ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "stability_plot.png"), dpi=100)
        plt.close()
        print("\n  Saved stability_plot.png")

    except Exception as e:
        print(f"\n  Plot failed: {e}")

    # =========================================================================
    # README
    # =========================================================================
    if obj_stably_better and sr03_effective and feature_matters:
        recommendation = "proceed_to_object_file_models"
    elif obj_stably_better and sr03_effective:
        recommendation = "proceed_to_object_file_models"
    elif not obj_stably_better:
        recommendation = "increase_training_budget"
    else:
        recommendation = "fix_feature_training"

    readme = f"""# SVT-v3.2.1: Stability Audit

## 1. Is ObjectCentricFeature stably better than MLPPositionFeature?

**{"Yes" if obj_stably_better else "No"}**

- ObjectCentricFeature mean identity_swap_only (swap_ratio>0): {fmt(obj_mean)}
- MLPPositionFeature mean identity_swap_only (swap_ratio>0): {fmt(mlp_mean)}
- Difference: {fmt(obj_mean - mlp_mean)}

{"ObjectCentricFeature consistently outperforms MLPPositionFeature across seeds and swap ratios." if obj_stably_better else "ObjectCentricFeature does NOT consistently outperform MLPPositionFeature. The advantage seen in v3.2 may be seed-dependent."}

## 2. Is swap_ratio=0.3 still an effective point?

**{"Yes" if sr03_effective else "No"}**

- ObjectCentricFeature at swap_ratio=0.3: mean identity_swap_only = {fmt(sr03_obj_mean)}
- {"swap_ratio=0.3 provides sufficient swap signal for ObjectCentricFeature to learn identity tracking." if sr03_effective else "swap_ratio=0.3 does NOT provide sufficient improvement."}

## 3. Is normal feature better than shuffled/zero feature?

**{"Yes" if feature_matters else "No"}**

- Normal feature mean: {fmt(normal_mean)}
- Shuffled feature mean: {fmt(shuffled_mean)}
- Zero feature mean: {fmt(zero_mean)}

{"Normal features significantly outperform shuffled/zero features, confirming the model is actually using feature information for identity tracking." if feature_matters else "Normal features do NOT significantly outperform shuffled/zero features. The model may not be effectively using feature information."}

## Final Recommendation

**{recommendation}**
"""

    with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print("\n" + "=" * 60)
    print("SVT-v3.2.1 FINAL SUMMARY")
    print("=" * 60)
    print(f"ObjectCentric stably better: {obj_stably_better}")
    print(f"swap_ratio=0.3 effective: {sr03_effective}")
    print(f"Feature matters: {feature_matters}")
    print(f"Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
