"""
SVT-v3.3: Feature-Binding Sanity Test

Tests whether identity prediction truly depends on features.
Uses N×N assignment head instead of binary swap prediction.
"""

import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v3_3_feature_binding"
SEED = 0
SWAP_RATIOS = [0.0, 0.1, 0.3, 0.5]
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


def random_wrong_features(features_obs, seed=42):
    if features_obs is None:
        return None
    rng = np.random.RandomState(seed)
    B, T, N, F = features_obs.shape
    wrong = rng.rand(B, T, N, F)
    wrong = (wrong > 0.5).astype(np.float32)
    row_sums = wrong.sum(axis=-1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    wrong = wrong / row_sums
    return wrong


def compute_assignment_accuracy(pred_identity, true_identity):
    correct = (pred_identity == true_identity).all(axis=1)
    return float(correct.mean())


def evaluate_model(model, test_data, uses_features=True, feat_override=None):
    obs_pos = test_data["observed_positions"]
    obs_feat = feat_override if feat_override is not None else (
        test_data.get("object_features_obs") if uses_features else None)

    pred_future = model.predict_future(obs_pos, obs_feat)
    if isinstance(pred_future, torch.Tensor):
        pred_future = pred_future.cpu().numpy()

    pred_metrics = compute_prediction_metrics(pred_future, test_data["future_positions"])

    pred_identity = model.predict_identity(obs_pos, obs_feat)
    if isinstance(pred_identity, torch.Tensor):
        pred_identity = pred_identity.cpu().numpy()

    true_identity = test_data["identity_labels"]

    assignment_acc = compute_assignment_accuracy(pred_identity, true_identity)
    breakdown = compute_identity_breakdown(pred_identity, true_identity)

    return pred_metrics, breakdown, assignment_acc


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SVT-v3.3: Feature-Binding Sanity Test")
    print("=" * 60)
    print(f"PyTorch: {'available' if TORCH_AVAILABLE else 'UNAVAILABLE'} (device={DEVICE})")

    if not TORCH_AVAILABLE:
        print("PyTorch unavailable. Exiting.")
        return

    from models.mlp_position_feature_assignment import MLPPositionFeatureAssignment
    from models.object_centric_feature_assignment import ObjectCentricFeatureAssignment
    from utils.torch_training import train_model

    # =========================================================================
    # Part 1: Assignment Results across swap ratios
    # =========================================================================
    print("\n=== Part 1: Assignment Results ===")

    assignment_results = []

    eval_ds = generate_v3_dataset(
        n_train=1000, n_test=200,
        feature_mode="feature_bearing",
        force_train_type="attractor",
        force_test_type="vortex",
        randomize_object_order=True,
        disjoint_init_split=True,
        seed=SEED,
    )

    for swap_ratio in SWAP_RATIOS:
        print(f"\n  swap_ratio={swap_ratio}")

        train_data = generate_swap_augmented_train(
            n_train=1000, swap_ratio=swap_ratio, seed=SEED,
        )

        for model_name, model_class in [
            ("MLPPositionFeatureAssignment", MLPPositionFeatureAssignment),
            ("ObjectCentricFeatureAssignment", ObjectCentricFeatureAssignment),
        ]:
            try:
                model = model_class(identity_weight=1.0)
                log = train_model(
                    model, train_data, val_data=eval_ds["clean_test_id"],
                    epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
                    uses_features=True, verbose=False,
                )

                # Evaluate on swap_only
                swap_test = eval_ds["identity_test_swap_only"]
                pred_met, bd, asgn_acc = evaluate_model(model, swap_test)

                # Evaluate on mixed
                mixed_test = eval_ds["identity_test_mixed"]
                _, bd_mixed, asgn_acc_mixed = evaluate_model(model, mixed_test)

                gap = bd_mixed["identity_overall"] - bd_mixed["identity_swap_only"] \
                    if not np.isnan(bd_mixed["identity_swap_only"]) else float("nan")

                assignment_results.append({
                    "swap_ratio": swap_ratio, "model": model_name,
                    "clean_skill": fmt(pred_met["skill_score"]),
                    "assignment_accuracy_swap_only": fmt(asgn_acc),
                    "identity_swap_only": fmt(bd["identity_swap_only"]),
                    "identity_overall_mixed": fmt(bd_mixed["identity_overall"]),
                    "identity_no_swap_mixed": fmt(bd_mixed["identity_no_swap"]),
                    "no_swap_bias_gap": fmt(gap),
                    "balanced_identity": fmt(bd_mixed["balanced_identity"]),
                })

                print(f"    {model_name}: asgn_acc={fmt(asgn_acc)} swap_only_id={fmt(bd['identity_swap_only'])}")

            except Exception as e:
                print(f"    {model_name} FAILED: {e}")
                import traceback
                traceback.print_exc()

    save_csv(assignment_results, "assignment_results.csv",
             ["swap_ratio", "model", "clean_skill", "assignment_accuracy_swap_only",
              "identity_swap_only", "identity_overall_mixed", "identity_no_swap_mixed",
              "no_swap_bias_gap", "balanced_identity"])

    # =========================================================================
    # Part 2: Feature Ablation (on swap_ratio=0.3 models)
    # =========================================================================
    print("\n=== Part 2: Feature Ablation ===")

    ablation_results = []
    swap_ratio = 0.3

    train_data = generate_swap_augmented_train(
        n_train=1000, swap_ratio=swap_ratio, seed=SEED,
    )

    for model_name, model_class in [
        ("MLPPositionFeatureAssignment", MLPPositionFeatureAssignment),
        ("ObjectCentricFeatureAssignment", ObjectCentricFeatureAssignment),
    ]:
        print(f"\n  Training {model_name} (swap_ratio={swap_ratio})...")

        try:
            model = model_class(identity_weight=1.0)
            log = train_model(
                model, train_data, val_data=eval_ds["clean_test_id"],
                epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=True, verbose=False,
            )

            swap_test = eval_ds["identity_test_swap_only"]

            for feat_mode in ["normal", "shuffled", "zero", "random_wrong"]:
                if feat_mode == "normal":
                    feat_override = None
                elif feat_mode == "shuffled":
                    feat_override = shuffle_features(
                        swap_test["object_features_obs"], seed=SEED)
                elif feat_mode == "zero":
                    feat_override = zero_features(swap_test["object_features_obs"])
                elif feat_mode == "random_wrong":
                    feat_override = random_wrong_features(
                        swap_test["object_features_obs"], seed=SEED)
                else:
                    feat_override = None

                pred_met, bd, asgn_acc = evaluate_model(
                    model, swap_test, uses_features=True, feat_override=feat_override)

                ablation_results.append({
                    "model": model_name, "feature_mode": feat_mode,
                    "swap_ratio": swap_ratio,
                    "assignment_accuracy": fmt(asgn_acc),
                    "identity_swap_only": fmt(bd["identity_swap_only"]),
                    "identity_overall": fmt(bd["identity_overall"]),
                    "clean_skill": fmt(pred_met["skill_score"]),
                })

                print(f"    {feat_mode}: asgn_acc={fmt(asgn_acc)} swap_only_id={fmt(bd['identity_swap_only'])}")

        except Exception as e:
            print(f"    {model_name} FAILED: {e}")

    save_csv(ablation_results, "feature_ablation.csv",
             ["model", "feature_mode", "swap_ratio", "assignment_accuracy",
              "identity_swap_only", "identity_overall", "clean_skill"])

    # =========================================================================
    # Part 3: Feature Dependency Summary
    # =========================================================================
    print("\n=== Part 3: Feature Dependency Summary ===")

    dependency_results = []

    for model_name in ["MLPPositionFeatureAssignment", "ObjectCentricFeatureAssignment"]:
        normal_rows = [r for r in ablation_results if r["model"] == model_name and r["feature_mode"] == "normal"]
        shuffled_rows = [r for r in ablation_results if r["model"] == model_name and r["feature_mode"] == "shuffled"]
        zero_rows = [r for r in ablation_results if r["model"] == model_name and r["feature_mode"] == "zero"]
        wrong_rows = [r for r in ablation_results if r["model"] == model_name and r["feature_mode"] == "random_wrong"]

        normal_swap = float(normal_rows[0]["identity_swap_only"]) if normal_rows and normal_rows[0]["identity_swap_only"] != "nan" else 0.0
        shuffled_swap = float(shuffled_rows[0]["identity_swap_only"]) if shuffled_rows and shuffled_rows[0]["identity_swap_only"] != "nan" else 0.0
        zero_swap = float(zero_rows[0]["identity_swap_only"]) if zero_rows and zero_rows[0]["identity_swap_only"] != "nan" else 0.0
        wrong_swap = float(wrong_rows[0]["identity_swap_only"]) if wrong_rows and wrong_rows[0]["identity_swap_only"] != "nan" else 0.0

        normal_asgn = float(normal_rows[0]["assignment_accuracy"]) if normal_rows and normal_rows[0]["assignment_accuracy"] != "nan" else 0.0
        shuffled_asgn = float(shuffled_rows[0]["assignment_accuracy"]) if shuffled_rows and shuffled_rows[0]["assignment_accuracy"] != "nan" else 0.0
        zero_asgn = float(zero_rows[0]["assignment_accuracy"]) if zero_rows and zero_rows[0]["assignment_accuracy"] != "nan" else 0.0
        wrong_asgn = float(wrong_rows[0]["assignment_accuracy"]) if wrong_rows and wrong_rows[0]["assignment_accuracy"] != "nan" else 0.0

        dep_score_swap = normal_swap - max(shuffled_swap, zero_swap, wrong_swap)
        dep_score_asgn = normal_asgn - max(shuffled_asgn, zero_asgn, wrong_asgn)

        feature_dep = dep_score_swap > 0.05

        dependency_results.append({
            "model": model_name,
            "normal_identity_swap_only": fmt(normal_swap),
            "shuffled_identity_swap_only": fmt(shuffled_swap),
            "zero_identity_swap_only": fmt(zero_swap),
            "wrong_identity_swap_only": fmt(wrong_swap),
            "normal_assignment_accuracy": fmt(normal_asgn),
            "shuffled_assignment_accuracy": fmt(shuffled_asgn),
            "zero_assignment_accuracy": fmt(zero_asgn),
            "wrong_assignment_accuracy": fmt(wrong_asgn),
            "feature_dependency_score_swap": fmt(dep_score_swap),
            "feature_dependency_score_asgn": fmt(dep_score_asgn),
            "feature_depends_on_features": str(feature_dep),
        })

        print(f"  {model_name}:")
        print(f"    normal={fmt(normal_swap)} shuffled={fmt(shuffled_swap)} zero={fmt(zero_swap)} wrong={fmt(wrong_swap)}")
        print(f"    dep_score_swap={fmt(dep_score_swap)} feature_dep={feature_dep}")

    save_csv(dependency_results, "feature_dependency_summary.csv",
             ["model", "normal_identity_swap_only", "shuffled_identity_swap_only",
              "zero_identity_swap_only", "wrong_identity_swap_only",
              "normal_assignment_accuracy", "shuffled_assignment_accuracy",
              "zero_assignment_accuracy", "wrong_assignment_accuracy",
              "feature_dependency_score_swap", "feature_dependency_score_asgn",
              "feature_depends_on_features"])

    # =========================================================================
    # Plots
    # =========================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Assignment accuracy by swap ratio
        ax = axes[0]
        for model_name, color in [
            ("MLPPositionFeatureAssignment", "coral"),
            ("ObjectCentricFeatureAssignment", "steelblue"),
        ]:
            rows = [r for r in assignment_results if r["model"] == model_name]
            if rows:
                ratios = [float(r["swap_ratio"]) for r in rows]
                vals = [float(r["assignment_accuracy_swap_only"]) for r in rows]
                ax.plot(ratios, vals, 'o-', color=color, label=model_name.replace("Assignment", ""))
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3)
        ax.set_xlabel("Swap Ratio")
        ax.set_ylabel("Assignment Accuracy (swap_only)")
        ax.set_title("Assignment Accuracy")
        ax.legend(fontsize=7)

        # Panel 2: Feature ablation bar chart
        ax = axes[1]
        models_short = ["MLP+Assign", "ObjCentric+Assign"]
        x = np.arange(len(models_short))
        width = 0.2
        for i, (fm, color) in enumerate([("normal", "steelblue"), ("shuffled", "orange"),
                                            ("zero", "gray"), ("random_wrong", "red")]):
            vals = []
            for mn in ["MLPPositionFeatureAssignment", "ObjectCentricFeatureAssignment"]:
                found = [r for r in ablation_results if r["model"] == mn and r["feature_mode"] == fm]
                vals.append(float(found[0]["identity_swap_only"]) if found and found[0]["identity_swap_only"] != "nan" else 0.0)
            ax.bar(x + i * width, vals, width, label=fm, color=color)
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(models_short, fontsize=8)
        ax.set_ylabel("Identity Swap-Only")
        ax.set_title("Feature Ablation (swap_ratio=0.3)")
        ax.legend(fontsize=7)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3)

        # Panel 3: Feature dependency score
        ax = axes[2]
        dep_models = [r["model"].replace("Assignment", "") for r in dependency_results]
        dep_scores = [float(r["feature_dependency_score_swap"]) for r in dependency_results]
        colors = ["green" if s > 0.05 else "red" for s in dep_scores]
        ax.bar(dep_models, dep_scores, color=colors)
        ax.axhline(y=0.05, color="blue", linestyle="--", alpha=0.5, label="Dependency threshold")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.set_ylabel("Feature Dependency Score")
        ax.set_title("Does Identity Depend on Features?")
        ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "feature_binding_plot.png"), dpi=100)
        plt.close()
        print("\n  Saved feature_binding_plot.png")

    except Exception as e:
        print(f"\n  Plot failed: {e}")

    # =========================================================================
    # README
    # =========================================================================
    oc_dep = [r for r in dependency_results if r["model"] == "ObjectCentricFeatureAssignment"]
    oc_feature_dep = bool(oc_dep and oc_dep[0]["feature_depends_on_features"] == "True")

    normal_higher = False
    for dep in dependency_results:
        ns = float(dep["normal_identity_swap_only"])
        ss = float(dep["shuffled_identity_swap_only"])
        zs = float(dep["zero_identity_swap_only"])
        ws = float(dep["wrong_identity_swap_only"])
        if ns > max(ss, zs, ws) + 0.05:
            normal_higher = True

    if normal_higher and oc_feature_dep:
        recommendation = "proceed_to_object_file_models"
    elif oc_feature_dep:
        recommendation = "proceed_to_object_file_models"
    else:
        recommendation = "add_contrastive_feature_binding_loss"

    readme = f"""# SVT-v3.3: Feature-Binding Sanity Test

## 1. Does ObjectCentric truly depend on features?

**{"Yes" if oc_feature_dep else "No"}"""

    if oc_dep:
        readme += f"""

Feature dependency score (swap): {oc_dep[0]["feature_dependency_score_swap"]}
- normal: {oc_dep[0]["normal_identity_swap_only"]}
- shuffled: {oc_dep[0]["shuffled_identity_swap_only"]}
- zero: {oc_dep[0]["zero_identity_swap_only"]}
- random_wrong: {oc_dep[0]["wrong_identity_swap_only"]}

{"ObjectCentricFeatureAssignment's identity prediction degrades when features are corrupted, indicating genuine feature dependency." if oc_feature_dep else "ObjectCentricFeatureAssignment's identity prediction does NOT degrade when features are corrupted. The model is using trajectory shortcuts, not feature binding."}"""

    readme += f"""

## 2. Is normal significantly higher than shuffled/zero?

**{"Yes" if normal_higher else "No"}"""

    for dep in dependency_results:
        readme += f"""

### {dep["model"]}
| Feature Mode | Identity Swap-Only | Assignment Accuracy |
|-------------|-------------------|-------------------|
| normal | {dep["normal_identity_swap_only"]} | {dep["normal_assignment_accuracy"]} |
| shuffled | {dep["shuffled_identity_swap_only"]} | {dep["shuffled_assignment_accuracy"]} |
| zero | {dep["zero_identity_swap_only"]} | {dep["zero_assignment_accuracy"]} |
| random_wrong | {dep["wrong_identity_swap_only"]} | {dep["wrong_assignment_accuracy"]} |

Feature dependency score: {dep["feature_dependency_score_swap"]}"""

    readme += f"""

## 3. Final Recommendation

**{recommendation}**
"""

    with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print("\n" + "=" * 60)
    print("SVT-v3.3 FINAL SUMMARY")
    print("=" * 60)
    print(f"ObjectCentric depends on features: {oc_feature_dep}")
    print(f"Normal > shuffled/zero: {normal_higher}")
    print(f"Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
