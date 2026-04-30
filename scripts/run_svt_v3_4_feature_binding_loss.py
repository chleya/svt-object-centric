"""
SVT-v3.4: Feature-Binding Loss Test
"""

import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v3_4_feature_binding_loss"
SEED = 0
SWAP_RATIO = 0.3
LAMBDAS = [0.0, 0.1, 1.0, 5.0]
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
    asgn_acc = compute_assignment_accuracy(pred_identity, true_identity)
    breakdown = compute_identity_breakdown(pred_identity, true_identity)

    return pred_metrics, breakdown, asgn_acc


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SVT-v3.4: Feature-Binding Loss Test")
    print("=" * 60)
    print(f"PyTorch: {'available' if TORCH_AVAILABLE else 'UNAVAILABLE'} (device={DEVICE})")

    if not TORCH_AVAILABLE:
        print("PyTorch unavailable. Exiting.")
        return

    from models.feature_binding_models import MLPFeatureBinding, ObjectCentricFeatureBinding
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

    # =========================================================================
    # Part 1: Lambda sweep
    # =========================================================================
    print("\n=== Lambda Sweep ===")

    sweep_results = []
    ablation_results = []

    for lambda_bind in LAMBDAS:
        print(f"\n  lambda_bind={lambda_bind}")

        for model_name, model_class in [
            ("MLPFeatureBinding", MLPFeatureBinding),
            ("ObjectCentricFeatureBinding", ObjectCentricFeatureBinding),
        ]:
            print(f"    Training {model_name}...")

            try:
                model = model_class(identity_weight=1.0, lambda_bind=lambda_bind)
                log = train_model(
                    model, train_data, val_data=eval_ds["clean_test_id"],
                    epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
                    uses_features=True, uses_future_features=True, verbose=False,
                )

                swap_test = eval_ds["identity_test_swap_only"]
                pred_met, bd, asgn_acc = evaluate_model(model, swap_test)

                sweep_results.append({
                    "lambda_bind": lambda_bind, "model": model_name,
                    "clean_skill": fmt(pred_met["skill_score"]),
                    "assignment_accuracy": fmt(asgn_acc),
                    "identity_swap_only": fmt(bd["identity_swap_only"]),
                    "identity_overall": fmt(bd["identity_overall"]),
                })

                print(f"      swap_only={fmt(bd['identity_swap_only'])} asgn={fmt(asgn_acc)} skill={fmt(pred_met['skill_score'])}")

                # Feature ablation on this model
                for feat_mode in ["normal", "shuffled", "zero", "random_wrong"]:
                    if feat_mode == "normal":
                        feat_override = None
                    elif feat_mode == "shuffled":
                        feat_override = shuffle_features(swap_test["object_features_obs"], seed=SEED)
                    elif feat_mode == "zero":
                        feat_override = zero_features(swap_test["object_features_obs"])
                    elif feat_mode == "random_wrong":
                        feat_override = random_wrong_features(swap_test["object_features_obs"], seed=SEED)
                    else:
                        feat_override = None

                    pred_met_abl, bd_abl, asgn_acc_abl = evaluate_model(
                        model, swap_test, uses_features=True, feat_override=feat_override)

                    ablation_results.append({
                        "lambda_bind": lambda_bind, "model": model_name,
                        "feature_mode": feat_mode,
                        "identity_swap_only": fmt(bd_abl["identity_swap_only"]),
                        "assignment_accuracy": fmt(asgn_acc_abl),
                        "clean_skill": fmt(pred_met_abl["skill_score"]),
                    })

            except Exception as e:
                print(f"      FAILED: {e}")
                import traceback
                traceback.print_exc()

    save_csv(sweep_results, "binding_loss_sweep.csv",
             ["lambda_bind", "model", "clean_skill", "assignment_accuracy",
              "identity_swap_only", "identity_overall"])

    save_csv(ablation_results, "feature_ablation.csv",
             ["lambda_bind", "model", "feature_mode", "identity_swap_only",
              "assignment_accuracy", "clean_skill"])

    # =========================================================================
    # Part 2: Feature Dependency Summary
    # =========================================================================
    print("\n=== Feature Dependency Summary ===")

    dependency_results = []

    for lambda_bind in LAMBDAS:
        for model_name in ["MLPFeatureBinding", "ObjectCentricFeatureBinding"]:
            normal_rows = [r for r in ablation_results
                           if r["lambda_bind"] == lambda_bind and r["model"] == model_name and r["feature_mode"] == "normal"]
            shuffled_rows = [r for r in ablation_results
                              if r["lambda_bind"] == lambda_bind and r["model"] == model_name and r["feature_mode"] == "shuffled"]
            zero_rows = [r for r in ablation_results
                          if r["lambda_bind"] == lambda_bind and r["model"] == model_name and r["feature_mode"] == "zero"]
            wrong_rows = [r for r in ablation_results
                           if r["lambda_bind"] == lambda_bind and r["model"] == model_name and r["feature_mode"] == "random_wrong"]

            normal_swap = float(normal_rows[0]["identity_swap_only"]) if normal_rows and normal_rows[0]["identity_swap_only"] != "nan" else 0.0
            shuffled_swap = float(shuffled_rows[0]["identity_swap_only"]) if shuffled_rows and shuffled_rows[0]["identity_swap_only"] != "nan" else 0.0
            zero_swap = float(zero_rows[0]["identity_swap_only"]) if zero_rows and zero_rows[0]["identity_swap_only"] != "nan" else 0.0
            wrong_swap = float(wrong_rows[0]["identity_swap_only"]) if wrong_rows and wrong_rows[0]["identity_swap_only"] != "nan" else 0.0

            dep_score = normal_swap - shuffled_swap
            feature_dep = dep_score > 0.2

            dependency_results.append({
                "lambda_bind": lambda_bind, "model": model_name,
                "identity_swap_only_normal": fmt(normal_swap),
                "identity_swap_only_shuffled": fmt(shuffled_swap),
                "identity_swap_only_zero": fmt(zero_swap),
                "identity_swap_only_wrong": fmt(wrong_swap),
                "feature_dependency_score": fmt(dep_score),
                "feature_depends_on_features": str(feature_dep),
            })

            print(f"  lambda={lambda_bind} {model_name}: normal={fmt(normal_swap)} shuffled={fmt(shuffled_swap)} dep={fmt(dep_score)} {'YES' if feature_dep else 'no'}")

    save_csv(dependency_results, "feature_dependency_summary.csv",
             ["lambda_bind", "model", "identity_swap_only_normal",
              "identity_swap_only_shuffled", "identity_swap_only_zero",
              "identity_swap_only_wrong", "feature_dependency_score",
              "feature_depends_on_features"])

    # =========================================================================
    # Plots
    # =========================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Lambda vs identity_swap_only (normal features)
        ax = axes[0]
        for model_name, color in [("MLPFeatureBinding", "coral"), ("ObjectCentricFeatureBinding", "steelblue")]:
            rows = [r for r in sweep_results if r["model"] == model_name]
            if rows:
                lambdas = [float(r["lambda_bind"]) for r in rows]
                vals = [float(r["identity_swap_only"]) for r in rows]
                ax.plot(lambdas, vals, 'o-', color=color, label=model_name.replace("FeatureBinding", "+Bind"))
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3)
        ax.set_xlabel("lambda_bind")
        ax.set_ylabel("Identity Swap-Only (normal features)")
        ax.set_title("Binding Loss Sweep")
        ax.legend(fontsize=7)

        # Panel 2: Feature ablation grouped bar chart (best lambda)
        ax = axes[1]
        best_lambda = None
        best_dep = -999
        for dep in dependency_results:
            ds = float(dep["feature_dependency_score"])
            if ds > best_dep:
                best_dep = ds
                best_lambda = dep["lambda_bind"]

        if best_lambda is not None:
            models_short = ["MLP+Bind", "ObjCent+Bind"]
            x = np.arange(len(models_short))
            width = 0.2
            for i, (fm, color) in enumerate([("normal", "steelblue"), ("shuffled", "orange"),
                                               ("zero", "gray"), ("random_wrong", "red")]):
                vals = []
                for mn in ["MLPFeatureBinding", "ObjectCentricFeatureBinding"]:
                    found = [r for r in ablation_results
                             if r["lambda_bind"] == best_lambda and r["model"] == mn and r["feature_mode"] == fm]
                    vals.append(float(found[0]["identity_swap_only"]) if found and found[0]["identity_swap_only"] != "nan" else 0.0)
                ax.bar(x + i * width, vals, width, label=fm, color=color)
            ax.set_xticks(x + 1.5 * width)
            ax.set_xticklabels(models_short, fontsize=8)
            ax.set_ylabel("Identity Swap-Only")
            ax.set_title(f"Feature Ablation (lambda={best_lambda})")
            ax.legend(fontsize=7)
            ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3)

        # Panel 3: Feature dependency score vs lambda
        ax = axes[2]
        for model_name, color in [("MLPFeatureBinding", "coral"), ("ObjectCentricFeatureBinding", "steelblue")]:
            rows = [r for r in dependency_results if r["model"] == model_name]
            if rows:
                lambdas = [float(r["lambda_bind"]) for r in rows]
                deps = [float(r["feature_dependency_score"]) for r in rows]
                ax.plot(lambdas, deps, 'o-', color=color, label=model_name.replace("FeatureBinding", "+Bind"))
        ax.axhline(y=0.2, color="blue", linestyle="--", alpha=0.5, label="Dependency threshold")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.set_xlabel("lambda_bind")
        ax.set_ylabel("Feature Dependency Score")
        ax.set_title("Does Binding Loss Create Feature Dependency?")
        ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "binding_loss_plot.png"), dpi=100)
        plt.close()
        print("\n  Saved binding_loss_plot.png")

    except Exception as e:
        print(f"\n  Plot failed: {e}")

    # =========================================================================
    # README
    # =========================================================================
    any_dep = any(r["feature_depends_on_features"] == "True" for r in dependency_results)
    best_dep_overall = max(float(r["feature_dependency_score"]) for r in dependency_results)

    if any_dep and best_dep_overall > 0.2:
        recommendation = "proceed_to_object_file_models"
    elif best_dep_overall > 0.05:
        recommendation = "increase_lambda_bind"
    elif best_dep_overall > -0.05:
        recommendation = "add_feature_reconstruction_loss"
    else:
        recommendation = "shortcut_still_dominates"

    readme = f"""# SVT-v3.4: Feature-Binding Loss Test

## 1. Purpose

v3.3 showed that assignment heads achieve high identity accuracy but do NOT depend on features (normal = shuffled). v3.4 adds a contrastive feature binding loss that forces observed/future feature alignment.

## 2. Method

- Feature embedding head: z_obs = encoder(obs_features), z_fut = encoder(fut_features)
- Cosine similarity matrix: [B, N, N]
- Binding loss: CrossEntropy on similarity matrix with identity_labels
- Total loss: mse + identity_ce + lambda_bind * binding_loss
- Lambda sweep: {LAMBDAS}
- Swap ratio: {SWAP_RATIO}

## 3. Binding Loss Sweep

| Lambda | Model | Clean Skill | Swap-Only ID | Assignment Acc |
|--------|-------|------------|-------------|---------------|
"""

    for r in sorted(sweep_results, key=lambda x: (float(x["lambda_bind"]), x["model"])):
        readme += f"| {r['lambda_bind']} | {r['model']} | {r['clean_skill']} | {r['identity_swap_only']} | {r['assignment_accuracy']} |\n"

    readme += f"""
## 4. Feature Dependency

| Lambda | Model | Normal | Shuffled | Zero | Wrong | Dep Score | Feature Dep? |
|--------|-------|--------|----------|------|-------|-----------|-------------|
"""

    for r in sorted(dependency_results, key=lambda x: (float(x["lambda_bind"]), x["model"])):
        readme += f"| {r['lambda_bind']} | {r['model']} | {r['identity_swap_only_normal']} | {r['identity_swap_only_shuffled']} | {r['identity_swap_only_zero']} | {r['identity_swap_only_wrong']} | {r['feature_dependency_score']} | {r['feature_depends_on_features']} |\n"

    readme += f"""
## 5. Key Finding

Best feature dependency score: **{fmt(best_dep_overall)}**

{"Contrastive binding loss creates genuine feature dependency!" if any_dep else "Contrastive binding loss does NOT create feature dependency."}

{"normal > shuffled by >0.2, confirming the model uses features for identity." if any_dep else "normal ≈ shuffled, meaning the model still relies on trajectory shortcuts despite the binding loss."}

## 6. Final Recommendation

**{recommendation}**
"""

    with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print("\n" + "=" * 60)
    print("SVT-v3.4 FINAL SUMMARY")
    print("=" * 60)
    print(f"Best feature dependency score: {fmt(best_dep_overall)}")
    print(f"Any model with feature_dep=True: {any_dep}")
    print(f"Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
