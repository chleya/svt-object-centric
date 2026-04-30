"""
SVT-v3.5: Feature-Similarity Assignment Head Test
"""

import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v3_5_feature_similarity_head"
SEED = 0
SWAP_RATIO = 0.3
BETAS = [0.0, 0.5, 1.0, 2.0, 5.0]
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


def shuffle_features(features, seed=42):
    if features is None:
        return None
    rng = np.random.RandomState(seed)
    shuffled = features.copy()
    B, T, N, F = shuffled.shape
    for b in range(B):
        for t in range(T):
            perm = rng.permutation(N)
            shuffled[b, t] = shuffled[b, t, perm]
    return shuffled


def zero_features(features):
    if features is None:
        return None
    return np.zeros_like(features)


def random_wrong_features(features, seed=42):
    if features is None:
        return None
    rng = np.random.RandomState(seed)
    B, T, N, F = features.shape
    wrong = rng.rand(B, T, N, F)
    wrong = (wrong > 0.5).astype(np.float32)
    row_sums = wrong.sum(axis=-1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    wrong = wrong / row_sums
    return wrong


def compute_assignment_accuracy(pred_identity, true_identity):
    correct = (pred_identity == true_identity).all(axis=1)
    return float(correct.mean())


def evaluate_model(model, test_data, uses_features=True,
                   obs_feat_override=None, fut_feat_override=None):
    obs_pos = test_data["observed_positions"]
    obs_feat = obs_feat_override if obs_feat_override is not None else (
        test_data.get("object_features_obs") if uses_features else None)
    fut_feat = fut_feat_override if fut_feat_override is not None else (
        test_data.get("object_features_fut") if uses_features else None)

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


def apply_feature_mode(test_data, feat_mode, seed=0):
    obs_feat = test_data.get("object_features_obs")
    fut_feat = test_data.get("object_features_fut")

    if feat_mode == "normal":
        return obs_feat, fut_feat
    elif feat_mode == "shuffled":
        return shuffle_features(obs_feat, seed=seed), shuffle_features(fut_feat, seed=seed + 1000)
    elif feat_mode == "zero":
        return zero_features(obs_feat), zero_features(fut_feat)
    elif feat_mode == "random_wrong":
        return random_wrong_features(obs_feat, seed=seed), random_wrong_features(fut_feat, seed=seed + 1000)
    return obs_feat, fut_feat


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SVT-v3.5: Feature-Similarity Assignment Head Test")
    print("=" * 60)
    print(f"PyTorch: {'available' if TORCH_AVAILABLE else 'UNAVAILABLE'} (device={DEVICE})")

    if not TORCH_AVAILABLE:
        print("PyTorch unavailable. Exiting.")
        return

    from models.feature_similarity_models import FeatureOnlyAssignmentHead, HybridTrajectoryFeatureAssignmentHead
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

    all_results = []
    ablation_results = []

    # =========================================================================
    # Part 1: FeatureOnlyAssignmentHead
    # =========================================================================
    print("\n=== FeatureOnlyAssignmentHead ===")

    model = FeatureOnlyAssignmentHead(identity_weight=1.0)
    log = train_model(
        model, train_data, val_data=eval_ds["clean_test_id"],
        epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
        uses_features=True, uses_future_features=True, verbose=False,
    )

    swap_test = eval_ds["identity_test_swap_only"]
    pred_met, bd, asgn_acc = evaluate_model(model, swap_test)

    all_results.append({
        "model": "FeatureOnlyAssignmentHead",
        "beta": "N/A",
        "clean_skill": fmt(pred_met["skill_score"]),
        "assignment_accuracy": fmt(asgn_acc),
        "identity_swap_only": fmt(bd["identity_swap_only"]),
        "identity_overall": fmt(bd["identity_overall"]),
    })
    print(f"  FeatureOnly: swap_only={fmt(bd['identity_swap_only'])} asgn={fmt(asgn_acc)} skill={fmt(pred_met['skill_score'])}")

    for feat_mode in ["normal", "shuffled", "zero", "random_wrong"]:
        obs_override, fut_override = apply_feature_mode(swap_test, feat_mode, seed=SEED)
        pred_met_abl, bd_abl, asgn_acc_abl = evaluate_model(
            model, swap_test, uses_features=True,
            obs_feat_override=obs_override, fut_feat_override=fut_override)

        ablation_results.append({
            "model": "FeatureOnlyAssignmentHead",
            "beta": "N/A",
            "feature_mode": feat_mode,
            "identity_swap_only": fmt(bd_abl["identity_swap_only"]),
            "assignment_accuracy": fmt(asgn_acc_abl),
            "clean_skill": fmt(pred_met_abl["skill_score"]),
        })
        print(f"    {feat_mode}: swap_only={fmt(bd_abl['identity_swap_only'])} asgn={fmt(asgn_acc_abl)}")

    # =========================================================================
    # Part 2: HybridTrajectoryFeatureAssignmentHead beta sweep
    # =========================================================================
    print("\n=== HybridTrajectoryFeatureAssignmentHead Beta Sweep ===")

    for beta in BETAS:
        print(f"\n  beta={beta}")

        model = HybridTrajectoryFeatureAssignmentHead(identity_weight=1.0, beta=beta)
        log = train_model(
            model, train_data, val_data=eval_ds["clean_test_id"],
            epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
            uses_features=True, uses_future_features=True, verbose=False,
        )

        pred_met, bd, asgn_acc = evaluate_model(model, swap_test)

        all_results.append({
            "model": "HybridTrajectoryFeatureAssignmentHead",
            "beta": str(beta),
            "clean_skill": fmt(pred_met["skill_score"]),
            "assignment_accuracy": fmt(asgn_acc),
            "identity_swap_only": fmt(bd["identity_swap_only"]),
            "identity_overall": fmt(bd["identity_overall"]),
        })
        print(f"    Hybrid(beta={beta}): swap_only={fmt(bd['identity_swap_only'])} asgn={fmt(asgn_acc)} skill={fmt(pred_met['skill_score'])}")

        for feat_mode in ["normal", "shuffled", "zero", "random_wrong"]:
            obs_override, fut_override = apply_feature_mode(swap_test, feat_mode, seed=SEED)
            pred_met_abl, bd_abl, asgn_acc_abl = evaluate_model(
                model, swap_test, uses_features=True,
                obs_feat_override=obs_override, fut_feat_override=fut_override)

            ablation_results.append({
                "model": "HybridTrajectoryFeatureAssignmentHead",
                "beta": str(beta),
                "feature_mode": feat_mode,
                "identity_swap_only": fmt(bd_abl["identity_swap_only"]),
                "assignment_accuracy": fmt(asgn_acc_abl),
                "clean_skill": fmt(pred_met_abl["skill_score"]),
            })
            print(f"      {feat_mode}: swap_only={fmt(bd_abl['identity_swap_only'])}")

    save_csv(all_results, "feature_similarity_results.csv",
             ["model", "beta", "clean_skill", "assignment_accuracy",
              "identity_swap_only", "identity_overall"])

    save_csv(ablation_results, "feature_ablation.csv",
             ["model", "beta", "feature_mode", "identity_swap_only",
              "assignment_accuracy", "clean_skill"])

    beta_sweep_rows = [r for r in all_results if r["model"] == "HybridTrajectoryFeatureAssignmentHead"]
    save_csv(beta_sweep_rows, "beta_sweep.csv",
             ["beta", "model", "clean_skill", "assignment_accuracy",
              "identity_swap_only", "identity_overall"])

    # =========================================================================
    # Part 3: Feature Dependency Summary
    # =========================================================================
    print("\n=== Feature Dependency Summary ===")

    dependency_results = []

    configs = [("FeatureOnlyAssignmentHead", "N/A")]
    for beta in BETAS:
        configs.append(("HybridTrajectoryFeatureAssignmentHead", str(beta)))

    for model_name, beta_val in configs:
        normal_rows = [r for r in ablation_results
                       if r["model"] == model_name and r["beta"] == beta_val and r["feature_mode"] == "normal"]
        shuffled_rows = [r for r in ablation_results
                         if r["model"] == model_name and r["beta"] == beta_val and r["feature_mode"] == "shuffled"]
        zero_rows = [r for r in ablation_results
                      if r["model"] == model_name and r["beta"] == beta_val and r["feature_mode"] == "zero"]
        wrong_rows = [r for r in ablation_results
                       if r["model"] == model_name and r["beta"] == beta_val and r["feature_mode"] == "random_wrong"]

        normal_swap = float(normal_rows[0]["identity_swap_only"]) if normal_rows and normal_rows[0]["identity_swap_only"] != "nan" else 0.0
        shuffled_swap = float(shuffled_rows[0]["identity_swap_only"]) if shuffled_rows and shuffled_rows[0]["identity_swap_only"] != "nan" else 0.0
        zero_swap = float(zero_rows[0]["identity_swap_only"]) if zero_rows and zero_rows[0]["identity_swap_only"] != "nan" else 0.0
        wrong_swap = float(wrong_rows[0]["identity_swap_only"]) if wrong_rows and wrong_rows[0]["identity_swap_only"] != "nan" else 0.0

        dep_score = normal_swap - shuffled_swap
        feature_dep = dep_score > 0.2

        dependency_results.append({
            "model": model_name,
            "beta": beta_val,
            "identity_swap_only_normal": fmt(normal_swap),
            "identity_swap_only_shuffled": fmt(shuffled_swap),
            "identity_swap_only_zero": fmt(zero_swap),
            "identity_swap_only_wrong": fmt(wrong_swap),
            "feature_dependency_score": fmt(dep_score),
            "feature_depends_on_features": str(feature_dep),
        })

        print(f"  {model_name}(beta={beta_val}): normal={fmt(normal_swap)} shuffled={fmt(shuffled_swap)} dep={fmt(dep_score)} {'YES' if feature_dep else 'no'}")

    save_csv(dependency_results, "feature_dependency_summary.csv",
             ["model", "beta", "identity_swap_only_normal",
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

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Panel 1: Beta vs identity_swap_only (normal features)
        ax = axes[0]
        hybrid_normal = [r for r in ablation_results
                         if r["model"] == "HybridTrajectoryFeatureAssignmentHead" and r["feature_mode"] == "normal"]
        hybrid_shuffled = [r for r in ablation_results
                           if r["model"] == "HybridTrajectoryFeatureAssignmentHead" and r["feature_mode"] == "shuffled"]
        if hybrid_normal:
            betas_plot = [float(r["beta"]) for r in hybrid_normal]
            normal_vals = [float(r["identity_swap_only"]) for r in hybrid_normal]
            shuffled_vals = [float(r["identity_swap_only"]) for r in hybrid_shuffled]
            ax.plot(betas_plot, normal_vals, 'o-', color="steelblue", label="normal features")
            ax.plot(betas_plot, shuffled_vals, 's--', color="orange", label="shuffled features")
        fo_normal = [r for r in ablation_results
                     if r["model"] == "FeatureOnlyAssignmentHead" and r["feature_mode"] == "normal"]
        fo_shuffled = [r for r in ablation_results
                       if r["model"] == "FeatureOnlyAssignmentHead" and r["feature_mode"] == "shuffled"]
        if fo_normal:
            fo_n = float(fo_normal[0]["identity_swap_only"])
            fo_s = float(fo_shuffled[0]["identity_swap_only"])
            ax.axhline(y=fo_n, color="steelblue", linestyle=":", alpha=0.5, label=f"FeatureOnly normal={fo_n:.2f}")
            ax.axhline(y=fo_s, color="orange", linestyle=":", alpha=0.5, label=f"FeatureOnly shuffled={fo_s:.2f}")
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3)
        ax.set_xlabel("beta")
        ax.set_ylabel("Identity Swap-Only")
        ax.set_title("Beta Sweep: Normal vs Shuffled")
        ax.legend(fontsize=6)

        # Panel 2: Feature ablation grouped bar chart
        ax = axes[1]
        best_dep_row = max(dependency_results, key=lambda r: float(r["feature_dependency_score"]))
        best_beta = best_dep_row["beta"]
        best_model = best_dep_row["model"]

        modes = ["normal", "shuffled", "zero", "random_wrong"]
        mode_colors = ["steelblue", "orange", "gray", "red"]

        if best_model == "FeatureOnlyAssignmentHead":
            group_labels = ["FeatureOnly"]
            group_data = []
            for fm in modes:
                found = [r for r in ablation_results
                         if r["model"] == best_model and r["feature_mode"] == fm]
                group_data.append(float(found[0]["identity_swap_only"]) if found else 0.0)
            x = np.arange(1)
            width = 0.18
            for i, (fm, color) in enumerate(zip(modes, mode_colors)):
                ax.bar(x + i * width, [group_data[i]], width, label=fm, color=color)
            ax.set_xticks(x + 1.5 * width)
            ax.set_xticklabels(group_labels, fontsize=8)
        else:
            group_labels = [f"beta={b}" for b in BETAS]
            x = np.arange(len(BETAS))
            width = 0.18
            for i, (fm, color) in enumerate(zip(modes, mode_colors)):
                vals = []
                for b in BETAS:
                    found = [r for r in ablation_results
                             if r["model"] == "HybridTrajectoryFeatureAssignmentHead"
                             and r["beta"] == str(b) and r["feature_mode"] == fm]
                    vals.append(float(found[0]["identity_swap_only"]) if found else 0.0)
                ax.bar(x + i * width, vals, width, label=fm, color=color)
            ax.set_xticks(x + 1.5 * width)
            ax.set_xticklabels(group_labels, fontsize=7)
        ax.set_ylabel("Identity Swap-Only")
        ax.set_title("Feature Ablation by Beta")
        ax.legend(fontsize=7)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3)

        # Panel 3: Feature dependency score vs beta
        ax = axes[2]
        hybrid_deps = [r for r in dependency_results if r["model"] == "HybridTrajectoryFeatureAssignmentHead"]
        if hybrid_deps:
            betas_dep = [float(r["beta"]) for r in hybrid_deps]
            dep_scores = [float(r["feature_dependency_score"]) for r in hybrid_deps]
            ax.plot(betas_dep, dep_scores, 'o-', color="steelblue", label="Hybrid")
        fo_deps = [r for r in dependency_results if r["model"] == "FeatureOnlyAssignmentHead"]
        if fo_deps:
            fo_dep = float(fo_deps[0]["feature_dependency_score"])
            ax.axhline(y=fo_dep, color="coral", linestyle=":", alpha=0.7, label=f"FeatureOnly dep={fo_dep:.3f}")
        ax.axhline(y=0.2, color="blue", linestyle="--", alpha=0.5, label="Threshold 0.2")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.set_xlabel("beta")
        ax.set_ylabel("Feature Dependency Score")
        ax.set_title("Does Feature Similarity Head Create Dependency?")
        ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "feature_similarity_plot.png"), dpi=100)
        plt.close()
        print("\n  Saved feature_similarity_plot.png")

    except Exception as e:
        print(f"\n  Plot failed: {e}")

    # =========================================================================
    # README
    # =========================================================================
    any_dep = any(r["feature_depends_on_features"] == "True" for r in dependency_results)
    best_dep_overall = max(float(r["feature_dependency_score"]) for r in dependency_results)

    fo_dep_rows = [r for r in dependency_results if r["model"] == "FeatureOnlyAssignmentHead"]
    fo_dep_score = float(fo_dep_rows[0]["feature_dependency_score"]) if fo_dep_rows else 0.0

    hybrid_dep_rows = [r for r in dependency_results if r["model"] == "HybridTrajectoryFeatureAssignmentHead"]
    hybrid_dep_increases = False
    if len(hybrid_dep_rows) >= 2:
        dep_by_beta = [(float(r["beta"]), float(r["feature_dependency_score"])) for r in hybrid_dep_rows]
        dep_by_beta.sort()
        for i in range(1, len(dep_by_beta)):
            if dep_by_beta[i][1] > dep_by_beta[0][1] + 0.05:
                hybrid_dep_increases = True
                break

    if any_dep and best_dep_overall > 0.2:
        recommendation = "feature_similarity_head_required"
    elif hybrid_dep_increases and best_dep_overall > 0.05:
        recommendation = "increase_lambda_bind"
    elif best_dep_overall > -0.05:
        recommendation = "fix_feature_pipeline"
    else:
        recommendation = "shortcut_still_dominates"

    readme = f"""# SVT-v3.5: Feature-Similarity Assignment Head Test

## 1. Purpose

v3.4 showed contrastive binding loss fails to create feature dependency. v3.5 changes the architecture: feature similarity directly participates in the assignment decision, rather than being an auxiliary loss.

## 2. Models

### FeatureOnlyAssignmentHead
- assignment_logits = cosine_similarity(z_fut, z_obs) / temperature
- No trajectory-based assignment logits
- Identity decision is purely feature-driven

### HybridTrajectoryFeatureAssignmentHead
- trajectory_logits from learned assignment head
- feature_logits from feature cosine similarity
- assignment_logits = trajectory_logits + beta * feature_logits
- Beta sweep: {BETAS}

## 3. Results

| Model | Beta | Clean Skill | Swap-Only ID | Assignment Acc |
|-------|------|------------|-------------|---------------|
"""

    for r in sorted(all_results, key=lambda x: (x["model"], x["beta"])):
        readme += f"| {r['model']} | {r['beta']} | {r['clean_skill']} | {r['identity_swap_only']} | {r['assignment_accuracy']} |\n"

    readme += f"""
## 4. Feature Dependency

| Model | Beta | Normal | Shuffled | Zero | Wrong | Dep Score | Feature Dep? |
|-------|------|--------|----------|------|-------|-----------|-------------|
"""

    for r in sorted(dependency_results, key=lambda x: (x["model"], x["beta"])):
        readme += f"| {r['model']} | {r['beta']} | {r['identity_swap_only_normal']} | {r['identity_swap_only_shuffled']} | {r['identity_swap_only_zero']} | {r['identity_swap_only_wrong']} | {r['feature_dependency_score']} | {r['feature_depends_on_features']} |\n"

    readme += f"""
## 5. Key Findings

- FeatureOnlyAssignmentHead dependency score: **{fmt(fo_dep_score)}**
- Best overall dependency score: **{fmt(best_dep_overall)}**
- Hybrid beta increases feature dependency: **{hybrid_dep_increases}**

{"Feature similarity head creates genuine feature dependency!" if any_dep else "Feature similarity head does NOT create feature dependency."}

{"normal > shuffled by >0.2, confirming the model uses features for identity." if any_dep else "normal ≈ shuffled across all configurations, meaning trajectory shortcuts still dominate even when feature similarity directly controls assignment."}

## 6. Final Recommendation

**{recommendation}**
"""

    with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print("\n" + "=" * 60)
    print("SVT-v3.5 FINAL SUMMARY")
    print("=" * 60)
    print(f"FeatureOnly dependency score: {fmt(fo_dep_score)}")
    print(f"Best feature dependency score: {fmt(best_dep_overall)}")
    print(f"Any model with feature_dep=True: {any_dep}")
    print(f"Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
