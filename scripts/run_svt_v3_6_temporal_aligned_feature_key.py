"""
SVT-v3.6: Temporal-Aligned Feature Key Fix

v3.5.1 proved mean pooling destroys swap-pre identity info (75% ceiling).
obs=first achieves oracle 100%. This script uses first-timestep pooling.
"""

import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v3_6_temporal_aligned_feature_key"
SEED = 0
SWAP_RATIO = 0.3
EPOCHS = 20

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
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
    B, T, N, F_dim = shuffled.shape
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
    B, T, N, F_dim = features.shape
    wrong = rng.rand(B, T, N, F_dim)
    wrong = (wrong > 0.5).astype(np.float32)
    row_sums = wrong.sum(axis=-1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    wrong = wrong / row_sums
    return wrong


def compute_assignment_accuracy(pred_identity, true_identity):
    correct = (pred_identity == true_identity).all(axis=1)
    return float(correct.mean())


def apply_feature_mode_pair(obs_feat, fut_feat, feat_mode, seed=0):
    if feat_mode == "normal":
        return obs_feat, fut_feat
    elif feat_mode == "shuffled":
        return shuffle_features(obs_feat, seed=seed), shuffle_features(fut_feat, seed=seed + 1000)
    elif feat_mode == "zero":
        return zero_features(obs_feat), zero_features(fut_feat)
    elif feat_mode == "random_wrong":
        return random_wrong_features(obs_feat, seed=seed), random_wrong_features(fut_feat, seed=seed + 1000)
    return obs_feat, fut_feat


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


def decompose_hybrid_pathways(model, test_data, obs_feat, fut_feat):
    model.eval()
    with torch.no_grad():
        obs_pos_t = torch.FloatTensor(test_data["observed_positions"])
        obs_feat_t = torch.FloatTensor(obs_feat) if obs_feat is not None else None
        fut_feat_t = torch.FloatTensor(fut_feat) if fut_feat is not None else None

        B = obs_pos_t.shape[0]

        if obs_feat_t is not None:
            x = torch.cat([obs_pos_t, obs_feat_t], dim=-1)
        else:
            padding = torch.zeros(B, model.t_obs, model.num_objects, model.feature_dim)
            x = torch.cat([obs_pos_t, padding], dim=-1)

        x = x.reshape(B, -1)
        shared_out = model.shared(x)

        traj_logits = model.traj_assignment_head(shared_out).reshape(B, model.num_objects, model.num_objects)

        feature_logits = None
        if obs_feat_t is not None and fut_feat_t is not None:
            from models.feature_similarity_models import compute_feature_similarity_logits
            feature_logits = compute_feature_similarity_logits(
                model.feature_encoder, obs_feat_t, fut_feat_t,
                model.num_objects, model.temperature)

        true_identity = test_data["identity_labels"]

        pred_traj = traj_logits.argmax(dim=-1).numpy()
        bd_traj = compute_identity_breakdown(pred_traj, true_identity)

        bd_feat = {"identity_swap_only": float("nan")}
        if feature_logits is not None:
            pred_feat = feature_logits.argmax(dim=-1).numpy()
            bd_feat = compute_identity_breakdown(pred_feat, true_identity)

        if feature_logits is not None:
            hybrid_logits = traj_logits + model.beta * feature_logits
        else:
            hybrid_logits = traj_logits
        pred_hybrid = hybrid_logits.argmax(dim=-1).numpy()
        bd_hybrid = compute_identity_breakdown(pred_hybrid, true_identity)

        return bd_traj, bd_feat, bd_hybrid


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SVT-v3.6: Temporal-Aligned Feature Key Fix")
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

    swap_test = eval_ds["identity_test_swap_only"]

    all_results = []
    ablation_results = []
    decomp_results = []

    configs = [
        ("FeatureOnly", FeatureOnlyAssignmentHead, None),
        ("Hybrid_b1.0", HybridTrajectoryFeatureAssignmentHead, 1.0),
        ("Hybrid_b2.0", HybridTrajectoryFeatureAssignmentHead, 2.0),
    ]

    for model_name, model_class, beta in configs:
        print(f"\n=== {model_name} ===")

        if beta is not None:
            model = model_class(identity_weight=1.0, beta=beta)
        else:
            model = model_class(identity_weight=1.0)

        log = train_model(
            model, train_data, val_data=eval_ds["clean_test_id"],
            epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
            uses_features=True, uses_future_features=True, verbose=False,
        )

        pred_met, bd, asgn_acc = evaluate_model(model, swap_test)

        all_results.append({
            "model": model_name,
            "beta": str(beta) if beta is not None else "N/A",
            "clean_skill": fmt(pred_met["skill_score"]),
            "assignment_accuracy": fmt(asgn_acc),
            "identity_swap_only": fmt(bd["identity_swap_only"]),
            "identity_overall": fmt(bd["identity_overall"]),
        })
        print(f"  swap_only={fmt(bd['identity_swap_only'])} asgn={fmt(asgn_acc)} skill={fmt(pred_met['skill_score'])}")

        for feat_mode in ["normal", "shuffled", "zero", "random_wrong"]:
            obs_override, fut_override = apply_feature_mode_pair(
                swap_test["object_features_obs"],
                swap_test["object_features_fut"],
                feat_mode, seed=SEED)

            pred_met_abl, bd_abl, asgn_acc_abl = evaluate_model(
                model, swap_test, uses_features=True,
                obs_feat_override=obs_override, fut_feat_override=fut_override)

            ablation_results.append({
                "model": model_name,
                "beta": str(beta) if beta is not None else "N/A",
                "feature_mode": feat_mode,
                "identity_swap_only": fmt(bd_abl["identity_swap_only"]),
                "assignment_accuracy": fmt(asgn_acc_abl),
                "clean_skill": fmt(pred_met_abl["skill_score"]),
            })
            print(f"    {feat_mode}: swap_only={fmt(bd_abl['identity_swap_only'])}")

        if beta is not None:
            print(f"  Pathway decomposition:")
            for feat_mode in ["normal", "shuffled", "zero", "random_wrong"]:
                obs_mod, fut_mod = apply_feature_mode_pair(
                    swap_test["object_features_obs"],
                    swap_test["object_features_fut"],
                    feat_mode, seed=SEED)

                bd_traj, bd_feat, bd_hybrid = decompose_hybrid_pathways(
                    model, swap_test, obs_mod, fut_mod)

                decomp_results.append({
                    "model": model_name,
                    "beta": str(beta),
                    "feature_mode": feat_mode,
                    "traj_swap_only": fmt(bd_traj["identity_swap_only"]),
                    "feat_swap_only": fmt(bd_feat["identity_swap_only"]),
                    "hybrid_swap_only": fmt(bd_hybrid["identity_swap_only"]),
                })
                print(f"    {feat_mode}: traj={fmt(bd_traj['identity_swap_only'])} feat={fmt(bd_feat['identity_swap_only'])} hybrid={fmt(bd_hybrid['identity_swap_only'])}")

    # Dependency scores for decomposition
    for model_name, beta_val in [("Hybrid_b1.0", "1.0"), ("Hybrid_b2.0", "2.0")]:
        normal_rows = [r for r in decomp_results
                       if r["model"] == model_name and r["feature_mode"] == "normal"]
        shuffled_rows = [r for r in decomp_results
                         if r["model"] == model_name and r["feature_mode"] == "shuffled"]

        if normal_rows and shuffled_rows:
            n_traj = float(normal_rows[0]["traj_swap_only"])
            s_traj = float(shuffled_rows[0]["traj_swap_only"])
            n_feat = float(normal_rows[0]["feat_swap_only"])
            s_feat = float(shuffled_rows[0]["feat_swap_only"])
            n_hybrid = float(normal_rows[0]["hybrid_swap_only"])
            s_hybrid = float(shuffled_rows[0]["hybrid_swap_only"])

            decomp_results.append({
                "model": model_name,
                "beta": beta_val,
                "feature_mode": "dep_score",
                "traj_swap_only": fmt(n_traj - s_traj),
                "feat_swap_only": fmt(n_feat - s_feat),
                "hybrid_swap_only": fmt(n_hybrid - s_hybrid),
            })

    save_csv(all_results, "temporal_aligned_results.csv",
             ["model", "beta", "clean_skill", "assignment_accuracy",
              "identity_swap_only", "identity_overall"])

    save_csv(ablation_results, "feature_ablation.csv",
             ["model", "beta", "feature_mode", "identity_swap_only",
              "assignment_accuracy", "clean_skill"])

    save_csv(decomp_results, "pathway_decomposition.csv",
             ["model", "beta", "feature_mode", "traj_swap_only",
              "feat_swap_only", "hybrid_swap_only"])

    # =========================================================================
    # README
    # =========================================================================
    fo_normal_rows = [r for r in ablation_results
                      if r["model"] == "FeatureOnly" and r["feature_mode"] == "normal"]
    fo_shuffled_rows = [r for r in ablation_results
                        if r["model"] == "FeatureOnly" and r["feature_mode"] == "shuffled"]
    fo_zero_rows = [r for r in ablation_results
                    if r["model"] == "FeatureOnly" and r["feature_mode"] == "zero"]

    fo_normal_swap = float(fo_normal_rows[0]["identity_swap_only"]) if fo_normal_rows else 0.0
    fo_shuffled_swap = float(fo_shuffled_rows[0]["identity_swap_only"]) if fo_shuffled_rows else 0.0
    fo_zero_swap = float(fo_zero_rows[0]["identity_swap_only"]) if fo_zero_rows else 0.0
    fo_dep = fo_normal_swap - fo_shuffled_swap

    q1_improved = fo_normal_swap > 0.9
    q2_normal_dominates = fo_normal_swap > fo_shuffled_swap + 0.2 and fo_normal_swap > fo_zero_swap + 0.2

    decomp_dep_rows = [r for r in decomp_results if r["feature_mode"] == "dep_score"]
    best_traj_dep = max(float(r["traj_swap_only"]) for r in decomp_dep_rows) if decomp_dep_rows else 0.0
    best_feat_dep = max(float(r["feat_swap_only"]) for r in decomp_dep_rows) if decomp_dep_rows else 0.0

    q3_feature_pathway = best_feat_dep > best_traj_dep

    if q1_improved and q2_normal_dominates and q3_feature_pathway:
        recommendation = "proceed_to_minimal_object_file"
    elif q1_improved and q2_normal_dominates:
        recommendation = "proceed_to_minimal_object_file"
    elif not q1_improved:
        recommendation = "feature_alignment_bug_remaining"
    else:
        recommendation = "shortcut_still_dominates"

    readme = f"""# SVT-v3.6: Temporal-Aligned Feature Key Fix

## 1. Purpose

v3.5.1 proved mean pooling destroys swap-pre identity info (75% ceiling).
Oracle with obs=first achieves 100%. v3.6 uses first-timestep pooling.

## 2. Results

| Model | Beta | Clean Skill | Swap-Only ID | Assignment Acc |
|-------|------|------------|-------------|---------------|
"""

    for r in all_results:
        readme += f"| {r['model']} | {r['beta']} | {r['clean_skill']} | {r['identity_swap_only']} | {r['assignment_accuracy']} |\n"

    readme += f"""
## 3. Feature Ablation

| Model | Beta | Feature Mode | Swap-Only ID |
|-------|------|-------------|-------------|
"""

    for r in ablation_results:
        readme += f"| {r['model']} | {r['beta']} | {r['feature_mode']} | {r['identity_swap_only']} |\n"

    readme += f"""
## 4. Pathway Decomposition

| Model | Beta | Feature Mode | Traj Swap | Feat Swap | Hybrid Swap |
|-------|------|-------------|-----------|-----------|-------------|
"""

    for r in decomp_results:
        readme += f"| {r['model']} | {r['beta']} | {r['feature_mode']} | {r['traj_swap_only']} | {r['feat_swap_only']} | {r['hybrid_swap_only']} |\n"

    readme += f"""
## 5. Answers

### Q1: Did FeatureOnly improve from 0.75 to near 1.0?

FeatureOnly swap-only identity: **{fmt(fo_normal_swap)}**

{"YES - first-timestep pooling fixed the ceiling." if q1_improved else "NO - the ceiling persists despite first-timestep pooling."}

### Q2: Is normal significantly higher than shuffled/zero/wrong?

normal={fmt(fo_normal_swap)}, shuffled={fmt(fo_shuffled_swap)}, zero={fmt(fo_zero_swap)}

{"YES - normal dominates all ablated conditions." if q2_normal_dominates else "NO - normal does not sufficiently dominate ablated conditions."}

### Q3: Does Hybrid's high score come from feature pathway or trajectory shortcut?

Traj dep_score: {fmt(best_traj_dep)}, Feat dep_score: {fmt(best_feat_dep)}

{"Feature pathway dominates." if q3_feature_pathway else "Trajectory shortcut dominates."}

### Q4: Final Recommendation

**{recommendation}**
"""

    with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print("\n" + "=" * 60)
    print("SVT-v3.6 FINAL SUMMARY")
    print("=" * 60)
    print(f"FeatureOnly swap-only (normal): {fmt(fo_normal_swap)}")
    print(f"FeatureOnly swap-only (shuffled): {fmt(fo_shuffled_swap)}")
    print(f"FeatureOnly swap-only (zero): {fmt(fo_zero_swap)}")
    print(f"Feature dependency score: {fmt(fo_dep)}")
    print(f"Best traj dep: {fmt(best_traj_dep)}, Best feat dep: {fmt(best_feat_dep)}")
    print(f"Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
