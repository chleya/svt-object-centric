"""
SVT-v3.5.1: Feature Pathway Audit

Three experiments:
1. FeatureOnly Oracle Check - raw one-hot features, no trained encoder
2. Temporal Pooling Ablation - compare pooling strategies
3. Hybrid Inference Decomposition - separate trajectory vs feature pathways
"""

import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v3_5_1_feature_pathway_audit"
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


def pool_features(features, strategy):
    if features is None:
        return None
    if strategy == "mean":
        return features.mean(axis=1)
    elif strategy == "first":
        return features[:, 0, :, :]
    elif strategy == "last":
        return features[:, -1, :, :]
    else:
        return features.mean(axis=1)


def oracle_assignment(obs_feat_pooled, fut_feat_pooled, num_objects=2):
    B = obs_feat_pooled.shape[0]
    obs_t = torch.FloatTensor(obs_feat_pooled)
    fut_t = torch.FloatTensor(fut_feat_pooled)

    obs_norm = F.normalize(obs_t, dim=-1)
    fut_norm = F.normalize(fut_t, dim=-1)

    sim_matrix = torch.bmm(fut_norm, obs_norm.transpose(1, 2))

    pred_assignment = sim_matrix.argmax(dim=-1).numpy()
    return pred_assignment


def compute_identity_metrics(pred_identity, true_identity):
    breakdown = compute_identity_breakdown(pred_identity, true_identity)
    correct = (pred_identity == true_identity).all(axis=1)
    asgn_acc = float(correct.mean())
    return breakdown, asgn_acc


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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SVT-v3.5.1: Feature Pathway Audit")
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
    true_identity = swap_test["identity_labels"]
    obs_feat = swap_test["object_features_obs"]
    fut_feat = swap_test["object_features_fut"]

    # =========================================================================
    # Experiment 1: FeatureOnly Oracle Check
    # =========================================================================
    print("\n=== Experiment 1: FeatureOnly Oracle Check ===")
    print("  Using raw one-hot features (no trained encoder)")

    oracle_results = []

    for pool_obs in ["mean", "first", "last"]:
        for pool_fut in ["mean", "first", "last"]:
            obs_pooled = pool_features(obs_feat, pool_obs)
            fut_pooled = pool_features(fut_feat, pool_fut)

            pred_identity = oracle_assignment(obs_pooled, fut_pooled)
            bd, asgn_acc = compute_identity_metrics(pred_identity, true_identity)

            oracle_results.append({
                "pool_obs": pool_obs,
                "pool_fut": pool_fut,
                "identity_swap_only": fmt(bd["identity_swap_only"]),
                "identity_overall": fmt(bd["identity_overall"]),
                "assignment_accuracy": fmt(asgn_acc),
            })
            print(f"  obs={pool_obs} fut={pool_fut}: swap_only={fmt(bd['identity_swap_only'])} overall={fmt(bd['identity_overall'])}")

    print("\n  Oracle with feature ablation (mean/mean pooling):")
    for feat_mode in ["normal", "shuffled", "zero", "random_wrong"]:
        obs_mod, fut_mod = apply_feature_mode_pair(obs_feat, fut_feat, feat_mode, seed=SEED)
        obs_pooled = pool_features(obs_mod, "mean")
        fut_pooled = pool_features(fut_mod, "mean")

        pred_identity = oracle_assignment(obs_pooled, fut_pooled)
        bd, asgn_acc = compute_identity_metrics(pred_identity, true_identity)

        oracle_results.append({
            "pool_obs": f"mean_{feat_mode}",
            "pool_fut": f"mean_{feat_mode}",
            "identity_swap_only": fmt(bd["identity_swap_only"]),
            "identity_overall": fmt(bd["identity_overall"]),
            "assignment_accuracy": fmt(asgn_acc),
        })
        print(f"    {feat_mode}: swap_only={fmt(bd['identity_swap_only'])} overall={fmt(bd['identity_overall'])}")

    save_csv(oracle_results, "feature_oracle_check.csv",
             ["pool_obs", "pool_fut", "identity_swap_only", "identity_overall", "assignment_accuracy"])

    # =========================================================================
    # Experiment 2: Temporal Pooling Ablation (trained FeatureOnly model)
    # =========================================================================
    print("\n=== Experiment 2: Temporal Pooling Ablation ===")

    pooling_results = []

    pooling_strategies = [
        ("mean", "mean"),
        ("first", "mean"),
        ("last", "mean"),
        ("mean", "first"),
        ("mean", "last"),
        ("first", "first"),
        ("last", "first"),
        ("last", "last"),
        ("first", "last"),
    ]

    model_fo = FeatureOnlyAssignmentHead(identity_weight=1.0)
    log = train_model(
        model_fo, train_data, val_data=eval_ds["clean_test_id"],
        epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
        uses_features=True, uses_future_features=True, verbose=False,
    )

    print("  FeatureOnly model trained. Testing pooling strategies:")

    for pool_obs, pool_fut in pooling_strategies:
        obs_pooled = pool_features(obs_feat, pool_obs)
        fut_pooled = pool_features(fut_feat, pool_fut)

        obs_t = torch.FloatTensor(obs_pooled)
        fut_t = torch.FloatTensor(fut_pooled)

        model_fo.eval()
        with torch.no_grad():
            z_obs = model_fo.feature_encoder(obs_t)
            z_fut = model_fo.feature_encoder(fut_t)
            z_obs_norm = F.normalize(z_obs, dim=-1)
            z_fut_norm = F.normalize(z_fut, dim=-1)
            sim_matrix = torch.bmm(z_fut_norm, z_obs_norm.transpose(1, 2))
            assignment_logits = sim_matrix / model_fo.temperature
            pred_identity = assignment_logits.argmax(dim=-1).numpy()

        bd, asgn_acc = compute_identity_metrics(pred_identity, true_identity)

        pooling_results.append({
            "pool_obs": pool_obs,
            "pool_fut": pool_fut,
            "identity_swap_only": fmt(bd["identity_swap_only"]),
            "identity_overall": fmt(bd["identity_overall"]),
            "assignment_accuracy": fmt(asgn_acc),
        })
        print(f"    obs={pool_obs} fut={pool_fut}: swap_only={fmt(bd['identity_swap_only'])}")

    print("\n  Pooling ablation with feature modes (trained encoder):")
    for feat_mode in ["normal", "shuffled", "zero", "random_wrong"]:
        obs_mod, fut_mod = apply_feature_mode_pair(obs_feat, fut_feat, feat_mode, seed=SEED)

        for pool_obs, pool_fut in [("mean", "mean"), ("last", "first"), ("last", "last")]:
            obs_pooled = pool_features(obs_mod, pool_obs)
            fut_pooled = pool_features(fut_mod, pool_fut)

            obs_t = torch.FloatTensor(obs_pooled)
            fut_t = torch.FloatTensor(fut_pooled)

            model_fo.eval()
            with torch.no_grad():
                z_obs = model_fo.feature_encoder(obs_t)
                z_fut = model_fo.feature_encoder(fut_t)
                z_obs_norm = F.normalize(z_obs, dim=-1)
                z_fut_norm = F.normalize(z_fut, dim=-1)
                sim_matrix = torch.bmm(z_fut_norm, z_obs_norm.transpose(1, 2))
                assignment_logits = sim_matrix / model_fo.temperature
                pred_identity = assignment_logits.argmax(dim=-1).numpy()

            bd, asgn_acc = compute_identity_metrics(pred_identity, true_identity)

            pooling_results.append({
                "pool_obs": f"{pool_obs}_{feat_mode}",
                "pool_fut": f"{pool_fut}_{feat_mode}",
                "identity_swap_only": fmt(bd["identity_swap_only"]),
                "identity_overall": fmt(bd["identity_overall"]),
                "assignment_accuracy": fmt(asgn_acc),
            })
            print(f"    {pool_obs}/{pool_fut} {feat_mode}: swap_only={fmt(bd['identity_swap_only'])}")

    save_csv(pooling_results, "temporal_pooling_ablation.csv",
             ["pool_obs", "pool_fut", "identity_swap_only", "identity_overall", "assignment_accuracy"])

    # =========================================================================
    # Experiment 3: Hybrid Inference Decomposition
    # =========================================================================
    print("\n=== Experiment 3: Hybrid Inference Decomposition ===")

    decomp_results = []

    for beta in [0.0, 1.0, 2.0, 5.0]:
        print(f"\n  Training Hybrid(beta={beta})...")
        model_hybrid = HybridTrajectoryFeatureAssignmentHead(identity_weight=1.0, beta=beta)
        log = train_model(
            model_hybrid, train_data, val_data=eval_ds["clean_test_id"],
            epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE,
            uses_features=True, uses_future_features=True, verbose=False,
        )

        for feat_mode in ["normal", "shuffled", "zero", "random_wrong"]:
            obs_mod, fut_mod = apply_feature_mode_pair(obs_feat, fut_feat, feat_mode, seed=SEED)

            obs_pos = swap_test["observed_positions"]

            model_hybrid.eval()
            with torch.no_grad():
                obs_pos_t = torch.FloatTensor(obs_pos)
                obs_feat_t = torch.FloatTensor(obs_mod) if obs_mod is not None else None
                fut_feat_t = torch.FloatTensor(fut_mod) if fut_mod is not None else None

                B = obs_pos_t.shape[0]

                if obs_feat_t is not None:
                    x = torch.cat([obs_pos_t, obs_feat_t], dim=-1)
                else:
                    padding = torch.zeros(B, model_hybrid.t_obs, model_hybrid.num_objects, model_hybrid.feature_dim)
                    x = torch.cat([obs_pos_t, padding], dim=-1)

                x = x.reshape(B, -1)
                shared_out = model_hybrid.shared(x)

                traj_logits = model_hybrid.traj_assignment_head(shared_out).reshape(B, model_hybrid.num_objects, model_hybrid.num_objects)

                feature_logits = None
                if obs_feat_t is not None and fut_feat_t is not None:
                    from models.feature_similarity_models import _pool_time, compute_feature_similarity_logits
                    feature_logits = compute_feature_similarity_logits(
                        model_hybrid.feature_encoder, obs_feat_t, fut_feat_t,
                        model_hybrid.num_objects, model_hybrid.temperature)

                # Trajectory-only prediction
                pred_traj = traj_logits.argmax(dim=-1).numpy()
                bd_traj, acc_traj = compute_identity_metrics(pred_traj, true_identity)

                # Feature-only prediction
                if feature_logits is not None:
                    pred_feat = feature_logits.argmax(dim=-1).numpy()
                    bd_feat, acc_feat = compute_identity_metrics(pred_feat, true_identity)
                else:
                    bd_feat = {"identity_swap_only": float("nan"), "identity_overall": float("nan")}
                    acc_feat = float("nan")

                # Hybrid prediction
                if feature_logits is not None:
                    hybrid_logits = traj_logits + model_hybrid.beta * feature_logits
                else:
                    hybrid_logits = traj_logits
                pred_hybrid = hybrid_logits.argmax(dim=-1).numpy()
                bd_hybrid, acc_hybrid = compute_identity_metrics(pred_hybrid, true_identity)

                decomp_results.append({
                    "beta": str(beta),
                    "feature_mode": feat_mode,
                    "traj_swap_only": fmt(bd_traj["identity_swap_only"]),
                    "traj_overall": fmt(bd_traj["identity_overall"]),
                    "feat_swap_only": fmt(bd_feat["identity_swap_only"]),
                    "feat_overall": fmt(bd_feat["identity_overall"]),
                    "hybrid_swap_only": fmt(bd_hybrid["identity_swap_only"]),
                    "hybrid_overall": fmt(bd_hybrid["identity_overall"]),
                })

                print(f"    {feat_mode}: traj={fmt(bd_traj['identity_swap_only'])} feat={fmt(bd_feat['identity_swap_only'])} hybrid={fmt(bd_hybrid['identity_swap_only'])}")

    # Compute feature_dependency for each pathway
    for beta in [0.0, 1.0, 2.0, 5.0]:
        normal_rows = [r for r in decomp_results
                       if r["beta"] == str(beta) and r["feature_mode"] == "normal"]
        shuffled_rows = [r for r in decomp_results
                         if r["beta"] == str(beta) and r["feature_mode"] == "shuffled"]

        if normal_rows and shuffled_rows:
            n_traj = float(normal_rows[0]["traj_swap_only"])
            s_traj = float(shuffled_rows[0]["traj_swap_only"])
            n_feat = float(normal_rows[0]["feat_swap_only"])
            s_feat = float(shuffled_rows[0]["feat_swap_only"])
            n_hybrid = float(normal_rows[0]["hybrid_swap_only"])
            s_hybrid = float(shuffled_rows[0]["hybrid_swap_only"])

            decomp_results.append({
                "beta": str(beta),
                "feature_mode": "dep_score",
                "traj_swap_only": fmt(n_traj - s_traj),
                "traj_overall": "",
                "feat_swap_only": fmt(n_feat - s_feat),
                "feat_overall": "",
                "hybrid_swap_only": fmt(n_hybrid - s_hybrid),
                "hybrid_overall": "",
            })

    save_csv(decomp_results, "hybrid_inference_decomposition.csv",
             ["beta", "feature_mode", "traj_swap_only", "traj_overall",
              "feat_swap_only", "feat_overall", "hybrid_swap_only", "hybrid_overall"])

    # =========================================================================
    # README
    # =========================================================================
    oracle_normal = [r for r in oracle_results if r["pool_obs"] == "mean" and r["pool_fut"] == "mean"]
    oracle_best = max(oracle_results, key=lambda r: float(r["identity_swap_only"]) if r["identity_swap_only"] != "nan" else 0)

    oracle_dep_rows = [r for r in oracle_results if "normal" in r.get("pool_obs", "") or "shuffled" in r.get("pool_obs", "")]
    oracle_normal_swap = 0.0
    oracle_shuffled_swap = 0.0
    for r in oracle_results:
        if r["pool_obs"] == "mean_normal" and r["pool_fut"] == "mean_normal":
            oracle_normal_swap = float(r["identity_swap_only"])
        if r["pool_obs"] == "mean_shuffled" and r["pool_fut"] == "mean_shuffled":
            oracle_shuffled_swap = float(r["identity_swap_only"])
    oracle_dep = oracle_normal_swap - oracle_shuffled_swap

    best_pool = max(
        [r for r in pooling_results if "_" not in r["pool_obs"]],
        key=lambda r: float(r["identity_swap_only"]) if r["identity_swap_only"] != "nan" else 0
    )

    decomp_dep_rows = [r for r in decomp_results if r["feature_mode"] == "dep_score"]
    best_traj_dep = max(float(r["traj_swap_only"]) for r in decomp_dep_rows) if decomp_dep_rows else 0.0
    best_feat_dep = max(float(r["feat_swap_only"]) for r in decomp_dep_rows) if decomp_dep_rows else 0.0

    oracle_ceiling = float(oracle_best["identity_swap_only"])

    if oracle_ceiling < 0.85:
        q1_answer = f"Feature-label/pooling problem. Oracle ceiling is only {fmt(oracle_ceiling)}, meaning raw features with mean pooling cannot achieve perfect assignment. The temporal structure of swap episodes (features flip at swap time) causes mean pooling to mix pre-swap and post-swap features, destroying discriminative information."
    elif oracle_ceiling >= 0.95:
        q1_answer = f"Model problem. Oracle achieves {fmt(oracle_ceiling)} with raw features, so the 75% ceiling is a model limitation, not a feature problem."
    else:
        q1_answer = f"Both. Oracle achieves {fmt(oracle_ceiling)}, better than trained model's 75%, but still not perfect. Temporal pooling is suboptimal AND the model doesn't fully learn the feature mapping."

    if best_traj_dep > 0.1:
        q2_answer = f"Hybrid's high score comes significantly from trajectory pathway (traj dep_score={fmt(best_traj_dep)}). The trajectory head learns better when trained with feature signal, creating indirect dependency."
    elif best_feat_dep > 0.2:
        q2_answer = f"Hybrid's high score comes primarily from feature pathway (feat dep_score={fmt(best_feat_dep)}). Feature similarity directly contributes to assignment."
    else:
        q2_answer = f"Neither pathway shows strong dependency. The hybrid score is a mix with no clear dominant pathway."

    if oracle_ceiling < 0.85:
        recommendation = "fix_feature_temporal_alignment"
    elif best_traj_dep > best_feat_dep:
        recommendation = "shortcut_still_dominates"
    elif best_feat_dep > 0.2:
        recommendation = "proceed_to_object_file_models"
    else:
        recommendation = "fix_feature_temporal_alignment"

    readme = f"""# SVT-v3.5.1: Feature Pathway Audit

## 1. Purpose

v3.5 showed FeatureOnly achieves 75% swap-only identity with feature_dependency=0.38. This audit explains WHY 75% is the ceiling and decomposes Hybrid's trajectory vs feature contributions.

## 2. Experiment 1: Feature Oracle Check

Raw one-hot features (no trained encoder) with cosine similarity assignment.

| Obs Pool | Fut Pool | Swap-Only | Overall |
|----------|----------|-----------|---------|
"""

    for r in oracle_results:
        if "_" not in r["pool_obs"]:
            readme += f"| {r['pool_obs']} | {r['pool_fut']} | {r['identity_swap_only']} | {r['identity_overall']} |\n"

    readme += f"""
Feature ablation (mean/mean pooling):

| Mode | Swap-Only | Overall |
|------|-----------|---------|
"""

    for r in oracle_results:
        if "_" in r["pool_obs"]:
            mode = r["pool_obs"].replace("mean_", "")
            readme += f"| {mode} | {r['identity_swap_only']} | {r['identity_overall']} |\n"

    readme += f"""
**Oracle ceiling: {fmt(oracle_ceiling)}**

## 3. Experiment 2: Temporal Pooling Ablation

Trained FeatureOnly encoder with different pooling strategies.

| Obs Pool | Fut Pool | Swap-Only |
|----------|----------|-----------|
"""

    for r in pooling_results:
        if "_" not in r["pool_obs"]:
            readme += f"| {r['pool_obs']} | {r['pool_fut']} | {r['identity_swap_only']} |\n"

    readme += f"""
Best pooling: obs={best_pool['pool_obs']}, fut={best_pool['pool_fut']}, swap_only={best_pool['identity_swap_only']}

## 4. Experiment 3: Hybrid Inference Decomposition

| Beta | Feature Mode | Traj Swap | Feat Swap | Hybrid Swap |
|------|-------------|-----------|-----------|-------------|
"""

    for r in decomp_results:
        if r["feature_mode"] != "dep_score":
            readme += f"| {r['beta']} | {r['feature_mode']} | {r['traj_swap_only']} | {r['feat_swap_only']} | {r['hybrid_swap_only']} |\n"

    readme += f"""
Dependency scores (normal - shuffled):

| Beta | Traj Dep | Feat Dep | Hybrid Dep |
|------|----------|----------|------------|
"""

    for r in decomp_dep_rows:
        readme += f"| {r['beta']} | {r['traj_swap_only']} | {r['feat_swap_only']} | {r['hybrid_swap_only']} |\n"

    readme += f"""
## 5. Answers

### Q1: Is FeatureOnly 75% a model problem or feature-label/pooling problem?

{q1_answer}

### Q2: Does Hybrid's high score come from trajectory or feature pathway?

{q2_answer}

### Q3: Next step

**{recommendation}**
"""

    with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print("\n" + "=" * 60)
    print("SVT-v3.5.1 FINAL SUMMARY")
    print("=" * 60)
    print(f"Oracle ceiling (best pooling): {fmt(oracle_ceiling)}")
    print(f"Oracle feature dep (normal-shuffled): {fmt(oracle_dep)}")
    print(f"Best pooling: obs={best_pool['pool_obs']}, fut={best_pool['pool_fut']}")
    print(f"Best traj dep_score: {fmt(best_traj_dep)}")
    print(f"Best feat dep_score: {fmt(best_feat_dep)}")
    print(f"Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
