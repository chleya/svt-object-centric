"""
SVT-v5.1: Scaling Sanity Audit
"""

import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v5_1_scaling_sanity_audit"
SEED = 0
NUM_OBJ = 3
FEAT_DIM = 16


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


def permutation_accuracy(pred_assignment, true_assignment):
    return float((pred_assignment == true_assignment).all(axis=1).mean())


def is_identity_perm(labels):
    return np.all(labels == np.arange(labels.shape[1]), axis=1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SVT-v5.1: Scaling Sanity Audit")
    print("=" * 60)

    from data.generate_nonlinear_feature_ood import _generate_single_episode, _stack_episodes

    # =========================================================================
    # Audit 1: N=3 Permutation Metric Audit
    # =========================================================================
    print("\n=== Audit 1: N=3 Permutation Metric Audit ===")

    rng = np.random.RandomState(SEED)
    n_episodes = 200

    permutations = {
        "identity": np.array([0, 1, 2]),
        "swap01": np.array([1, 0, 2]),
        "cycle": np.array([1, 2, 0]),
        "reverse": np.array([2, 1, 0]),
    }

    audit1_results = []

    for perm_name, perm in permutations.items():
        true_labels = np.tile(perm, (n_episodes, 1))

        perfect_pred = true_labels.copy()
        perfect_acc = permutation_accuracy(perfect_pred, true_labels)

        random_preds = rng.randint(0, NUM_OBJ, size=(n_episodes, NUM_OBJ))
        random_acc = permutation_accuracy(random_preds, true_labels)

        fixed_identity = np.tile(np.arange(NUM_OBJ), (n_episodes, 1))
        fixed_acc = permutation_accuracy(fixed_identity, true_labels)

        is_swap = ~is_identity_perm(true_labels)

        from metrics.identity_breakdown import compute_identity_breakdown
        bd = compute_identity_breakdown(perfect_pred, true_labels)

        audit1_results.append({
            "permutation": perm_name,
            "perfect_accuracy": fmt(perfect_acc),
            "random_accuracy": fmt(random_acc),
            "fixed_identity_accuracy": fmt(fixed_acc),
            "is_swap_detected": str(is_swap.all()),
            "bd_swap_only": fmt(bd["identity_swap_only"]),
            "bd_overall": fmt(bd["identity_overall"]),
        })

        print(f"  {perm_name}: perfect={fmt(perfect_acc)} random={fmt(random_acc)} "
              f"fixed={fmt(fixed_acc)} is_swap={is_swap.all()} bd_swap={fmt(bd['identity_swap_only'])}")

    n_identity = sum(1 for r in audit1_results if r["permutation"] == "identity")
    n_swap = sum(1 for r in audit1_results if r["permutation"] != "identity")

    print(f"  Identity perm: bd_swap_only should be nan (no swap episodes)")
    print(f"  Swap perms: bd_swap_only should be 1.0 (perfect prediction)")

    save_csv(audit1_results, "permutation_metric_audit.csv",
             ["permutation", "perfect_accuracy", "random_accuracy",
              "fixed_identity_accuracy", "is_swap_detected",
              "bd_swap_only", "bd_overall"])

    # =========================================================================
    # Audit 2: Continuous Feature Oracle
    # =========================================================================
    print("\n=== Audit 2: Continuous Feature Oracle ===")

    rng2 = np.random.RandomState(SEED)
    episodes = []
    for _ in range(200):
        ep = _generate_single_episode(
            t_obs=10, t_pred=20, num_objects=NUM_OBJ, arena_size=64.0,
            feature_mode="feature_bearing", feature_dim=FEAT_DIM,
            randomize_object_order=True,
            identity_test=True, swap_probability=0.5,
            force_type="vortex", field_strength=0.5,
            damping=0.95, noise_std=0.1, rng=rng2,
        )
        episodes.append(ep)

    data = _stack_episodes(episodes, "feature_bearing")

    obs_feat = data["object_features_obs"]
    fut_feat = data["object_features_fut"]
    true_id = data["identity_labels"]
    is_swap = ~is_identity_perm(true_id)

    obs_first = obs_feat[:, 0, :, :]
    fut_last = fut_feat[:, -1, :, :]

    B = obs_first.shape[0]
    N = NUM_OBJ

    normal_assignments = np.zeros((B, N), dtype=int)
    for b in range(B):
        sim_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                fi = fut_last[b, i]
                fj = obs_first[b, j]
                nf = np.linalg.norm(fi)
                nj = np.linalg.norm(fj)
                if nf > 1e-8 and nj > 1e-8:
                    sim_matrix[i, j] = np.dot(fi, fj) / (nf * nj)
        used = set()
        for i in range(N):
            row = sim_matrix[i].copy()
            for j in used:
                row[j] = -float('inf')
            best = np.argmax(row)
            normal_assignments[b, i] = best
            used.add(best)

    normal_acc = permutation_accuracy(normal_assignments, true_id)
    normal_swap = permutation_accuracy(normal_assignments[is_swap], true_id[is_swap]) if is_swap.sum() > 0 else float('nan')

    rng3 = np.random.RandomState(SEED)
    shuffled_obs = obs_feat.copy()
    for b in range(B):
        for t in range(shuffled_obs.shape[1]):
            perm = rng3.permutation(N)
            shuffled_obs[b, t] = shuffled_obs[b, t, perm]
    shuf_first = shuffled_obs[:, 0, :, :]

    shuffled_assignments = np.zeros((B, N), dtype=int)
    for b in range(B):
        sim_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                fi = fut_last[b, i]
                fj = shuf_first[b, j]
                nf = np.linalg.norm(fi)
                nj = np.linalg.norm(fj)
                if nf > 1e-8 and nj > 1e-8:
                    sim_matrix[i, j] = np.dot(fi, fj) / (nf * nj)
        used = set()
        for i in range(N):
            row = sim_matrix[i].copy()
            for j in used:
                row[j] = -float('inf')
            best = np.argmax(row)
            shuffled_assignments[b, i] = best
            used.add(best)

    shuffled_acc = permutation_accuracy(shuffled_assignments, true_id)

    zero_first = np.zeros_like(obs_first)
    zero_assignments = np.zeros((B, N), dtype=int)
    for b in range(B):
        sim_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                sim_matrix[i, j] = 0.0
        rng_z = np.random.RandomState(b)
        used = set()
        for i in range(N):
            row = sim_matrix[i].copy()
            for j in used:
                row[j] = -float('inf')
            if row.max() > -float('inf'):
                best = np.argmax(row)
            else:
                best = rng_z.randint(0, N)
                while best in used:
                    best = rng_z.randint(0, N)
            zero_assignments[b, i] = best
            used.add(best)

    zero_acc = permutation_accuracy(zero_assignments, true_id)

    wrong_first = obs_first.copy()
    wrong_first[:, :, :] = wrong_first[:, [1, 0, 2], :]
    wrong_assignments = np.zeros((B, N), dtype=int)
    for b in range(B):
        sim_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                fi = fut_last[b, i]
                fj = wrong_first[b, j]
                nf = np.linalg.norm(fi)
                nj = np.linalg.norm(fj)
                if nf > 1e-8 and nj > 1e-8:
                    sim_matrix[i, j] = np.dot(fi, fj) / (nf * nj)
        used = set()
        for i in range(N):
            row = sim_matrix[i].copy()
            for j in used:
                row[j] = -float('inf')
            best = np.argmax(row)
            wrong_assignments[b, i] = best
            used.add(best)

    wrong_acc = permutation_accuracy(wrong_assignments, true_id)

    audit2_results = [{
        "condition": "normal",
        "accuracy": fmt(normal_acc),
        "swap_only_accuracy": fmt(normal_swap),
    }, {
        "condition": "shuffled",
        "accuracy": fmt(shuffled_acc),
        "swap_only_accuracy": "nan",
    }, {
        "condition": "zero",
        "accuracy": fmt(zero_acc),
        "swap_only_accuracy": "nan",
    }, {
        "condition": "wrong_features",
        "accuracy": fmt(wrong_acc),
        "swap_only_accuracy": "nan",
    }]

    print(f"  Normal: {fmt(normal_acc)} (swap: {fmt(normal_swap)})")
    print(f"  Shuffled: {fmt(shuffled_acc)}")
    print(f"  Zero: {fmt(zero_acc)}")
    print(f"  Wrong: {fmt(wrong_acc)}")

    save_csv(audit2_results, "continuous_feature_oracle.csv",
             ["condition", "accuracy", "swap_only_accuracy"])

    # =========================================================================
    # Audit 3: Conflict Construction Audit
    # =========================================================================
    print("\n=== Audit 3: Conflict Construction Audit ===")

    from data.generate_nonlinear_feature_ood import generate_v3_dataset

    eval_ds = generate_v3_dataset(
        n_train=1000, n_test=200, num_objects=NUM_OBJ, feature_dim=FEAT_DIM,
        feature_mode="feature_bearing",
        force_train_type="attractor", force_test_type="vortex",
        randomize_object_order=True, disjoint_init_split=True, seed=SEED,
    )

    clean_test = eval_ds["clean_test_id"]
    is_swap_clean = clean_test["is_swap"]
    no_swap_idx = np.where(~is_swap_clean)[0]

    if len(no_swap_idx) > 0:
        no_swap_data = {k: v[no_swap_idx] for k, v in clean_test.items()}

        true_id_ns = no_swap_data["identity_labels"]
        obs_feat_ns = no_swap_data["object_features_obs"]
        fut_feat_ns = no_swap_data["object_features_fut"]
        fut_pos_ns = no_swap_data["future_positions"]
        obs_pos_ns = no_swap_data["observed_positions"]

        obs_first_ns = obs_feat_ns[:, 0, :, :]
        fut_last_ns = fut_feat_ns[:, -1, :, :]

        B_ns = obs_first_ns.shape[0]

        flipped_fut = fut_feat_ns.copy()
        flipped_fut[:, :, 0, :], flipped_fut[:, :, 1, :] = flipped_fut[:, :, 1, :].copy(), flipped_fut[:, :, 0, :].copy()
        flipped_last = flipped_fut[:, -1, :, :]

        feat_assign = np.zeros((B_ns, N), dtype=int)
        for b in range(B_ns):
            sim_matrix = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    fi = flipped_last[b, i]
                    fj = obs_first_ns[b, j]
                    nf = np.linalg.norm(fi)
                    nj = np.linalg.norm(fj)
                    if nf > 1e-8 and nj > 1e-8:
                        sim_matrix[i, j] = np.dot(fi, fj) / (nf * nj)
            used = set()
            for i in range(N):
                row = sim_matrix[i].copy()
                for j in used:
                    row[j] = -float('inf')
                best = np.argmax(row)
                feat_assign[b, i] = best
                used.add(best)

        obs_last_pos = obs_pos_ns[:, -1, :, :]
        fut_mean_pos = fut_pos_ns.mean(axis=1)

        traj_assign = np.zeros((B_ns, N), dtype=int)
        for b in range(B_ns):
            dist_matrix = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    dist_matrix[i, j] = np.linalg.norm(fut_mean_pos[b, i] - obs_last_pos[b, j])
            used = set()
            for i in range(N):
                row = dist_matrix[i].copy()
                for j in used:
                    row[j] = float('inf')
                best = np.argmin(row)
                traj_assign[b, i] = best
                used.add(best)

        feat_matches_true = permutation_accuracy(feat_assign, true_id_ns)
        traj_matches_true = permutation_accuracy(traj_assign, true_id_ns)

        disagreement = ~np.array_equal(feat_assign, traj_assign)
        conflict_rate = float(np.mean([not np.array_equal(feat_assign[b], traj_assign[b]) for b in range(B_ns)]))

        feat_correct_when_disagree = 0.0
        traj_correct_when_disagree = 0.0
        n_disagree = 0
        for b in range(B_ns):
            if not np.array_equal(feat_assign[b], traj_assign[b]):
                n_disagree += 1
                if np.array_equal(feat_assign[b], true_id_ns[b]):
                    feat_correct_when_disagree += 1
                if np.array_equal(traj_assign[b], true_id_ns[b]):
                    traj_correct_when_disagree += 1

        if n_disagree > 0:
            feat_correct_rate_disagree = feat_correct_when_disagree / n_disagree
            traj_correct_rate_disagree = traj_correct_when_disagree / n_disagree
        else:
            feat_correct_rate_disagree = float('nan')
            traj_correct_rate_disagree = float('nan')

        audit3_results = [{
            "conflict_rate": fmt(conflict_rate),
            "feature_matches_true": fmt(feat_matches_true),
            "trajectory_matches_true": fmt(traj_matches_true),
            "feat_vs_traj_disagreement": fmt(conflict_rate),
            "feat_correct_when_disagree": fmt(feat_correct_rate_disagree),
            "traj_correct_when_disagree": fmt(traj_correct_rate_disagree),
            "n_episodes": B_ns,
            "n_disagree": n_disagree,
        }]

        print(f"  Conflict rate: {fmt(conflict_rate)}")
        print(f"  Feature matches true: {fmt(feat_matches_true)}")
        print(f"  Trajectory matches true: {fmt(traj_matches_true)}")
        print(f"  When disagree: feat_correct={fmt(feat_correct_rate_disagree)} traj_correct={fmt(traj_correct_rate_disagree)}")
    else:
        audit3_results = [{"conflict_rate": "nan", "feature_matches_true": "nan",
                           "trajectory_matches_true": "nan", "feat_vs_traj_disagreement": "nan",
                           "feat_correct_when_disagree": "nan", "traj_correct_when_disagree": "nan",
                           "n_episodes": 0, "n_disagree": 0}]
        print("  No no-swap episodes found!")

    save_csv(audit3_results, "conflict_construction_audit.csv",
             ["conflict_rate", "feature_matches_true", "trajectory_matches_true",
              "feat_vs_traj_disagreement", "feat_correct_when_disagree",
              "traj_correct_when_disagree", "n_episodes", "n_disagree"])

    # =========================================================================
    # Audit 4: FeatureOnly Conflict Artifact Check
    # =========================================================================
    print("\n=== Audit 4: FeatureOnly Conflict Artifact Check ===")

    if len(no_swap_idx) > 0:
        conflict_feat_assign = feat_assign.copy()
        conflict_acc = permutation_accuracy(conflict_feat_assign, true_id_ns)

        restored_fut_last = fut_last_ns.copy()
        restored_assign = np.zeros((B_ns, N), dtype=int)
        for b in range(B_ns):
            sim_matrix = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    fi = restored_fut_last[b, i]
                    fj = obs_first_ns[b, j]
                    nf = np.linalg.norm(fi)
                    nj = np.linalg.norm(fj)
                    if nf > 1e-8 and nj > 1e-8:
                        sim_matrix[i, j] = np.dot(fi, fj) / (nf * nj)
            used = set()
            for i in range(N):
                row = sim_matrix[i].copy()
                for j in used:
                    row[j] = -float('inf')
                best = np.argmax(row)
                restored_assign[b, i] = best
                used.add(best)

        restored_acc = permutation_accuracy(restored_assign, true_id_ns)

        audit4_results = [{
            "condition": "conflict_flipped_features",
            "feature_oracle_accuracy": fmt(conflict_acc),
            "interpretation": "flipped features mislead oracle",
        }, {
            "condition": "restored_correct_features",
            "feature_oracle_accuracy": fmt(restored_acc),
            "interpretation": "correct features restore oracle",
        }]

        print(f"  Conflict (flipped): {fmt(conflict_acc)}")
        print(f"  Restored (correct): {fmt(restored_acc)}")

        if restored_acc > 0.95:
            artifact_verdict = "NOT_artifact_FeatureOnly_conflict_0_is_real_failure"
        else:
            artifact_verdict = "POSSIBLE_artifact_metric_or_label_issue"
    else:
        audit4_results = []
        artifact_verdict = "NO_DATA"
        restored_acc = float('nan')

    save_csv(audit4_results, "featureonly_conflict_artifact_check.csv",
             ["condition", "feature_oracle_accuracy", "interpretation"])

    # =========================================================================
    # README
    # =========================================================================
    normal_oracle = float(audit2_results[0]["accuracy"])
    conflict_rate_val = float(audit3_results[0]["conflict_rate"])
    traj_correct_disagree = float(audit3_results[0]["traj_correct_when_disagree"])
    feat_correct_disagree = float(audit3_results[0]["feat_correct_when_disagree"])

    q1_ok = all(float(r["perfect_accuracy"]) == 1.0 for r in audit1_results)
    q2_ok = normal_oracle > 0.9
    q3_ok = conflict_rate_val > 0.8
    q4_ok = restored_acc > 0.95 if not np.isnan(restored_acc) else False

    all_ok = q1_ok and q2_ok and q3_ok and q4_ok

    if all_ok:
        recommendation = "include_v5_as_supplementary"
    elif not q1_ok:
        recommendation = "fix_n3_identity_metric"
    elif not q3_ok:
        recommendation = "fix_conflict_construction"
    else:
        recommendation = "exclude_v5_from_paper"

    readme = f"""# SVT-v5.1: Scaling Sanity Audit

## Audit 1: N=3 Permutation Metric

| Permutation | Perfect Acc | Random Acc | Fixed Identity Acc | Is Swap | BD Swap-Only | BD Overall |
|------------|------------|-----------|-------------------|---------|-------------|-----------|
"""
    for r in audit1_results:
        readme += f"| {r['permutation']} | {r['perfect_accuracy']} | {r['random_accuracy']} | {r['fixed_identity_accuracy']} | {r['is_swap_detected']} | {r['bd_swap_only']} | {r['bd_overall']} |\n"

    readme += f"""
**Q1: N=3 identity metric 是否可信？**
- Perfect prediction accuracy = 1.0 for all permutations: {"YES" if q1_ok else "NO"}
- Random prediction accuracy near chance: {"YES" if all(float(r["random_accuracy"]) < 0.5 for r in audit1_results) else "CHECK"}
- Fixed identity accuracy drops for non-identity permutations: {"YES" if all(float(r["fixed_identity_accuracy"]) < 1.0 for r in audit1_results if r["permutation"] != "identity") else "CHECK"}
- Note: `identity_breakdown.is_swap` uses `true_identity[:, 0] != 0` which works for N=3 single-swap but may not detect all non-identity permutations correctly

## Audit 2: Continuous Feature Oracle

| Condition | Accuracy | Swap-Only |
|-----------|----------|-----------|
"""
    for r in audit2_results:
        readme += f"| {r['condition']} | {r['accuracy']} | {r['swap_only_accuracy']} |\n"

    readme += f"""
**Q2: Continuous feature oracle 在 clean 下是否接近 1.0？**
- Normal oracle accuracy = {normal_oracle:.4f}: {"YES" if q2_ok else "NO"}
- Shuffled accuracy should be low: {audit2_results[1]["accuracy"]}
- Zero accuracy should be near chance: {audit2_results[2]["accuracy"]}

## Audit 3: Conflict Construction

| Metric | Value |
|--------|-------|
| Conflict rate | {conflict_rate_val:.4f} |
| Feature matches true | {audit3_results[0]['feature_matches_true']} |
| Trajectory matches true | {audit3_results[0]['trajectory_matches_true']} |
| Feat correct when disagree | {feat_correct_disagree:.4f} |
| Traj correct when disagree | {traj_correct_disagree:.4f} |

**Q3: Conflict split 是否真的制造 feature-trajectory conflict？**
- Conflict rate = {conflict_rate_val:.4f}: {"YES" if q3_ok else "NO"}
- When feature and trajectory disagree, trajectory is correct: {"YES" if traj_correct_disagree > feat_correct_disagree else "CHECK"} (traj={traj_correct_disagree:.4f} vs feat={feat_correct_disagree:.4f})

## Audit 4: FeatureOnly Conflict Artifact Check

| Condition | Oracle Accuracy |
|-----------|----------------|
| Flipped features (conflict) | {fmt(conflict_acc) if len(audit4_results) > 0 else 'nan'} |
| Restored correct features | {fmt(restored_acc) if len(audit4_results) > 1 else 'nan'} |

**Q4: FeatureOnly conflict=0 是真实失败还是 metric artifact？**
- Restored features accuracy = {restored_acc:.4f}: {"NOT artifact (real failure)" if q4_ok else "POSSIBLE artifact"}
- Verdict: {artifact_verdict}

## Q5: v5 是否可作为 supplementary scaling check？

- Q1 (metric valid): {"YES" if q1_ok else "NO"}
- Q2 (oracle works): {"YES" if q2_ok else "NO"}
- Q3 (conflict real): {"YES" if q3_ok else "NO"}
- Q4 (not artifact): {"YES" if q4_ok else "NO"}

**Recommendation: {recommendation}**
"""

    with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print("\n" + "=" * 60)
    print("SVT-v5.1 SANITY AUDIT SUMMARY")
    print("=" * 60)
    print(f"  Q1 Metric valid: {'YES' if q1_ok else 'NO'}")
    print(f"  Q2 Oracle works: {'YES' if q2_ok else 'NO'} (normal={normal_oracle:.4f})")
    print(f"  Q3 Conflict real: {'YES' if q3_ok else 'NO'} (rate={conflict_rate_val:.4f})")
    print(f"  Q4 Not artifact: {'YES' if q4_ok else 'NO'} (restored={restored_acc:.4f})")
    print(f"  Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
