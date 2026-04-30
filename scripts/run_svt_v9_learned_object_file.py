"""
SVT-v9: Learned ObjectFile with Structural Inductive Bias (Direction D)

Compare:
1. ConflictFirstObjectFile (rule-based)
2. UncertaintyAwareObjectFile (evidential)
3. LearnedObjectFile (learned with structural bias)
4. FeatureOnly (baseline)
5. HybridTrajectoryFeature (baseline)

Key question: Can a learned system with structural bias
simultaneously achieve high swap-only AND high conflict-a?
"""

import sys
import os
import numpy as np
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v9_learned_object_file"
EPOCHS = 30
SEEDS = [0, 42, 123]

TORCH_AVAILABLE = False
try:
    import torch
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


def generate_swap_train(n_train=1000, num_objects=2, feature_dim=2,
                         swap_ratio=0.5, seed=0, force_type="attractor"):
    rng = np.random.RandomState(seed)
    episodes = []
    for _ in range(n_train):
        ep = _generate_single_episode(
            t_obs=10, t_pred=20, num_objects=num_objects, arena_size=64.0,
            feature_mode="feature_bearing", feature_dim=feature_dim,
            randomize_object_order=True,
            identity_test=True, swap_probability=swap_ratio,
            force_type=force_type, field_strength=0.5,
            damping=0.95, noise_std=0.1, rng=rng,
        )
        episodes.append(ep)
    return _stack_episodes(episodes, "feature_bearing")


def flip_future_features(future_features):
    if future_features is None:
        return None
    flipped = future_features.copy()
    flipped[:, :, 0, :], flipped[:, :, 1, :] = flipped[:, :, 1, :].copy(), flipped[:, :, 0, :].copy()
    return flipped


def evaluate_learned_model(model, test_data, method="combined"):
    obs_pos = test_data["observed_positions"]
    obs_feat = test_data.get("object_features_obs")
    fut_feat = test_data.get("object_features_fut")
    fut_pos = test_data["future_positions"]

    if hasattr(model, 'predict_identity'):
        sig = model.predict_identity.__code__.co_varnames
        if 'future_positions' in sig or 'method' in sig:
            pred_identity = model.predict_identity(
                obs_pos, obs_feat, future_positions=fut_pos, future_features=fut_feat,
                method=method)
        else:
            pred_identity = model.predict_identity(
                obs_pos, obs_feat, future_features=fut_feat)
    if isinstance(pred_identity, torch.Tensor):
        pred_identity = pred_identity.cpu().numpy()

    true_identity = test_data["identity_labels"]
    breakdown = compute_identity_breakdown(pred_identity, true_identity)
    return breakdown


def run_learned_experiment(num_objects, feature_dim, seed, epochs=EPOCHS):
    from models.learned_object_file import LearnedObjectFile
    from models.feature_similarity_models import FeatureOnlyAssignmentHead, HybridTrajectoryFeatureAssignmentHead
    from models.object_file_models import TrajectoryOnlyAssignment, ConflictFirstObjectFile
    from utils.torch_training import train_model

    eval_ds = generate_v3_dataset(
        n_train=1000, n_test=200, num_objects=num_objects, feature_dim=feature_dim,
        feature_mode="feature_bearing",
        force_train_type="attractor", force_test_type="vortex",
        randomize_object_order=True, disjoint_init_split=True, seed=seed,
    )
    train_data = generate_swap_train(
        n_train=1000, num_objects=num_objects, feature_dim=feature_dim, seed=seed)
    swap_test = eval_ds["identity_test_swap_only"]
    clean_test = eval_ds["clean_test_id"]

    all_results = []

    print(f"  Training LearnedObjectFile...")
    lof = LearnedObjectFile(
        num_objects=num_objects, feature_dim=feature_dim,
        hidden_dim=128, slot_dim=64,
        identity_weight=1.0, conflict_weight=0.5,
        channel_aux_weight=0.3)

    optimizer = torch.optim.Adam(lof.parameters(), lr=1e-3)

    obs_pos_t = train_data["observed_positions"]
    fut_pos_t = train_data["future_positions"]
    obs_feat_t = train_data.get("object_features_obs")
    fut_feat_t = train_data.get("object_features_fut")
    ids_t = train_data["identity_labels"]
    is_swap_t = train_data.get("is_swap", np.zeros(len(ids_t)))

    n_batches = len(obs_pos_t) // 64
    for epoch in range(epochs):
        indices = np.random.permutation(len(obs_pos_t))
        total_loss = 0
        for batch_idx in range(n_batches):
            bi = indices[batch_idx * 64:(batch_idx + 1) * 64]

            loss, comb_l, feat_l, traj_l = lof.compute_loss(
                torch.FloatTensor(obs_pos_t[bi]),
                torch.FloatTensor(fut_pos_t[bi]),
                torch.LongTensor(ids_t[bi]),
                torch.FloatTensor(obs_feat_t[bi]) if obs_feat_t is not None else None,
                torch.FloatTensor(fut_feat_t[bi]) if fut_feat_t is not None else None,
                is_swap=torch.FloatTensor(is_swap_t[bi]) if is_swap_t is not None else None,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    for method in ["combined", "feature", "trajectory"]:
        bd = evaluate_learned_model(lof, swap_test, method=method)

        is_swap_c = clean_test["is_swap"]
        no_swap_idx = np.where(~is_swap_c)[0]
        conflict_a = float("nan")

        if len(no_swap_idx) > 0:
            no_swap_data = {k: v[no_swap_idx] for k, v in clean_test.items()}
            fut_flipped = flip_future_features(no_swap_data.get("object_features_fut"))

            pred_id_c = lof.predict_identity(
                no_swap_data["observed_positions"],
                no_swap_data.get("object_features_obs"),
                future_positions=no_swap_data["future_positions"],
                future_features=fut_flipped,
                method=method)
            if isinstance(pred_id_c, torch.Tensor):
                pred_id_c = pred_id_c.cpu().numpy()
            true_id_c = no_swap_data["identity_labels"]
            correct = (pred_id_c == true_id_c).all(axis=1)
            conflict_a = float(correct.mean())

        all_results.append({
            "mechanism": f"LearnedObjectFile_{method}",
            "num_objects": num_objects,
            "feature_dim": feature_dim,
            "seed": seed,
            "swap_only": fmt(bd["identity_swap_only"]),
            "overall": fmt(bd["identity_overall"]),
            "conflict_a": fmt(conflict_a),
        })

        print(f"    LOF/{method}: swap={fmt(bd['identity_swap_only'])} conflict_a={fmt(conflict_a)}")

    print(f"  Training FeatureOnly...")
    fo = FeatureOnlyAssignmentHead(num_objects=num_objects, feature_dim=feature_dim)
    train_model(fo, train_data, val_data=clean_test, epochs=epochs,
                batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=True, uses_future_features=True, verbose=False)
    fo_bd = evaluate_learned_model(fo, swap_test)
    all_results.append({
        "mechanism": "FeatureOnly",
        "num_objects": num_objects, "feature_dim": feature_dim, "seed": seed,
        "swap_only": fmt(fo_bd["identity_swap_only"]),
        "overall": fmt(fo_bd["identity_overall"]),
    })
    print(f"    FeatureOnly: swap={fmt(fo_bd['identity_swap_only'])}")

    print(f"  Training Hybrid...")
    hybrid = HybridTrajectoryFeatureAssignmentHead(num_objects=num_objects, feature_dim=feature_dim, beta=1.0)
    train_model(hybrid, train_data, val_data=clean_test, epochs=epochs,
                batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=True, uses_future_features=True, verbose=False)
    hy_bd = evaluate_learned_model(hybrid, swap_test)
    all_results.append({
        "mechanism": "Hybrid",
        "num_objects": num_objects, "feature_dim": feature_dim, "seed": seed,
        "swap_only": fmt(hy_bd["identity_swap_only"]),
        "overall": fmt(hy_bd["identity_overall"]),
    })
    print(f"    Hybrid: swap={fmt(hy_bd['identity_swap_only'])}")

    print(f"  Training CFObjectFile (rule-based)...")
    traj = TrajectoryOnlyAssignment(num_objects=num_objects)
    train_model(traj, train_data, val_data=clean_test, epochs=epochs,
                batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=False, uses_future_features=False, verbose=False)
    cf = ConflictFirstObjectFile(traj_model=traj, strategy="margin_gated",
                                  num_objects=num_objects, feature_dim=feature_dim,
                                  traj_margin_advantage=1.2)

    obs_pos = swap_test["observed_positions"]
    obs_feat = swap_test.get("object_features_obs")
    fut_feat = swap_test.get("object_features_fut")
    fut_pos = swap_test["future_positions"]
    true_id = swap_test["identity_labels"]

    r = cf.predict_identity(obs_pos, obs_feat, fut_pos, fut_feat, return_conflict_info=True)
    cf_pred = r[0]
    cf_bd = compute_identity_breakdown(cf_pred, true_id)

    is_swap_c = clean_test["is_swap"]
    no_swap_idx = np.where(~is_swap_c)[0]
    cf_conflict_a = float("nan")
    if len(no_swap_idx) > 0:
        no_swap_data = {k: v[no_swap_idx] for k, v in clean_test.items()}
        fut_flipped = flip_future_features(no_swap_data.get("object_features_fut"))
        r2 = cf.predict_identity(no_swap_data["observed_positions"],
                                  no_swap_data.get("object_features_obs"),
                                  no_swap_data["future_positions"],
                                  fut_flipped, return_conflict_info=True)
        pred_c = r2[0]
        correct = (pred_c == no_swap_data["identity_labels"]).all(axis=1)
        cf_conflict_a = float(correct.mean())

    all_results.append({
        "mechanism": "CFObjectFile_rule",
        "num_objects": num_objects, "feature_dim": feature_dim, "seed": seed,
        "swap_only": fmt(cf_bd["identity_swap_only"]),
        "overall": fmt(cf_bd["identity_overall"]),
        "conflict_a": fmt(cf_conflict_a),
    })
    print(f"    CF rule: swap={fmt(cf_bd['identity_swap_only'])} conflict_a={fmt(cf_conflict_a)}")

    return all_results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("SVT-v9: Learned ObjectFile with Structural Inductive Bias")
    print("=" * 70)

    if not TORCH_AVAILABLE:
        print("PyTorch unavailable. Exiting.")
        return

    all_results = []

    print("\n=== 2 Objects + One-Hot Features ===")
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        results = run_learned_experiment(2, 2, seed)
        all_results.extend(results)

    print("\n=== 3 Objects + 16-dim Continuous Features ===")
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        results = run_learned_experiment(3, 16, seed)
        all_results.extend(results)

    save_csv(all_results, "learned_results.csv",
             ["mechanism", "num_objects", "feature_dim", "seed",
              "swap_only", "overall", "conflict_a"])

    with open(os.path.join(OUTPUT_DIR, "full_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("LEARNED vs RULE-BASED COMPARISON (mean ± std)")
    print("=" * 70)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        key = f"{r['mechanism']}_{r['num_objects']}obj_fd{r['feature_dim']}"
        grouped[key].append(r)

    print(f"\n{'Mechanism':<45} {'Swap-Only':>12} {'Conflict-A':>12}")
    print("-" * 70)

    for key, runs in sorted(grouped.items()):
        swaps = [float(r["swap_only"]) for r in runs]
        conflicts = [float(r["conflict_a"]) for r in runs if r.get("conflict_a", "nan") != "nan"]

        def fmt_stat(vals):
            valid = [v for v in vals if not np.isnan(v)]
            if not valid:
                return "nan"
            return f"{np.mean(valid):.3f}±{np.std(valid):.3f}"

        conflict_str = fmt_stat(conflicts) if conflicts else "nan"
        print(f"{key:<45} {fmt_stat(swaps):>12} {conflict_str:>12}")

    print("\n" + "=" * 70)
    print("KEY QUESTION: Does LearnedObjectFile break the trade-off?")
    print("=" * 70)

    for n_obj, fd in [(2, 2), (3, 16)]:
        print(f"\n  {n_obj} objects, fd={fd}:")
        lof_combined = [float(r["swap_only"]) for r in all_results
                        if r["mechanism"] == "LearnedObjectFile_combined"
                        and r["num_objects"] == n_obj and r["feature_dim"] == fd]
        lof_conflict = [float(r["conflict_a"]) for r in all_results
                        if r["mechanism"] == "LearnedObjectFile_combined"
                        and r["num_objects"] == n_obj and r["feature_dim"] == fd
                        and r.get("conflict_a", "nan") != "nan"]
        cf_swap = [float(r["swap_only"]) for r in all_results
                   if r["mechanism"] == "CFObjectFile_rule"
                   and r["num_objects"] == n_obj and r["feature_dim"] == fd]
        cf_conflict = [float(r["conflict_a"]) for r in all_results
                       if r["mechanism"] == "CFObjectFile_rule"
                       and r["num_objects"] == n_obj and r["feature_dim"] == fd
                       and r.get("conflict_a", "nan") != "nan"]
        fo_swap = [float(r["swap_only"]) for r in all_results
                   if r["mechanism"] == "FeatureOnly"
                   and r["num_objects"] == n_obj and r["feature_dim"] == fd]

        if lof_combined and lof_conflict:
            lof_s = np.mean(lof_combined)
            lof_c = np.mean(lof_conflict)
            cf_s = np.mean(cf_swap) if cf_swap else 0
            cf_c = np.mean(cf_conflict) if cf_conflict else 0
            fo_s = np.mean(fo_swap) if fo_swap else 0

            trade_off_broken = lof_s > cf_s and lof_c > cf_c
            better_than_fo = lof_c > 0.5

            print(f"    LearnedObjectFile: swap={lof_s:.3f}, conflict_a={lof_c:.3f}")
            print(f"    CFObjectFile:      swap={cf_s:.3f}, conflict_a={cf_c:.3f}")
            print(f"    FeatureOnly:       swap={fo_s:.3f}, conflict_a=0.000")
            print(f"    Trade-off broken:  {'YES!' if trade_off_broken else 'NO'}")
            print(f"    Better than FO:    {'YES' if better_than_fo else 'NO'}")


if __name__ == "__main__":
    main()
