"""
SVT-v6b: Extended Diagnosis — DINOSAUR + 3-Object + Multi-Seed

Extends v6 with:
1. DINOSAUR model (Seitzer et al., 2024)
2. 3-object + 16-dim continuous feature test
3. Multi-seed stability (3 seeds)
"""

import sys
import os
import numpy as np
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v6b_extended"
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


def shuffle_features(features, seed=42):
    if features is None:
        return None
    rng = np.random.RandomState(seed)
    shuffled = features.copy()
    B, T, N, Fd = shuffled.shape
    for b in range(B):
        for t in range(T):
            perm = rng.permutation(N)
            shuffled[b, t] = shuffled[b, t, perm]
    return shuffled


def zero_features(features):
    if features is None:
        return None
    return np.zeros_like(features)


def flip_future_features(future_features):
    if future_features is None:
        return None
    flipped = future_features.copy()
    flipped[:, :, 0, :], flipped[:, :, 1, :] = flipped[:, :, 1, :].copy(), flipped[:, :, 0, :].copy()
    return flipped


def compute_assignment_accuracy(pred_identity, true_identity):
    correct = (pred_identity == true_identity).all(axis=1)
    return float(correct.mean())


def evaluate_learned_model(model, test_data, obs_feat_override=None, fut_feat_override=None):
    obs_pos = test_data["observed_positions"]
    obs_feat = obs_feat_override if obs_feat_override is not None else test_data.get("object_features_obs")
    fut_feat = fut_feat_override if fut_feat_override is not None else test_data.get("object_features_fut")

    pred_future = model.predict_future(obs_pos, obs_feat)
    if isinstance(pred_future, torch.Tensor):
        pred_future = pred_future.cpu().numpy()

    pred_identity = model.predict_identity(obs_pos, obs_feat, future_features=fut_feat)
    if isinstance(pred_identity, torch.Tensor):
        pred_identity = pred_identity.cpu().numpy()

    true_identity = test_data["identity_labels"]
    breakdown = compute_identity_breakdown(pred_identity, true_identity)
    return breakdown


def evaluate_trajectory_only(model, test_data):
    obs_pos = test_data["observed_positions"]
    fut_pos = test_data["future_positions"]

    pred_identity = model.predict_identity(obs_pos, future_positions=fut_pos)
    if isinstance(pred_identity, torch.Tensor):
        pred_identity = pred_identity.cpu().numpy()

    true_identity = test_data["identity_labels"]
    breakdown = compute_identity_breakdown(pred_identity, true_identity)
    return breakdown


def evaluate_obj_file(mech, test_data, obs_feat_override=None,
                       fut_feat_override=None, occlusion_mask=None):
    obs_pos = test_data["observed_positions"]
    obs_feat = obs_feat_override if obs_feat_override is not None else test_data.get("object_features_obs")
    fut_feat = fut_feat_override if fut_feat_override is not None else test_data.get("object_features_fut")
    fut_pos = test_data["future_positions"]

    result = mech.predict_identity(obs_pos, obs_feat, fut_pos, fut_feat,
                                   occlusion_mask=occlusion_mask, return_conflict_info=True)

    pred_identity = result[0]
    true_identity = test_data["identity_labels"]
    breakdown = compute_identity_breakdown(pred_identity, true_identity)
    return breakdown


def run_key_tests(model, model_type, test_data, clean_test=None):
    results = {}

    if model_type == "learned":
        bd = evaluate_learned_model(model, test_data)
    elif model_type == "traj_only":
        bd = evaluate_trajectory_only(model, test_data)
    elif model_type == "obj_file":
        bd = evaluate_obj_file(model, test_data)

    results["swap_only"] = bd["identity_swap_only"]
    results["overall"] = bd["identity_overall"]

    obs_feat = test_data.get("object_features_obs")
    fut_feat = test_data.get("object_features_fut")

    conflict_data = clean_test if clean_test is not None else test_data
    is_swap_c = conflict_data["is_swap"]
    no_swap_idx = np.where(~is_swap_c)[0]

    if len(no_swap_idx) > 0:
        no_swap_data = {k: v[no_swap_idx] for k, v in conflict_data.items()}
        fut_flipped = flip_future_features(no_swap_data.get("object_features_fut"))

        true_id = no_swap_data["identity_labels"]
        N = true_id.shape[1]
        is_actually_swap = np.zeros(len(true_id), dtype=bool)
        for i in range(N):
            is_actually_swap = is_actually_swap | (true_id[:, i] != i)
        n_true_no_swap = int((~is_actually_swap).sum())

        if model_type == "learned":
            pred_id = model.predict_identity(
                no_swap_data["observed_positions"],
                no_swap_data.get("object_features_obs"),
                future_features=fut_flipped)
            if isinstance(pred_id, torch.Tensor):
                pred_id = pred_id.cpu().numpy()
        elif model_type == "traj_only":
            pred_id = model.predict_identity(
                no_swap_data["observed_positions"],
                future_positions=no_swap_data["future_positions"])
            if isinstance(pred_id, torch.Tensor):
                pred_id = pred_id.cpu().numpy()
        elif model_type == "obj_file":
            r = model.predict_identity(
                no_swap_data["observed_positions"],
                no_swap_data.get("object_features_obs"),
                no_swap_data["future_positions"],
                fut_flipped, return_conflict_info=True)
            pred_id = r[0]

        correct = (pred_id == true_id).all(axis=1)
        results["conflict_a"] = float(correct.mean())

        if n_true_no_swap > 0:
            results["conflict_a_traj_correct"] = float(correct[~is_actually_swap].mean())
        else:
            results["conflict_a_traj_correct"] = float("nan")
    else:
        results["conflict_a"] = float("nan")
        results["conflict_a_traj_correct"] = float("nan")

    obs_shuf = shuffle_features(obs_feat, seed=42)
    fut_shuf = shuffle_features(fut_feat, seed=43)
    obs_zero = zero_features(obs_feat)
    fut_zero = zero_features(fut_feat)

    if model_type == "learned":
        bd_shuf = evaluate_learned_model(model, test_data, obs_feat_override=obs_shuf, fut_feat_override=fut_shuf)
        bd_zero = evaluate_learned_model(model, test_data, obs_feat_override=obs_zero, fut_feat_override=fut_zero)
    elif model_type == "obj_file":
        bd_shuf = evaluate_obj_file(model, test_data, obs_feat_override=obs_shuf, fut_feat_override=fut_shuf)
        bd_zero = evaluate_obj_file(model, test_data, obs_feat_override=obs_zero, fut_feat_override=fut_zero)
    else:
        bd_shuf = bd
        bd_zero = bd

    results["feat_dep"] = bd["identity_swap_only"] - bd_shuf["identity_swap_only"]
    results["traj_dep"] = bd_zero["identity_swap_only"]

    return results


def run_experiment(num_objects, feature_dim, seed, epochs=EPOCHS):
    from models.slot_attention_model import SetBasedSlotAttentionModel
    from models.rims_model import RIMsModel
    from models.savi_model import SAViModel
    from models.dinosaur_model import DINOSAURModel
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

    models_config = [
        ("SlotAttention", "learned", lambda: SetBasedSlotAttentionModel(
            num_objects=num_objects, feature_dim=feature_dim, slot_dim=64, sa_iters=3),
         True, True),
        ("RIMs", "learned", lambda: RIMsModel(
            num_objects=num_objects, feature_dim=feature_dim, num_rims=num_objects, rim_dim=64, top_k=1),
         True, True),
        ("SAVi", "learned", lambda: SAViModel(
            num_objects=num_objects, feature_dim=feature_dim, slot_dim=64, sa_iters=3),
         True, True),
        ("DINOSAUR", "learned", lambda: DINOSAURModel(
            num_objects=num_objects, feature_dim=feature_dim, slot_dim=64, sa_iters=3),
         True, True),
        ("FeatureOnly", "learned", lambda: FeatureOnlyAssignmentHead(
            num_objects=num_objects, feature_dim=feature_dim),
         True, True),
        ("Hybrid", "learned", lambda: HybridTrajectoryFeatureAssignmentHead(
            num_objects=num_objects, feature_dim=feature_dim, beta=1.0),
         True, True),
        ("TrajOnly", "traj_only", lambda: TrajectoryOnlyAssignment(
            num_objects=num_objects),
         False, False),
    ]

    all_results = []

    for model_name, model_type, model_fn, uses_feat, uses_fut_feat in models_config:
        print(f"  Training {model_name} (seed={seed}, {num_objects}obj, fdim={feature_dim})...")
        model = model_fn()
        train_model(model, train_data, val_data=clean_test, epochs=epochs,
                    batch_size=64, lr=1e-3, device=DEVICE,
                    uses_features=uses_feat, uses_future_features=uses_fut_feat, verbose=False)

        test_results = run_key_tests(model, model_type, swap_test, clean_test=clean_test)

        all_results.append({
            "model": model_name,
            "model_type": model_type,
            "num_objects": num_objects,
            "feature_dim": feature_dim,
            "seed": seed,
            "swap_only": fmt(test_results["swap_only"]),
            "overall": fmt(test_results["overall"]),
            "conflict_a": fmt(test_results["conflict_a"]),
            "conflict_a_traj_correct": fmt(test_results.get("conflict_a_traj_correct", "nan")),
            "feat_dep": fmt(test_results["feat_dep"]),
            "traj_dep": fmt(test_results["traj_dep"]),
        })

        print(f"    {model_name}: swap={fmt(test_results['swap_only'])} conflict_a={fmt(test_results['conflict_a'])} feat_dep={fmt(test_results['feat_dep'])} traj_dep={fmt(test_results['traj_dep'])}")

    traj_model = TrajectoryOnlyAssignment(num_objects=num_objects)
    train_model(traj_model, train_data, val_data=clean_test, epochs=epochs,
                batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=False, uses_future_features=False, verbose=False)
    cf_model = ConflictFirstObjectFile(traj_model=traj_model, strategy="margin_gated",
                                        num_objects=num_objects, feature_dim=feature_dim,
                                        traj_margin_advantage=1.2)

    cf_results = run_key_tests(cf_model, "obj_file", swap_test, clean_test=clean_test)
    all_results.append({
        "model": "CFObjectFile",
        "model_type": "obj_file",
        "num_objects": num_objects,
        "feature_dim": feature_dim,
        "seed": seed,
        "swap_only": fmt(cf_results["swap_only"]),
        "overall": fmt(cf_results["overall"]),
        "conflict_a": fmt(cf_results["conflict_a"]),
        "conflict_a_traj_correct": fmt(cf_results.get("conflict_a_traj_correct", "nan")),
        "feat_dep": fmt(cf_results["feat_dep"]),
        "traj_dep": fmt(cf_results["traj_dep"]),
    })
    print(f"    CFObjectFile: swap={fmt(cf_results['swap_only'])} conflict_a={fmt(cf_results['conflict_a'])} feat_dep={fmt(cf_results['feat_dep'])} traj_dep={fmt(cf_results['traj_dep'])}")

    return all_results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("SVT-v6b: Extended Diagnosis — DINOSAUR + 3-Obj + Multi-Seed")
    print("=" * 70)

    if not TORCH_AVAILABLE:
        print("PyTorch unavailable. Exiting.")
        return

    all_results = []

    print("\n" + "=" * 70)
    print("Experiment 1: 2 Objects + One-Hot Features (3 seeds)")
    print("=" * 70)
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        results = run_experiment(num_objects=2, feature_dim=2, seed=seed)
        all_results.extend(results)

    print("\n" + "=" * 70)
    print("Experiment 2: 3 Objects + 16-dim Continuous Features (3 seeds)")
    print("=" * 70)
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        results = run_experiment(num_objects=3, feature_dim=16, seed=seed)
        all_results.extend(results)

    save_csv(all_results, "extended_results.csv",
             ["model", "model_type", "num_objects", "feature_dim", "seed",
              "swap_only", "overall", "conflict_a", "conflict_a_traj_correct",
              "feat_dep", "traj_dep"])

    with open(os.path.join(OUTPUT_DIR, "full_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("MULTI-SEED SUMMARY (mean ± std)")
    print("=" * 70)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        key = f"{r['model']}_{r['num_objects']}obj_fd{r['feature_dim']}"
        grouped[key].append(r)

    print(f"\n{'Config':<35} {'Swap-Only':>15} {'Conflict-A':>15} {'Feat-Dep':>15} {'Traj-Dep':>15}")
    print("-" * 95)

    for key, runs in sorted(grouped.items()):
        swaps = [float(r["swap_only"]) for r in runs]
        conflicts = [float(r["conflict_a"]) for r in runs]
        feat_deps = [float(r["feat_dep"]) for r in runs]
        traj_deps = [float(r["traj_dep"]) for r in runs]

        def fmt_stat(vals):
            return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"

        print(f"{key:<35} {fmt_stat(swaps):>15} {fmt_stat(conflicts):>15} {fmt_stat(feat_deps):>15} {fmt_stat(traj_deps):>15}")

    print("\n" + "=" * 70)
    print("KEY FINDING: Do published models fail under conflict?")
    print("=" * 70)

    published_models = ["SlotAttention", "RIMs", "SAVi", "DINOSAUR"]
    for model_name in published_models:
        conflict_vals = []
        for r in all_results:
            if r["model"] == model_name:
                conflict_vals.append(float(r["conflict_a"]))
        if conflict_vals:
            mean_conflict = np.mean(conflict_vals)
            status = "FAIL (0.000)" if mean_conflict < 0.01 else f"OK ({mean_conflict:.3f})"
            print(f"  {model_name}: Conflict-A = {mean_conflict:.4f} -> {status}")

    robust_models = ["Hybrid", "TrajOnly", "CFObjectFile"]
    for model_name in robust_models:
        conflict_vals = []
        for r in all_results:
            if r["model"] == model_name:
                conflict_vals.append(float(r["conflict_a"]))
        if conflict_vals:
            mean_conflict = np.mean(conflict_vals)
            status = "ROBUST" if mean_conflict > 0.9 else f"PARTIAL ({mean_conflict:.3f})"
            print(f"  {model_name}: Conflict-A = {mean_conflict:.4f} -> {status}")


if __name__ == "__main__":
    main()
