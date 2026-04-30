"""
SVT-v7: Physics-Informed Trajectory Predictors (Direction B)

Test whether HNN/LNN trajectory predictors improve ObjectFile performance
and break the prediction-identity trade-off.

Key question: Can physics-informed trajectory prediction
make ObjectFile's swap-only identity go from 0.519 to 0.7+?
"""

import sys
import os
import numpy as np
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v7_physics_predictors"
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


def compute_assignment_accuracy(pred_identity, true_identity):
    correct = (pred_identity == true_identity).all(axis=1)
    return float(correct.mean())


def evaluate_trajectory_model(model, test_data):
    obs_pos = test_data["observed_positions"]
    fut_pos = test_data["future_positions"]

    pred_identity = model.predict_identity(obs_pos, future_positions=fut_pos)
    if isinstance(pred_identity, torch.Tensor):
        pred_identity = pred_identity.cpu().numpy()

    true_identity = test_data["identity_labels"]
    breakdown = compute_identity_breakdown(pred_identity, true_identity)

    pred_future = model.predict_future(obs_pos)
    if isinstance(pred_future, torch.Tensor):
        pred_future = pred_future.detach().cpu().numpy()

    skill = compute_prediction_metrics(pred_future, fut_pos)

    return breakdown, skill


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


def run_physics_experiment(num_objects, feature_dim, seed, epochs=EPOCHS):
    from models.physics_predictors import HNNTrajectoryPredictor, LNNTrajectoryPredictor
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
    ood_test = eval_ds.get("ood_force_test", swap_test)

    all_results = []

    predictors = [
        ("MLP", lambda: TrajectoryOnlyAssignment(num_objects=num_objects),
         False, False),
        ("HNN", lambda: HNNTrajectoryPredictor(num_objects=num_objects, hidden_dim=128, hnn_hidden=128),
         False, False),
    ]

    trained_predictors = {}

    for pred_name, pred_fn, uses_feat, uses_fut_feat in predictors:
        print(f"  Training {pred_name} (seed={seed})...")
        model = pred_fn()

        try:
            train_model(model, train_data, val_data=clean_test, epochs=epochs,
                        batch_size=64, lr=1e-3, device=DEVICE,
                        uses_features=uses_feat, uses_future_features=uses_fut_feat, verbose=False)
        except Exception as e:
            print(f"    {pred_name} training failed: {e}")
            continue

        bd, skill = evaluate_trajectory_model(model, swap_test)
        ood_bd, ood_skill = evaluate_trajectory_model(model, ood_test)

        all_results.append({
            "predictor": pred_name,
            "config": "trajectory_only",
            "num_objects": num_objects,
            "feature_dim": feature_dim,
            "seed": seed,
            "swap_only": fmt(bd["identity_swap_only"]),
            "overall": fmt(bd["identity_overall"]),
            "clean_skill": fmt(skill.get("skill_score", 0)),
            "ood_skill": fmt(ood_skill.get("skill_score", 0)),
            "ood_swap_only": fmt(ood_bd["identity_swap_only"]),
        })

        print(f"    {pred_name}: swap={fmt(bd['identity_swap_only'])} skill={fmt(skill.get('skill_score',0))} ood_skill={fmt(ood_skill.get('skill_score',0))}")

        trained_predictors[pred_name] = model

        cf_model = ConflictFirstObjectFile(
            traj_model=model, strategy="margin_gated",
            num_objects=num_objects, feature_dim=feature_dim,
            traj_margin_advantage=1.2)

        cf_bd = evaluate_obj_file(cf_model, swap_test)

        is_swap_c = clean_test["is_swap"]
        no_swap_idx = np.where(~is_swap_c)[0]
        conflict_a = float("nan")

        if len(no_swap_idx) > 0:
            no_swap_data = {k: v[no_swap_idx] for k, v in clean_test.items()}
            fut_flipped = flip_future_features(no_swap_data.get("object_features_fut"))

            r = cf_model.predict_identity(
                no_swap_data["observed_positions"],
                no_swap_data.get("object_features_obs"),
                no_swap_data["future_positions"],
                fut_flipped, return_conflict_info=True)
            pred_id_c = r[0]
            if isinstance(pred_id_c, torch.Tensor):
                pred_id_c = pred_id_c.cpu().numpy()
            true_id_c = no_swap_data["identity_labels"]
            correct = (pred_id_c == true_id_c).all(axis=1)
            conflict_a = float(correct.mean())

        all_results.append({
            "predictor": f"CF+{pred_name}",
            "config": "conflict_first",
            "num_objects": num_objects,
            "feature_dim": feature_dim,
            "seed": seed,
            "swap_only": fmt(cf_bd["identity_swap_only"]),
            "overall": fmt(cf_bd["identity_overall"]),
            "clean_skill": "",
            "ood_skill": "",
            "ood_swap_only": "",
            "conflict_a": fmt(conflict_a),
        })

        print(f"    CF+{pred_name}: swap={fmt(cf_bd['identity_swap_only'])} conflict_a={fmt(conflict_a)}")

    return all_results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("SVT-v7: Physics-Informed Trajectory Predictors")
    print("=" * 70)

    if not TORCH_AVAILABLE:
        print("PyTorch unavailable. Exiting.")
        return

    all_results = []

    print("\n=== 2 Objects + One-Hot Features ===")
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        results = run_physics_experiment(2, 2, seed)
        all_results.extend(results)

    print("\n=== 3 Objects + 16-dim Continuous Features ===")
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        results = run_physics_experiment(3, 16, seed)
        all_results.extend(results)

    save_csv(all_results, "physics_predictor_results.csv",
             ["predictor", "config", "num_objects", "feature_dim", "seed",
              "swap_only", "overall", "clean_skill", "ood_skill",
              "ood_swap_only", "conflict_a"])

    with open(os.path.join(OUTPUT_DIR, "full_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("PHYSICS PREDICTOR COMPARISON")
    print("=" * 70)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        if r["config"] == "trajectory_only":
            key = f"{r['predictor']}_{r['num_objects']}obj_fd{r['feature_dim']}"
            grouped[key].append(r)

    print(f"\n{'Predictor':<30} {'Swap-Only':>12} {'Skill':>12} {'OOD-Skill':>12} {'OOD-Swap':>12}")
    print("-" * 80)

    for key, runs in sorted(grouped.items()):
        swaps = [float(r["swap_only"]) for r in runs]
        skills = [float(r["clean_skill"]) for r in runs if r["clean_skill"]]
        ood_skills = [float(r["ood_skill"]) for r in runs if r["ood_skill"]]
        ood_swaps = [float(r["ood_swap_only"]) for r in runs if r["ood_swap_only"]]

        def fmt_stat(vals):
            valid = [v for v in vals if not np.isnan(v)]
            if not valid:
                return "nan"
            return f"{np.mean(valid):.3f}±{np.std(valid):.3f}"

        print(f"{key:<30} {fmt_stat(swaps):>12} {fmt_stat(skills):>12} {fmt_stat(ood_skills):>12} {fmt_stat(ood_swaps):>12}")

    print("\n--- ObjectFile with Different Trajectory Predictors ---")
    cf_grouped = defaultdict(list)
    for r in all_results:
        if r["config"] == "conflict_first":
            key = f"{r['predictor']}_{r['num_objects']}obj_fd{r['feature_dim']}"
            cf_grouped[key].append(r)

    print(f"\n{'ObjectFile Config':<30} {'Swap-Only':>12} {'Conflict-A':>12}")
    print("-" * 55)

    for key, runs in sorted(cf_grouped.items()):
        swaps = [float(r["swap_only"]) for r in runs]
        conflicts = [float(r.get("conflict_a", "nan")) for r in runs if r.get("conflict_a", "nan") != "nan"]

        def fmt_stat(vals):
            valid = [v for v in vals if not np.isnan(v)]
            if not valid:
                return "nan"
            return f"{np.mean(valid):.3f}±{np.std(valid):.3f}"

        conflict_str = fmt_stat(conflicts) if conflicts else "nan"
        print(f"{key:<30} {fmt_stat(swaps):>12} {conflict_str:>12}")


if __name__ == "__main__":
    main()
