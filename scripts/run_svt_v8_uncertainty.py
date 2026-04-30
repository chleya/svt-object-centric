"""
SVT-v8: Uncertainty-Aware ObjectFile (Direction C)

Compare rule-based vs evidential uncertainty for ObjectFile:
1. ConflictFirstObjectFile (rule-based confidence)
2. UncertaintyAwareObjectFile (evidential uncertainty)

Key question: Does learned uncertainty improve conflict resolution
and confidence calibration?
"""

import sys
import os
import numpy as np
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v8_uncertainty"
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
from metrics.object_file_metrics import compute_confidence_calibration


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


def evaluate_obj_file(mech, test_data, return_conf_info=False):
    obs_pos = test_data["observed_positions"]
    obs_feat = test_data.get("object_features_obs")
    fut_feat = test_data.get("object_features_fut")
    fut_pos = test_data["future_positions"]

    if return_conf_info:
        result = mech.predict_identity(obs_pos, obs_feat, fut_pos, fut_feat,
                                       return_conflict_info=True)
        pred_identity = result[0]
        confidences = result[1]
        sources = result[2]
        abstain_flags = result[3]
    else:
        pred_identity = mech.predict_identity(obs_pos, obs_feat, fut_pos, fut_feat)
        confidences = []
        sources = []
        abstain_flags = []

    true_identity = test_data["identity_labels"]
    breakdown = compute_identity_breakdown(pred_identity, true_identity)

    if confidences and return_conf_info:
        try:
            conf_array = np.array([float(c) if not isinstance(c, dict) else float(c.get('confidence', 0.5)) for c in confidences])
            cal = compute_confidence_calibration(pred_identity, true_identity, conf_array)
            breakdown['confidence_calibration'] = cal
        except Exception:
            breakdown['confidence_calibration'] = None

        source_counts = {}
        for s in sources:
            source_counts[s] = source_counts.get(s, 0) + 1
        breakdown['source_counts'] = source_counts

        abstain_rate = sum(abstain_flags) / len(abstain_flags) if abstain_flags else 0
        breakdown['abstain_rate'] = abstain_rate

    return breakdown


def run_uncertainty_experiment(num_objects, feature_dim, seed, epochs=EPOCHS):
    from models.object_file_models import TrajectoryOnlyAssignment, ConflictFirstObjectFile
    from models.evidential_object_file import EvidentialIdentityHead, UncertaintyAwareObjectFile
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

    print(f"  Training trajectory model...")
    traj_model = TrajectoryOnlyAssignment(num_objects=num_objects)
    train_model(traj_model, train_data, val_data=clean_test, epochs=epochs,
                batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=False, uses_future_features=False, verbose=False)

    print(f"  Evaluating ConflictFirstObjectFile (rule-based)...")
    cf_model = ConflictFirstObjectFile(
        traj_model=traj_model, strategy="margin_gated",
        num_objects=num_objects, feature_dim=feature_dim,
        traj_margin_advantage=1.2)

    cf_bd = evaluate_obj_file(cf_model, swap_test, return_conf_info=True)
    all_results.append({
        "mechanism": "CFObjectFile_rule",
        "num_objects": num_objects,
        "feature_dim": feature_dim,
        "seed": seed,
        "swap_only": fmt(cf_bd["identity_swap_only"]),
        "overall": fmt(cf_bd["identity_overall"]),
        "no_swap": fmt(cf_bd["identity_no_swap"]),
        "abstain_rate": fmt(cf_bd.get("abstain_rate", 0)),
    })
    print(f"    CF rule: swap={fmt(cf_bd['identity_swap_only'])} abstain={fmt(cf_bd.get('abstain_rate',0))}")

    print(f"  Training EvidentialIdentityHead...")
    ev_head = EvidentialIdentityHead(
        feature_dim=feature_dim, slot_dim=64,
        num_objects=num_objects, hidden_dim=128)

    obs_feat_train = train_data.get("object_features_obs")
    fut_feat_train = train_data.get("object_features_fut")
    obs_pos_train = train_data["observed_positions"]
    fut_pos_train = train_data["future_positions"]
    ids_train = train_data["identity_labels"]

    pred_traj_train = traj_model.predict_future(obs_pos_train)
    if isinstance(pred_traj_train, torch.Tensor):
        pred_traj_train = pred_traj_train.detach().cpu().numpy()

    optimizer = torch.optim.Adam(ev_head.parameters(), lr=1e-3)
    ev_head.train()

    n_batches = len(obs_pos_train) // 64
    for epoch in range(epochs):
        indices = np.random.permutation(len(obs_pos_train))
        total_loss = 0
        for batch_idx in range(n_batches):
            batch_indices = indices[batch_idx * 64:(batch_idx + 1) * 64]

            obs_f = torch.FloatTensor(obs_feat_train[batch_indices, 0, :, :])
            fut_f = torch.FloatTensor(fut_feat_train[batch_indices, 0, :, :])
            fut_p = torch.FloatTensor(fut_pos_train[batch_indices, 0, :, :])
            pred_p = torch.FloatTensor(pred_traj_train[batch_indices, 0, :, :])
            last_v = torch.FloatTensor(
                obs_pos_train[batch_indices, -1, :, :] -
                obs_pos_train[batch_indices, -2, :, :])
            ids_t = torch.LongTensor(ids_train[batch_indices])

            loss, feat_l, traj_l, kl = ev_head.compute_loss(
                obs_f, fut_f, ids_t, fut_p, pred_p, last_v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    for strategy in ["lower_uncertainty", "epistemic_gated"]:
        print(f"  Evaluating UncertaintyAwareObjectFile ({strategy})...")
        ua_model = UncertaintyAwareObjectFile(
            ev_head, traj_model=traj_model, num_objects=num_objects,
            feature_dim=feature_dim, conflict_strategy=strategy,
            epistemic_threshold=0.5)

        ua_bd = evaluate_obj_file(ua_model, swap_test, return_conf_info=True)

        is_swap_c = clean_test["is_swap"]
        no_swap_idx = np.where(~is_swap_c)[0]
        conflict_a = float("nan")

        if len(no_swap_idx) > 0:
            no_swap_data = {k: v[no_swap_idx] for k, v in clean_test.items()}
            fut_flipped = flip_future_features(no_swap_data.get("object_features_fut"))

            r = ua_model.predict_identity(
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
            "mechanism": f"UAObjectFile_{strategy}",
            "num_objects": num_objects,
            "feature_dim": feature_dim,
            "seed": seed,
            "swap_only": fmt(ua_bd["identity_swap_only"]),
            "overall": fmt(ua_bd["identity_overall"]),
            "no_swap": fmt(ua_bd["identity_no_swap"]),
            "abstain_rate": fmt(ua_bd.get("abstain_rate", 0)),
            "conflict_a": fmt(conflict_a),
        })

        print(f"    UA {strategy}: swap={fmt(ua_bd['identity_swap_only'])} conflict_a={fmt(conflict_a)} abstain={fmt(ua_bd.get('abstain_rate',0))}")

    return all_results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("SVT-v8: Uncertainty-Aware ObjectFile")
    print("=" * 70)

    if not TORCH_AVAILABLE:
        print("PyTorch unavailable. Exiting.")
        return

    all_results = []

    print("\n=== 2 Objects + One-Hot Features ===")
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        results = run_uncertainty_experiment(2, 2, seed)
        all_results.extend(results)

    print("\n=== 3 Objects + 16-dim Continuous Features ===")
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        results = run_uncertainty_experiment(3, 16, seed)
        all_results.extend(results)

    save_csv(all_results, "uncertainty_results.csv",
             ["mechanism", "num_objects", "feature_dim", "seed",
              "swap_only", "overall", "no_swap", "abstain_rate", "conflict_a"])

    with open(os.path.join(OUTPUT_DIR, "full_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("UNCERTAINTY COMPARISON (mean ± std)")
    print("=" * 70)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        key = f"{r['mechanism']}_{r['num_objects']}obj_fd{r['feature_dim']}"
        grouped[key].append(r)

    print(f"\n{'Mechanism':<40} {'Swap-Only':>12} {'Conflict-A':>12} {'Abstain':>12}")
    print("-" * 78)

    for key, runs in sorted(grouped.items()):
        swaps = [float(r["swap_only"]) for r in runs]
        conflicts = [float(r["conflict_a"]) for r in runs if r.get("conflict_a", "nan") != "nan"]
        abstains = [float(r["abstain_rate"]) for r in runs]

        def fmt_stat(vals):
            valid = [v for v in vals if not np.isnan(v)]
            if not valid:
                return "nan"
            return f"{np.mean(valid):.3f}±{np.std(valid):.3f}"

        conflict_str = fmt_stat(conflicts) if conflicts else "nan"
        print(f"{key:<40} {fmt_stat(swaps):>12} {conflict_str:>12} {fmt_stat(abstains):>12}")


if __name__ == "__main__":
    main()
