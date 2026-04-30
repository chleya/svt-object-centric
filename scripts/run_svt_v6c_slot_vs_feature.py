"""
SVT-v6c: Slot-Based vs Feature-Based Identity Matching

Critical test: Do slot representations carry identity information
beyond what features alone provide?

If slot-based matching also fails under conflict → even stronger conclusion
If slot-based matching survives conflict → slots do help, but need correct usage
"""

import sys
import os
import numpy as np
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v6c_slot_vs_feature"
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


def compute_assignment_accuracy(pred_identity, true_identity):
    correct = (pred_identity == true_identity).all(axis=1)
    return float(correct.mean())


def run_slot_vs_feature_experiment(num_objects, feature_dim, seed, epochs=EPOCHS):
    from models.slot_attention_model import SetBasedSlotAttentionModel
    from models.dinosaur_model import DINOSAURModel
    from models.slot_based_identity import SlotBasedIdentityWrapper
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

    models_config = [
        ("SlotAttention", lambda: SetBasedSlotAttentionModel(
            num_objects=num_objects, feature_dim=feature_dim, slot_dim=64, sa_iters=3)),
        ("DINOSAUR", lambda: DINOSAURModel(
            num_objects=num_objects, feature_dim=feature_dim, slot_dim=64, sa_iters=3)),
    ]

    for model_name, model_fn in models_config:
        print(f"  Training {model_name} with slot-based identity (seed={seed})...")
        base_model = model_fn()

        wrapper = SlotBasedIdentityWrapper(
            base_model, model_name,
            num_objects=num_objects, feature_dim=feature_dim,
            slot_dim=64, t_obs=10, t_pred=20,
            identity_weight=1.0, temperature=1.0,
        )

        train_model(wrapper, train_data, val_data=clean_test, epochs=epochs,
                    batch_size=64, lr=1e-3, device=DEVICE,
                    uses_features=True, uses_future_features=True, verbose=False)

        swap_test_data = swap_test
        clean_test_data = clean_test

        obs_pos = swap_test_data["observed_positions"]
        obs_feat = swap_test_data.get("object_features_obs")
        fut_feat = swap_test_data.get("object_features_fut")
        fut_pos = swap_test_data["future_positions"]
        true_id = swap_test_data["identity_labels"]

        for method in ["slot", "feature"]:
            if method == "slot":
                pred_id = wrapper.predict_identity_slot(
                    obs_pos, obs_feat, future_positions=fut_pos, future_features=fut_feat)
            else:
                pred_id = wrapper.predict_identity_feature(obs_pos, obs_feat, future_features=fut_feat)

            if isinstance(pred_id, torch.Tensor):
                pred_id = pred_id.cpu().numpy()

            bd = compute_identity_breakdown(pred_id, true_id)

            is_swap_c = clean_test_data["is_swap"]
            no_swap_idx = np.where(~is_swap_c)[0]
            conflict_a = float("nan")

            if len(no_swap_idx) > 0:
                no_swap_data = {k: v[no_swap_idx] for k, v in clean_test_data.items()}
                fut_flipped = flip_future_features(no_swap_data.get("object_features_fut"))

                if method == "slot":
                    pred_id_c = wrapper.predict_identity_slot(
                        no_swap_data["observed_positions"],
                        no_swap_data.get("object_features_obs"),
                        future_positions=no_swap_data["future_positions"],
                        future_features=fut_flipped)
                else:
                    pred_id_c = wrapper.predict_identity_feature(
                        no_swap_data["observed_positions"],
                        no_swap_data.get("object_features_obs"),
                        future_features=fut_flipped)

                if isinstance(pred_id_c, torch.Tensor):
                    pred_id_c = pred_id_c.cpu().numpy()

                true_id_c = no_swap_data["identity_labels"]
                correct = (pred_id_c == true_id_c).all(axis=1)
                conflict_a = float(correct.mean())

            all_results.append({
                "model": model_name,
                "method": method,
                "num_objects": num_objects,
                "feature_dim": feature_dim,
                "seed": seed,
                "swap_only": fmt(bd["identity_swap_only"]),
                "overall": fmt(bd["identity_overall"]),
                "conflict_a": fmt(conflict_a),
            })

            print(f"    {model_name}/{method}: swap={fmt(bd['identity_swap_only'])} conflict_a={fmt(conflict_a)}")

    return all_results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("SVT-v6c: Slot-Based vs Feature-Based Identity Matching")
    print("=" * 70)

    if not TORCH_AVAILABLE:
        print("PyTorch unavailable. Exiting.")
        return

    all_results = []

    print("\n=== 2 Objects + One-Hot Features ===")
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        results = run_slot_vs_feature_experiment(2, 2, seed)
        all_results.extend(results)

    print("\n=== 3 Objects + 16-dim Continuous Features ===")
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        results = run_slot_vs_feature_experiment(3, 16, seed)
        all_results.extend(results)

    save_csv(all_results, "slot_vs_feature_results.csv",
             ["model", "method", "num_objects", "feature_dim", "seed",
              "swap_only", "overall", "conflict_a"])

    with open(os.path.join(OUTPUT_DIR, "full_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("SLOT vs FEATURE COMPARISON (mean across seeds)")
    print("=" * 70)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        key = f"{r['model']}/{r['method']}_{r['num_objects']}obj_fd{r['feature_dim']}"
        grouped[key].append(r)

    print(f"\n{'Config':<45} {'Swap-Only':>12} {'Conflict-A':>12}")
    print("-" * 70)

    for key, runs in sorted(grouped.items()):
        swaps = [float(r["swap_only"]) for r in runs]
        conflicts = [float(r["conflict_a"]) for r in runs if r["conflict_a"] != "nan"]

        def fmt_stat(vals):
            valid = [v for v in vals if not np.isnan(v)]
            if not valid:
                return "nan"
            return f"{np.mean(valid):.3f}±{np.std(valid):.3f}"

        conflict_str = fmt_stat(conflicts) if conflicts else "nan"
        print(f"{key:<45} {fmt_stat(swaps):>12} {conflict_str:>12}")

    print("\n" + "=" * 70)
    print("KEY QUESTION: Does slot-based matching survive conflict?")
    print("=" * 70)

    for model_name in ["SlotAttention", "DINOSAUR"]:
        for method in ["slot", "feature"]:
            conflicts = []
            for r in all_results:
                if r["model"] == model_name and r["method"] == method and r["conflict_a"] != "nan":
                    conflicts.append(float(r["conflict_a"]))
            if conflicts:
                mean_c = np.mean(conflicts)
                status = "SURVIVES" if mean_c > 0.5 else "FAILS"
                print(f"  {model_name}/{method}: Conflict-A = {mean_c:.4f} -> {status}")


if __name__ == "__main__":
    main()
