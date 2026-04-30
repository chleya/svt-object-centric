"""
SVT-v6: Diagnosing Published Object-Centric Models

Direction A: Use SVT stress tests to diagnose Slot Attention, RIMs, SAVi,
and compare with our existing models (FeatureOnly, TrajectoryOnly, ObjectFile).

Key question: Do published object-centric models also fail under
feature-trajectory conflict? If yes, this is a structural deficiency
across the field, not just our ObjectFile's problem.
"""

import sys
import os
import numpy as np
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v6_published_models"
SEED = 0
EPOCHS = 30

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
from metrics.object_file_metrics import (
    compute_confidence_calibration, compute_conflict_gate_stats,
    compute_abstention_metrics,
)


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


def add_feature_noise(features, noise_std=0.1, seed=42):
    if features is None:
        return None
    rng = np.random.RandomState(seed)
    noise = rng.randn(*features.shape).astype(features.dtype) * noise_std
    return features + noise


def make_occlusion_mask(future_positions, occlusion_ratio):
    if occlusion_ratio <= 0:
        return None
    B, T, N, _ = future_positions.shape
    mask = np.zeros((B, T, N), dtype=bool)
    n_occluded = int(T * occlusion_ratio)
    start = (T - n_occluded) // 2
    mask[:, start:start + n_occluded, :] = True
    return mask


def occlude_features(future_features, occlusion_mask):
    if future_features is None or occlusion_mask is None:
        return future_features
    occluded = future_features.copy()
    for b in range(occluded.shape[0]):
        for t in range(occluded.shape[1]):
            for n in range(occluded.shape[2]):
                if occlusion_mask[b, t, n]:
                    occluded[b, t, n, :] = 0.0
    return occluded


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
    asgn_acc = compute_assignment_accuracy(pred_identity, true_identity)
    breakdown = compute_identity_breakdown(pred_identity, true_identity)
    return breakdown, asgn_acc


def evaluate_trajectory_only(model, test_data):
    obs_pos = test_data["observed_positions"]
    fut_pos = test_data["future_positions"]

    pred_identity = model.predict_identity(obs_pos, future_positions=fut_pos)
    if isinstance(pred_identity, torch.Tensor):
        pred_identity = pred_identity.cpu().numpy()

    true_identity = test_data["identity_labels"]
    asgn_acc = compute_assignment_accuracy(pred_identity, true_identity)
    breakdown = compute_identity_breakdown(pred_identity, true_identity)
    return breakdown, asgn_acc


def evaluate_obj_file(mech, test_data, obs_feat_override=None,
                       fut_feat_override=None, occlusion_mask=None):
    obs_pos = test_data["observed_positions"]
    obs_feat = obs_feat_override if obs_feat_override is not None else test_data.get("object_features_obs")
    fut_feat = fut_feat_override if fut_feat_override is not None else test_data.get("object_features_fut")
    fut_pos = test_data["future_positions"]

    result = mech.predict_identity(obs_pos, obs_feat, fut_pos, fut_feat,
                                   occlusion_mask=occlusion_mask, return_conflict_info=True)

    pred_identity = result[0]
    confidences = result[1] if len(result) > 1 else []
    sources = result[2] if len(result) > 2 else []
    abstain_flags = result[3] if len(result) > 3 else [False] * len(pred_identity)

    true_identity = test_data["identity_labels"]
    asgn_acc = compute_assignment_accuracy(pred_identity, true_identity)
    breakdown = compute_identity_breakdown(pred_identity, true_identity)
    return breakdown, asgn_acc, confidences, sources, abstain_flags


def run_stress_tests(model, model_type, test_data, model_name):
    results = []

    if model_type == "learned":
        bd, _ = evaluate_learned_model(model, test_data)
    elif model_type == "traj_only":
        bd, _ = evaluate_trajectory_only(model, test_data)
    elif model_type == "obj_file":
        bd, _, _, _, _ = evaluate_obj_file(model, test_data)

    results.append({
        "test": "baseline",
        "swap_only": fmt(bd["identity_swap_only"]),
        "overall": fmt(bd["identity_overall"]),
        "no_swap": fmt(bd["identity_no_swap"]),
    })

    obs_feat = test_data.get("object_features_obs")
    fut_feat = test_data.get("object_features_fut")

    for noise_std in [0.1, 0.3, 0.5]:
        obs_noisy = add_feature_noise(obs_feat, noise_std, seed=42)
        fut_noisy = add_feature_noise(fut_feat, noise_std, seed=43)

        if model_type == "learned":
            bd_n, _ = evaluate_learned_model(model, test_data, obs_feat_override=obs_noisy, fut_feat_override=fut_noisy)
        elif model_type == "obj_file":
            bd_n, _, _, _, _ = evaluate_obj_file(model, test_data, obs_feat_override=obs_noisy, fut_feat_override=fut_noisy)
        else:
            bd_n = bd

        results.append({
            "test": f"feature_noise_{noise_std}",
            "swap_only": fmt(bd_n["identity_swap_only"]),
            "overall": fmt(bd_n["identity_overall"]),
            "no_swap": fmt(bd_n["identity_no_swap"]),
        })

    for occ_ratio in [0.25, 0.5, 0.75]:
        occ_mask = make_occlusion_mask(test_data["future_positions"], occ_ratio)
        fut_occ = occlude_features(fut_feat, occ_mask)

        if model_type == "learned":
            bd_o, _ = evaluate_learned_model(model, test_data, fut_feat_override=fut_occ)
        elif model_type == "obj_file":
            bd_o, _, _, _, _ = evaluate_obj_file(model, test_data, fut_feat_override=fut_occ, occlusion_mask=occ_mask)
        else:
            bd_o = bd

        results.append({
            "test": f"occlusion_{occ_ratio}",
            "swap_only": fmt(bd_o["identity_swap_only"]),
            "overall": fmt(bd_o["identity_overall"]),
            "no_swap": fmt(bd_o["identity_no_swap"]),
        })

    is_swap = test_data["is_swap"]
    no_swap_idx = np.where(~is_swap)[0]

    if len(no_swap_idx) > 0:
        no_swap_data = {k: v[no_swap_idx] for k, v in test_data.items()}
        fut_flipped = flip_future_features(no_swap_data.get("object_features_fut"))

        true_id = no_swap_data["identity_labels"]
        is_actually_swap = true_id[:, 0] != np.arange(test_data["identity_labels"].shape[1])[0]
        for i in range(1, test_data["identity_labels"].shape[1]):
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
        conflict_acc = float(correct.mean())

        traj_correct = float("nan")
        if n_true_no_swap > 0:
            traj_correct = float(correct[~is_actually_swap].mean())

        results.append({
            "test": "conflict_type_A",
            "swap_only": fmt(conflict_acc),
            "overall": fmt(conflict_acc),
            "no_swap": fmt(traj_correct),
        })

    obs_shuf = shuffle_features(obs_feat, seed=SEED)
    fut_shuf = shuffle_features(fut_feat, seed=SEED + 1000)
    obs_zero = zero_features(obs_feat)
    fut_zero = zero_features(fut_feat)

    if model_type == "learned":
        bd_shuf, _ = evaluate_learned_model(model, test_data, obs_feat_override=obs_shuf, fut_feat_override=fut_shuf)
        bd_zero, _ = evaluate_learned_model(model, test_data, obs_feat_override=obs_zero, fut_feat_override=fut_zero)
    elif model_type == "obj_file":
        bd_shuf, _, _, _, _ = evaluate_obj_file(model, test_data, obs_feat_override=obs_shuf, fut_feat_override=fut_shuf)
        bd_zero, _, _, _, _ = evaluate_obj_file(model, test_data, obs_feat_override=obs_zero, fut_feat_override=fut_zero)
    else:
        bd_shuf = bd
        bd_zero = bd

    feat_dep = bd["identity_swap_only"] - bd_shuf["identity_swap_only"]
    traj_dep = bd_zero["identity_swap_only"]

    results.append({
        "test": "feature_ablation_shuffled",
        "swap_only": fmt(bd_shuf["identity_swap_only"]),
        "overall": fmt(bd_shuf["identity_overall"]),
        "no_swap": fmt(bd_shuf["identity_no_swap"]),
    })
    results.append({
        "test": "feature_ablation_zeroed",
        "swap_only": fmt(bd_zero["identity_swap_only"]),
        "overall": fmt(bd_zero["identity_overall"]),
        "no_swap": fmt(bd_zero["identity_no_swap"]),
    })
    results.append({
        "test": "feature_dependency",
        "swap_only": fmt(feat_dep),
        "overall": "",
        "no_swap": "",
    })
    results.append({
        "test": "trajectory_dependency",
        "swap_only": fmt(traj_dep),
        "overall": "",
        "no_swap": "",
    })

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("SVT-v6: Diagnosing Published Object-Centric Models")
    print("=" * 70)

    if not TORCH_AVAILABLE:
        print("PyTorch unavailable. Exiting.")
        return

    from models.slot_attention_model import SetBasedSlotAttentionModel
    from models.rims_model import RIMsModel
    from models.savi_model import SAViModel
    from models.feature_similarity_models import FeatureOnlyAssignmentHead, HybridTrajectoryFeatureAssignmentHead
    from models.object_file_models import (
        TrajectoryOnlyAssignment, MultiTaskTrajectoryPredictor,
        ConflictFirstObjectFile,
    )
    from utils.torch_training import train_model

    eval_ds = generate_v3_dataset(
        n_train=1000, n_test=200, num_objects=2, feature_dim=2,
        feature_mode="feature_bearing",
        force_train_type="attractor", force_test_type="vortex",
        randomize_object_order=True, disjoint_init_split=True, seed=SEED,
    )
    train_data = generate_swap_train(n_train=1000, num_objects=2, feature_dim=2, seed=SEED)
    swap_test = eval_ds["identity_test_swap_only"]
    clean_test = eval_ds["clean_test_id"]

    models_to_test = []

    print("\n[1/6] Training Slot Attention...")
    sa_model = SetBasedSlotAttentionModel(num_objects=2, feature_dim=2, slot_dim=64, sa_iters=3)
    train_model(sa_model, train_data, val_data=clean_test, epochs=EPOCHS,
                batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=True, uses_future_features=True, verbose=False)
    models_to_test.append(("SlotAttention", "learned", sa_model))

    print("[2/6] Training RIMs...")
    rims_model = RIMsModel(num_objects=2, feature_dim=2, num_rims=2, rim_dim=64, top_k=1)
    train_model(rims_model, train_data, val_data=clean_test, epochs=EPOCHS,
                batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=True, uses_future_features=True, verbose=False)
    models_to_test.append(("RIMs", "learned", rims_model))

    print("[3/6] Training SAVi...")
    savi_model = SAViModel(num_objects=2, feature_dim=2, slot_dim=64, sa_iters=3)
    train_model(savi_model, train_data, val_data=clean_test, epochs=EPOCHS,
                batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=True, uses_future_features=True, verbose=False)
    models_to_test.append(("SAVi", "learned", savi_model))

    print("[4/6] Training FeatureOnly (our baseline)...")
    fo_model = FeatureOnlyAssignmentHead(num_objects=2, feature_dim=2)
    train_model(fo_model, train_data, val_data=clean_test, epochs=EPOCHS,
                batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=True, uses_future_features=True, verbose=False)
    models_to_test.append(("FeatureOnly", "learned", fo_model))

    print("[5/6] Training Hybrid (our baseline)...")
    hybrid_model = HybridTrajectoryFeatureAssignmentHead(num_objects=2, feature_dim=2, beta=1.0)
    train_model(hybrid_model, train_data, val_data=clean_test, epochs=EPOCHS,
                batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=True, uses_future_features=True, verbose=False)
    models_to_test.append(("Hybrid", "learned", hybrid_model))

    print("[6/6] Training TrajectoryOnly + ConflictFirstObjectFile...")
    traj_model = TrajectoryOnlyAssignment(num_objects=2)
    train_model(traj_model, train_data, val_data=clean_test, epochs=EPOCHS,
                batch_size=64, lr=1e-3, device=DEVICE,
                uses_features=False, uses_future_features=False, verbose=False)
    cf_model = ConflictFirstObjectFile(traj_model=traj_model, strategy="margin_gated",
                                        num_objects=2, feature_dim=2, traj_margin_advantage=1.2)
    models_to_test.append(("TrajOnly", "traj_only", traj_model))
    models_to_test.append(("CFObjectFile", "obj_file", cf_model))

    all_results = []

    print("\n" + "=" * 70)
    print("Running Stress Tests on All Models")
    print("=" * 70)

    for model_name, model_type, model in models_to_test:
        print(f"\n--- {model_name} ({model_type}) ---")

        stress_results = run_stress_tests(model, model_type, swap_test, model_name)

        for r in stress_results:
            all_results.append({
                "model": model_name,
                "model_type": model_type,
                **r,
            })

        baseline = next(r for r in stress_results if r["test"] == "baseline")
        conflict = next((r for r in stress_results if r["test"] == "conflict_type_A"), None)
        feat_dep = next((r for r in stress_results if r["test"] == "feature_dependency"), None)
        traj_dep = next((r for r in stress_results if r["test"] == "trajectory_dependency"), None)

        print(f"  Baseline swap_only: {baseline['swap_only']}")
        if conflict:
            print(f"  Conflict Type A:   {conflict['swap_only']}")
        if feat_dep:
            print(f"  Feature dep:       {feat_dep['swap_only']}")
        if traj_dep:
            print(f"  Trajectory dep:    {traj_dep['swap_only']}")

    save_csv(all_results, "stress_test_results.csv",
             ["model", "model_type", "test", "swap_only", "overall", "no_swap"])

    print("\n" + "=" * 70)
    print("COMPARATIVE SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Swap-Only':>10} {'Conflict-A':>12} {'Feat-Dep':>10} {'Traj-Dep':>10}")
    print("-" * 65)

    for model_name, model_type, model in models_to_test:
        model_results = [r for r in all_results if r["model"] == model_name]
        baseline = next(r for r in model_results if r["test"] == "baseline")
        conflict = next((r for r in model_results if r["test"] == "conflict_type_A"), None)
        feat_dep = next((r for r in model_results if r["test"] == "feature_dependency"), None)
        traj_dep = next((r for r in model_results if r["test"] == "trajectory_dependency"), None)

        conflict_val = conflict["swap_only"] if conflict else "n/a"
        feat_val = feat_dep["swap_only"] if feat_dep else "n/a"
        traj_val = traj_dep["swap_only"] if traj_dep else "n/a"

        print(f"{model_name:<20} {baseline['swap_only']:>10} {conflict_val:>12} {feat_val:>10} {traj_val:>10}")

    with open(os.path.join(OUTPUT_DIR, "full_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    readme = f"""# SVT-v6: Diagnosing Published Object-Centric Models

## 1. Purpose

Test whether published object-centric models (Slot Attention, RIMs, SAVi) fail
under the same SVT stress tests that exposed our ObjectFile's weaknesses.

**Key question**: Is conflict-resolution failure a structural deficiency across
the entire field, or just our ObjectFile's problem?

## 2. Models Tested

| Model | Type | Source |
|-------|------|--------|
| Slot Attention | Published (Locatello et al., 2020) | Set-based adaptation |
| RIMs | Published (Goyal et al., 2020) | Set-based adaptation |
| SAVi | Published (Kipf et al., 2022) | Set-based adaptation |
| FeatureOnly | Our baseline | Feature similarity only |
| Hybrid | Our baseline | Trajectory + feature |
| TrajOnly | Our baseline | Trajectory only |
| CFObjectFile | Our mechanism | Conflict-first ObjectFile |

## 3. Stress Tests

| Test | Description |
|------|-------------|
| baseline | Normal evaluation on swap-only test |
| feature_noise_0.1/0.3/0.5 | Gaussian noise on features |
| occlusion_0.25/0.5/0.75 | Features zeroed during occlusion |
| conflict_type_A | No-swap + flipped features (feature says swap, trajectory says no-swap) |
| feature_ablation_shuffled | Shuffled features |
| feature_ablation_zeroed | Zeroed features |
| feature_dependency | Drop in swap-only when features shuffled |
| trajectory_dependency | Swap-only with zeroed features |

## 4. Key Expected Outcomes

1. **Slot Attention should fail under conflict** — it uses feature similarity
   for slot assignment, so flipped features should mislead it (like FeatureOnly)
2. **RIMs may be more robust** — independent mechanisms with input attention
   could learn trajectory-based routing
3. **SAVi should show intermediate behavior** — temporal slot persistence helps
   but doesn't add explicit conflict resolution
4. **If all published models fail under conflict**, this validates the claim
   that "current object-centric models lack conflict-resolution structure"

## 5. Implications

- If Slot Attention fails → SVT diagnoses a field-wide structural deficiency
- If RIMs succeeds → competitive attention isn't the answer, but independent
  mechanisms with selective attention might be
- If SAVi succeeds → temporal slot persistence partially addresses the issue
"""

    with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
