"""
External Audit v0: Slot Attention under SVT Stress Tests

Runs the full SVT stress test pipeline on a Slot Attention model
adapted for the SVT (position, feature) input format.

Stress tests:
  1. Clean identity (baseline)
  2. Feature ablation (zero features)
  3. Occlusion without feature (partial feature zeroing)
  4. Feature-trajectory conflict (swapped features)
  5. Confidence calibration (if available)

Output: Structural fingerprint for Slot Attention
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.slot_attention_model import SetBasedSlotAttentionModel
from adapters.base_adapter import SlotAttentionAdapter


def gen_train(n=1000, nobj=2, fdim=2, seed=0):
    rng = np.random.RandomState(seed)
    eps = []
    for _ in range(n):
        ep = _generate_single_episode(t_obs=10, t_pred=20, num_objects=nobj, arena_size=64.0,
            feature_mode="feature_bearing", feature_dim=fdim, randomize_object_order=True,
            identity_test=True, swap_probability=0.5, force_type="attractor",
            field_strength=0.5, damping=0.95, noise_std=0.1, rng=rng)
        eps.append(ep)
    return _stack_episodes(eps, "feature_bearing")


def train_slot_model(model, train_data, epochs=30, batch_size=64, lr=1e-3):
    obs_pos = torch.FloatTensor(train_data["observed_positions"])
    fut_pos = torch.FloatTensor(train_data["future_positions"])
    ids = torch.LongTensor(train_data["identity_labels"])
    obs_feat = torch.FloatTensor(train_data["object_features_obs"]) if "object_features_obs" in train_data else None
    fut_feat = torch.FloatTensor(train_data["object_features_fut"]) if "object_features_fut" in train_data else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches = len(obs_pos) // batch_size

    for epoch in range(epochs):
        indices = np.random.permutation(len(obs_pos))
        total_loss = 0
        for bi in range(n_batches):
            idx = indices[bi * batch_size:(bi + 1) * batch_size]
            loss, mse, id_loss, _ = model.compute_loss(
                obs_pos[idx], fut_pos[idx], ids[idx],
                observed_features=obs_feat[idx] if obs_feat is not None else None,
                future_features=fut_feat[idx] if fut_feat is not None else None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")


def run_stress_test(adapter, test_data, test_name, feature_modifier=None):
    obs_pos = test_data["observed_positions"]
    obs_feat = test_data.get("object_features_obs")
    fut_pos = test_data["future_positions"]
    fut_feat = test_data.get("object_features_fut")
    true_id = test_data["identity_labels"]

    if feature_modifier is not None:
        obs_feat, fut_feat = feature_modifier(obs_feat, fut_feat)

    pred = adapter.predict_identity(obs_pos, obs_feat, fut_pos, fut_feat)
    bd = compute_identity_breakdown(pred, true_id)

    conf = adapter.predict_confidence(obs_pos, obs_feat, fut_pos, fut_feat)

    result = {
        "test_name": test_name,
        "identity_swap_only": bd["identity_swap_only"],
        "identity_overall": bd["identity_overall"],
        "no_swap_bias_gap": bd["identity_overall"] - bd["identity_swap_only"],
    }

    if conf is not None:
        correct_mask = (pred == true_id).all(axis=1)
        if correct_mask.any():
            result["avg_confidence_correct"] = float(conf[correct_mask].mean())
        if (~correct_mask).any():
            result["avg_confidence_incorrect"] = float(conf[~correct_mask].mean())

    print(f"  {test_name}: swap_only={result['identity_swap_only']:.4f}, "
          f"overall={result['identity_overall']:.4f}")

    return result


def modify_ablation(obs_feat, fut_feat):
    return None, None


def modify_occlusion(obs_feat, fut_feat):
    if obs_feat is not None:
        obs_feat_mod = obs_feat.copy()
        obs_feat_mod[:] = 0.0
    else:
        obs_feat_mod = None
    if fut_feat is not None:
        fut_feat_mod = fut_feat.copy()
        fut_feat_mod[:] = 0.0
    else:
        fut_feat_mod = None
    return obs_feat_mod, fut_feat_mod


def modify_conflict(obs_feat, fut_feat):
    if fut_feat is None:
        return obs_feat, fut_feat
    fut_feat_mod = fut_feat.copy()
    if fut_feat_mod.ndim == 4:
        fut_feat_mod[:, :, 0, :], fut_feat_mod[:, :, 1, :] = \
            fut_feat[:, :, 1, :].copy(), fut_feat[:, :, 0, :].copy()
    elif fut_feat_mod.ndim == 3:
        fut_feat_mod[:, 0, :], fut_feat_mod[:, 1, :] = \
            fut_feat[:, 1, :].copy(), fut_feat[:, 0, :].copy()
    return obs_feat, fut_feat_mod


def modify_shuffle(obs_feat, fut_feat):
    if fut_feat is None:
        return obs_feat, fut_feat
    rng = np.random.RandomState(42)
    fut_feat_mod = fut_feat.copy()
    B = fut_feat_mod.shape[0]
    for b in range(B):
        if fut_feat_mod.ndim == 4:
            N = fut_feat_mod.shape[2]
            perm = rng.permutation(N)
            temp = fut_feat_mod[b].copy()
            for i in range(N):
                fut_feat_mod[b, :, i, :] = temp[:, perm[i], :]
        elif fut_feat_mod.ndim == 3:
            N = fut_feat_mod.shape[1]
            perm = rng.permutation(N)
            temp = fut_feat_mod[b].copy()
            for i in range(N):
                fut_feat_mod[b, i, :] = temp[perm[i], :]
    return obs_feat, fut_feat_mod


def main():
    seed = 42
    print("=" * 70)
    print("External Audit v0: Slot Attention under SVT Stress Tests")
    print("=" * 70)

    print("\nGenerating datasets...")
    eval_ds = generate_v3_dataset(n_train=1000, n_test=200, num_objects=2, feature_dim=2,
        feature_mode="feature_bearing", force_train_type="attractor", force_test_type="vortex",
        randomize_object_order=True, disjoint_init_split=True, seed=seed)
    train_data = gen_train(n=1000, seed=seed)
    swap_test = eval_ds["identity_test_swap_only"]
    clean_test = eval_ds["clean_test_id"]

    print("\nTraining Slot Attention model...")
    model = SetBasedSlotAttentionModel(
        t_obs=10, t_pred=20, num_objects=2, dim=2, feature_dim=2,
        slot_dim=64, n_slots=2, sa_iters=3, hidden_dim=128,
        identity_weight=1.0, temperature=1.0)
    train_slot_model(model, train_data, epochs=30, batch_size=64, lr=1e-3)

    adapter = SlotAttentionAdapter(model)

    print("\n" + "=" * 70)
    print("Running SVT Stress Tests on Slot Attention")
    print("=" * 70)

    results = []

    print("\n[1/5] Clean identity test...")
    results.append(run_stress_test(adapter, clean_test, "clean_identity"))

    print("\n[2/5] Feature ablation test...")
    results.append(run_stress_test(adapter, clean_test, "feature_ablation",
                                    feature_modifier=modify_ablation))

    print("\n[3/5] Occlusion without feature test...")
    results.append(run_stress_test(adapter, clean_test, "occlusion_no_feature",
                                    feature_modifier=modify_occlusion))

    print("\n[4/5] Feature-trajectory conflict test...")
    results.append(run_stress_test(adapter, swap_test, "feature_trajectory_conflict",
                                    feature_modifier=modify_conflict))

    print("\n[5/5] Shuffled feature test...")
    results.append(run_stress_test(adapter, clean_test, "shuffled_features",
                                    feature_modifier=modify_shuffle))

    print("\n" + "=" * 70)
    print("STRUCTURAL FINGERPRINT: Slot Attention")
    print("=" * 70)
    print(f"{'Test':<30} {'Swap-only':>10} {'Overall':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['test_name']:<30} {r['identity_swap_only']:>10.4f} {r['identity_overall']:>10.4f}")

    clean_swap = results[0]["identity_swap_only"]
    ablation_swap = results[1]["identity_swap_only"]
    conflict_swap = results[3]["identity_swap_only"]

    feature_dep = clean_swap - ablation_swap
    trajectory_dep = ablation_swap

    print(f"\n{'Metric':<35} {'Value':>10}")
    print("-" * 45)
    print(f"{'identity_swap_only (clean)':<35} {clean_swap:>10.4f}")
    print(f"{'feature_dependency_score':<35} {feature_dep:>10.4f}")
    print(f"{'trajectory_dependency_score':<35} {trajectory_dep:>10.4f}")
    print(f"{'conflict_resolution':<35} {conflict_swap:>10.4f}")
    print(f"{'no_swap_bias_gap':<35} {results[0]['no_swap_bias_gap']:>10.4f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    if clean_swap > 0.8 and conflict_swap < 0.3:
        profile = "feature-reader-like"
        print(f"  Profile: {profile}")
        print(f"  Slot Attention achieves high clean accuracy but fails under conflict.")
        print(f"  This suggests the model reads features but lacks structural adjudication.")
    elif clean_swap > 0.8 and conflict_swap > 0.5:
        profile = "object-file-like"
        print(f"  Profile: {profile}")
        print(f"  Slot Attention maintains identity under conflict.")
        print(f"  This suggests the model has structural bias toward identity persistence.")
    elif clean_swap < 0.3:
        profile = "no-identity-binding"
        print(f"  Profile: {profile}")
        print(f"  Slot Attention cannot reliably assign identity even under clean conditions.")
    else:
        profile = "mixed"
        print(f"  Profile: {profile}")
        print(f"  Slot Attention shows partial identity binding with mixed stress test results.")

    print(f"\n  Under SVT stress tests, Slot Attention shows a {profile} profile.")


if __name__ == "__main__":
    main()
