"""
External Audit v1: All Published Models under SVT Stress Tests

Runs the full SVT stress test pipeline on four object-centric models:
  1. Slot Attention (Locatello et al., 2020)
  2. RIMs (Goyal et al., 2021)
  3. SAVi (Kipf et al., 2022)
  4. DINOSAUR (Seitzer et al., 2024)

Output: Structural fingerprint for each model + comparison table
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.slot_attention_model import SetBasedSlotAttentionModel
from models.rims_model import RIMsModel
from models.savi_model import SAViModel
from models.dinosaur_model import DINOSAURModel
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


def train_model(model, train_data, epochs=30, batch_size=64, lr=1e-3):
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


def run_stress_tests(model, model_name, clean_test, swap_test):
    results = {}

    def predict(obs_pos, obs_feat, fut_pos, fut_feat):
        model.eval()
        with torch.no_grad():
            pred = model.predict_identity(obs_pos, obs_feat, future_features=fut_feat)
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        return pred

    def eval_pred(pred, true_id, test_label):
        bd = compute_identity_breakdown(pred, true_id)
        results[test_label] = {
            "swap_only": bd["identity_swap_only"],
            "overall": bd["identity_overall"],
        }

    obs_pos_c = clean_test["observed_positions"]
    obs_feat_c = clean_test.get("object_features_obs")
    fut_pos_c = clean_test["future_positions"]
    fut_feat_c = clean_test.get("object_features_fut")
    true_id_c = clean_test["identity_labels"]

    obs_pos_s = swap_test["observed_positions"]
    obs_feat_s = swap_test.get("object_features_obs")
    fut_pos_s = swap_test["future_positions"]
    fut_feat_s = swap_test.get("object_features_fut")
    true_id_s = swap_test["identity_labels"]

    pred = predict(obs_pos_c, obs_feat_c, fut_pos_c, fut_feat_c)
    eval_pred(pred, true_id_c, "clean")

    pred = predict(obs_pos_c, None, fut_pos_c, None)
    eval_pred(pred, true_id_c, "feature_ablation")

    if obs_feat_c is not None:
        obs_feat_zero = np.zeros_like(obs_feat_c)
        fut_feat_zero = np.zeros_like(fut_feat_c)
    else:
        obs_feat_zero = None
        fut_feat_zero = None
    pred = predict(obs_pos_c, obs_feat_zero, fut_pos_c, fut_feat_zero)
    eval_pred(pred, true_id_c, "occlusion")

    if fut_feat_s is not None:
        fut_feat_conflict = fut_feat_s.copy()
        if fut_feat_conflict.ndim == 4:
            fut_feat_conflict[:, :, 0, :], fut_feat_conflict[:, :, 1, :] = \
                fut_feat_s[:, :, 1, :].copy(), fut_feat_s[:, :, 0, :].copy()
        elif fut_feat_conflict.ndim == 3:
            fut_feat_conflict[:, 0, :], fut_feat_conflict[:, 1, :] = \
                fut_feat_s[:, 1, :].copy(), fut_feat_s[:, 0, :].copy()
    else:
        fut_feat_conflict = None
    pred = predict(obs_pos_s, obs_feat_s, fut_pos_s, fut_feat_conflict)
    eval_pred(pred, true_id_s, "conflict")

    if fut_feat_c is not None:
        rng = np.random.RandomState(42)
        fut_feat_shuffled = fut_feat_c.copy()
        B = fut_feat_shuffled.shape[0]
        for b in range(B):
            if fut_feat_shuffled.ndim == 4:
                N = fut_feat_shuffled.shape[2]
                perm = rng.permutation(N)
                temp = fut_feat_shuffled[b].copy()
                for i in range(N):
                    fut_feat_shuffled[b, :, i, :] = temp[:, perm[i], :]
            elif fut_feat_shuffled.ndim == 3:
                N = fut_feat_shuffled.shape[1]
                perm = rng.permutation(N)
                temp = fut_feat_shuffled[b].copy()
                for i in range(N):
                    fut_feat_shuffled[b, i, :] = temp[perm[i], :]
    else:
        fut_feat_shuffled = None
    pred = predict(obs_pos_c, obs_feat_c, fut_pos_c, fut_feat_shuffled)
    eval_pred(pred, true_id_c, "shuffled")

    return results


def main():
    seed = 42
    print("=" * 70)
    print("External Audit v1: All Published Models under SVT Stress Tests")
    print("=" * 70)

    print("\nGenerating datasets...")
    eval_ds = generate_v3_dataset(n_train=1000, n_test=200, num_objects=2, feature_dim=2,
        feature_mode="feature_bearing", force_train_type="attractor", force_test_type="vortex",
        randomize_object_order=True, disjoint_init_split=True, seed=seed)
    train_data = gen_train(n=1000, seed=seed)
    swap_test = eval_ds["identity_test_swap_only"]
    clean_test = eval_ds["clean_test_id"]

    models_config = [
        ("Slot Attention", SetBasedSlotAttentionModel(
            t_obs=10, t_pred=20, num_objects=2, dim=2, feature_dim=2,
            slot_dim=64, n_slots=2, sa_iters=3, hidden_dim=128,
            identity_weight=1.0, temperature=1.0)),
        ("RIMs", RIMsModel(
            t_obs=10, t_pred=20, num_objects=2, dim=2, feature_dim=2,
            num_rims=2, rim_dim=64, hidden_dim=128, top_k=1,
            identity_weight=1.0, temperature=1.0)),
        ("SAVi", SAViModel(
            t_obs=10, t_pred=20, num_objects=2, dim=2, feature_dim=2,
            slot_dim=64, n_slots=2, sa_iters=3, hidden_dim=128,
            identity_weight=1.0, temperature=1.0)),
        ("DINOSAUR", DINOSAURModel(
            t_obs=10, t_pred=20, num_objects=2, dim=2, feature_dim=2,
            slot_dim=64, n_slots=2, sa_iters=3, hidden_dim=128,
            identity_weight=1.0, temperature=1.0)),
    ]

    all_results = {}

    for model_name, model in models_config:
        print(f"\n{'='*70}")
        print(f"Training: {model_name}")
        print(f"{'='*70}")
        train_model(model, train_data, epochs=30, batch_size=64, lr=1e-3)

        print(f"\nRunning SVT stress tests on {model_name}...")
        results = run_stress_tests(model, model_name, clean_test, swap_test)
        all_results[model_name] = results

        for test_name, r in results.items():
            print(f"  {test_name}: swap_only={r['swap_only']:.4f}, overall={r['overall']:.4f}")

    print(f"\n{'='*70}")
    print("STRUCTURAL FINGERPRINT COMPARISON")
    print(f"{'='*70}")

    tests = ["clean", "feature_ablation", "occlusion", "conflict", "shuffled"]
    print(f"{'Model':<20} {'Clean':>8} {'Ablat':>8} {'Occl':>8} {'Conf':>8} {'Shuff':>8} {'Profile':<20}")
    print("-" * 80)

    for model_name, results in all_results.items():
        clean_ov = results["clean"]["overall"]
        ablat_ov = results["feature_ablation"]["overall"]
        occl_ov = results["occlusion"]["overall"]
        conf_swap = results["conflict"]["swap_only"]
        shuff_ov = results["shuffled"]["overall"]

        if clean_ov > 0.8 and conf_swap < 0.3:
            profile = "feature-reader"
        elif clean_ov > 0.8 and conf_swap > 0.5:
            profile = "object-file-like"
        elif clean_ov < 0.3:
            profile = "no-binding"
        else:
            profile = "mixed"

        print(f"{model_name:<20} {clean_ov:>8.3f} {ablat_ov:>8.3f} {occl_ov:>8.3f} "
              f"{conf_swap:>8.3f} {shuff_ov:>8.3f} {profile:<20}")

    print(f"\n{'='*70}")
    print("KEY FINDING")
    print(f"{'='*70}")
    feature_readers = sum(1 for r in all_results.values()
                          if r["clean"]["overall"] > 0.8 and r["conflict"]["swap_only"] < 0.3)
    total = len(all_results)
    print(f"\n  {feature_readers}/{total} tested object-centric models show a feature-reader-like profile:")
    print(f"  high clean accuracy but complete failure under feature-trajectory conflict.")
    print(f"\n  Under SVT stress tests, these models show that clean feature matching")
    print(f"  can read out identity without constituting an object-file mechanism.")


if __name__ == "__main__":
    main()
