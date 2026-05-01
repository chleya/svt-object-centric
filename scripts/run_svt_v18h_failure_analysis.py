"""
SVT-v18h: Trajectory Scorer Failure Mode Analysis

v18g showed that the ceiling is 100% with perfect trajectory scorer.
Current GRU trajectory scorer achieves ~90%. What causes the 10% failures?

This script analyzes:
  1. Are failures concentrated in specific trajectory patterns?
  2. Do failures correlate with object proximity?
  3. Do failures correlate with trajectory curvature?
  4. Can we predict which samples will fail?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from models.dual_pathway_object_file import DualPathwayObjectFile


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


def train_model(model, train_data, epochs=60, batch_size=64, lr=1e-3, p_conflict=0.4):
    obs_pos = torch.FloatTensor(train_data["observed_positions"])
    fut_pos = torch.FloatTensor(train_data["future_positions"])
    ids = torch.LongTensor(train_data["identity_labels"])
    obs_feat = torch.FloatTensor(train_data["object_features_obs"]) if "object_features_obs" in train_data else None
    fut_feat = torch.FloatTensor(train_data["object_features_fut"]) if "object_features_fut" in train_data else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches = len(obs_pos) // batch_size
    for epoch in range(epochs):
        indices = np.random.permutation(len(obs_pos))
        for bi in range(n_batches):
            idx = indices[bi * batch_size:(bi + 1) * batch_size]
            loss, _, _, _ = model.compute_loss(
                obs_pos[idx], fut_pos[idx], ids[idx],
                observed_features=obs_feat[idx] if obs_feat is not None else None,
                future_features=fut_feat[idx] if fut_feat is not None else None,
                p_conflict=p_conflict)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def compute_trajectory_features(positions):
    B = positions.shape[0]
    T = positions.shape[1]
    N = positions.shape[2]

    features = {}
    velocities = np.zeros_like(positions)
    velocities[:, 1:, :, :] = positions[:, 1:, :, :] - positions[:, :-1, :, :]
    velocities[:, 0, :, :] = velocities[:, 1, :, :]

    speeds = np.sqrt((velocities ** 2).sum(axis=-1))
    features['mean_speed'] = speeds.mean(axis=(1, 2))
    features['speed_std'] = speeds.std(axis=(1, 2))

    if T > 2:
        acc = np.zeros_like(velocities)
        acc[:, 1:, :, :] = velocities[:, 1:, :, :] - velocities[:, :-1, :, :]
        features['mean_accel'] = np.sqrt((acc ** 2).sum(axis=-1)).mean(axis=(1, 2))

    if N >= 2:
        dist = np.sqrt(((positions[:, :, 0, :] - positions[:, :, 1, :]) ** 2).sum(axis=-1))
        features['mean_inter_object_dist'] = dist.mean(axis=1)
        features['min_inter_object_dist'] = dist.min(axis=1)

        vel_diff = velocities[:, :, 0, :] - velocities[:, :, 1, :]
        vel_sim = np.zeros(B)
        for b in range(B):
            for t in range(T):
                s1 = np.linalg.norm(velocities[b, t, 0, :])
                s2 = np.linalg.norm(velocities[b, t, 1, :])
                if s1 > 1e-6 and s2 > 1e-6:
                    vel_sim[b] += np.dot(velocities[b, t, 0, :], velocities[b, t, 1, :]) / (s1 * s2)
            vel_sim[b] /= T
        features['velocity_similarity'] = vel_sim

    if T > 2:
        curvature = np.zeros((B, N))
        for j in range(N):
            for b in range(B):
                curvs = []
                for t in range(1, T - 1):
                    v1 = positions[b, t, j, :] - positions[b, t-1, j, :]
                    v2 = positions[b, t+1, j, :] - positions[b, t, j, :]
                    cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
                    s1 = np.linalg.norm(v1)
                    s2 = np.linalg.norm(v2)
                    if s1 > 1e-6 and s2 > 1e-6:
                        curvs.append(cross / (s1 * s2))
                if curvs:
                    curvature[b, j] = np.mean(curvs)
        features['mean_curvature'] = curvature.mean(axis=1)

    return features


def main():
    seed = 42
    print("Generating datasets...")
    eval_ds = generate_v3_dataset(n_train=1000, n_test=200, num_objects=2, feature_dim=2,
        feature_mode="feature_bearing", force_train_type="attractor", force_test_type="vortex",
        randomize_object_order=True, disjoint_init_split=True, seed=seed)
    train_data = gen_train(n=1000, seed=seed)
    swap_test = eval_ds["identity_test_swap_only"]
    clean_test = eval_ds["clean_test_id"]

    print("Training model...")
    model = DualPathwayObjectFile(
        num_objects=2, feature_dim=2, hidden_dim=128, slot_dim=64,
        t_obs=10, t_pred=20, conflict_switch_temp=0.1)
    train_model(model, train_data, epochs=60, p_conflict=0.4)

    obs_pos = swap_test["observed_positions"]
    obs_feat = swap_test.get("object_features_obs")
    fut_pos = swap_test["future_positions"]
    fut_feat = swap_test.get("object_features_fut")
    true_id = swap_test["identity_labels"]

    traj_pred = model.predict_identity(
        obs_pos, obs_feat, future_positions=fut_pos,
        future_features=fut_feat, method="trajectory_only")
    if isinstance(traj_pred, torch.Tensor):
        traj_pred = traj_pred.cpu().numpy()

    correct = np.all(traj_pred == true_id, axis=1)
    print(f"\nTrajectory scorer accuracy: {correct.mean():.4f}")
    print(f"  Correct: {correct.sum()}, Wrong: {(~correct).sum()}")

    obs_features = compute_trajectory_features(obs_pos)
    fut_features = compute_trajectory_features(fut_pos)

    print("\n--- Failure Mode Analysis ---")
    print(f"\n  Feature comparison (correct vs wrong):")
    for fname in obs_features:
        c_vals = obs_features[fname][correct]
        w_vals = obs_features[fname][~correct]
        if len(w_vals) > 0:
            print(f"    obs_{fname:30s}: correct={c_vals.mean():.4f}, wrong={w_vals.mean():.4f}, "
                  f"diff={w_vals.mean()-c_vals.mean():+.4f}")

    for fname in fut_features:
        c_vals = fut_features[fname][correct]
        w_vals = fut_features[fname][~correct]
        if len(w_vals) > 0:
            print(f"    fut_{fname:30s}: correct={c_vals.mean():.4f}, wrong={w_vals.mean():.4f}, "
                  f"diff={w_vals.mean()-c_vals.mean():+.4f}")

    print("\n--- Correlation with accuracy ---")
    all_features = {}
    for k, v in obs_features.items():
        all_features[f"obs_{k}"] = v
    for k, v in fut_features.items():
        all_features[f"fut_{k}"] = v

    for fname, fvals in sorted(all_features.items()):
        if len(fvals) != len(correct):
            continue
        try:
            corr = np.corrcoef(fvals, correct.astype(float))[0, 1]
            if abs(corr) > 0.05:
                print(f"    {fname:35s}: r={corr:+.4f}")
        except:
            pass

    print("\n--- Proximity Analysis ---")
    min_dist = obs_features.get('min_inter_object_dist', np.zeros(len(obs_pos)))
    dist_bins = [0, 5, 10, 20, 50, 100]
    for i in range(len(dist_bins) - 1):
        mask = (min_dist >= dist_bins[i]) & (min_dist < dist_bins[i+1])
        if mask.sum() > 0:
            acc = correct[mask].mean()
            print(f"    min_dist [{dist_bins[i]:3d}, {dist_bins[i+1]:3d}): n={mask.sum():4d}, acc={acc:.4f}")

    print("\n--- Curvature Analysis ---")
    if 'mean_curvature' in obs_features:
        curv = obs_features['mean_curvature']
        curv_bins = [0, 0.01, 0.05, 0.1, 0.2, 1.0]
        for i in range(len(curv_bins) - 1):
            mask = (curv >= curv_bins[i]) & (curv < curv_bins[i+1])
            if mask.sum() > 0:
                acc = correct[mask].mean()
                print(f"    curvature [{curv_bins[i]:.2f}, {curv_bins[i+1]:.2f}): n={mask.sum():4d}, acc={acc:.4f}")

    print("\n--- Velocity Similarity Analysis ---")
    if 'velocity_similarity' in obs_features:
        vsim = obs_features['velocity_similarity']
        vsim_bins = [-1.0, -0.5, 0.0, 0.5, 0.8, 1.0]
        for i in range(len(vsim_bins) - 1):
            mask = (vsim >= vsim_bins[i]) & (vsim < vsim_bins[i+1])
            if mask.sum() > 0:
                acc = correct[mask].mean()
                print(f"    vel_sim [{vsim_bins[i]:.1f}, {vsim_bins[i+1]:.1f}): n={mask.sum():4d}, acc={acc:.4f}")

    print("\n" + "="*70)
    print("v18h FAILURE MODE SUMMARY")
    print("="*70)
    print("  Key question: What makes trajectory scoring fail?")
    print("  - If proximity is the main factor: need better trajectory disentangling")
    print("  - If curvature is the main factor: need better dynamics modeling")
    print("  - If velocity similarity is the main factor: need trajectory-specific features")


if __name__ == "__main__":
    main()
