"""
SVT-v18g: Oracle Trajectory Analysis — Performance Ceiling

The current bottleneck is trajectory scorer quality (0.67-0.92).
v18g asks: if we had a PERFECT trajectory scorer, what would be
the ceiling for conditional binding?

Approach: Replace the learned trajectory scorer with an oracle that
uses ground-truth trajectory information directly. This gives us
the theoretical maximum for the dual-pathway approach.

Oracle trajectory scorer:
  - Uses ground-truth observed positions (not encoded)
  - Computes Euclidean distance between predicted and actual positions
  - Assigns identity by minimum distance (nearest-neighbor)

This tells us:
  1. The ceiling for dual-pathway conditional binding
  2. How much of the gap is due to trajectory encoding vs trajectory prediction
  3. Whether the agreement-based switch works perfectly with oracle trajectory
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.dual_pathway_object_file import DualPathwayObjectFile
from scipy.optimize import linear_sum_assignment


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


def oracle_trajectory_assignment(obs_positions, fut_positions, identity_labels):
    B = obs_positions.shape[0]
    N = obs_positions.shape[2]
    predictions = np.zeros_like(fut_positions)

    for b in range(B):
        for j in range(N):
            last_pos = obs_positions[b, -1, j, :]
            last_vel = obs_positions[b, -1, j, :] - obs_positions[b, -2, j, :]
            for t in range(fut_positions.shape[1]):
                predictions[b, t, j, :] = last_pos + last_vel * (t + 1)

    cost_matrix = np.zeros((B, N, N))
    for b in range(B):
        for i in range(N):
            for j in range(N):
                cost_matrix[b, i, j] = np.mean(
                    (fut_positions[b, :, i, :] - predictions[b, :, j, :]) ** 2)

    pred = np.zeros((B, N), dtype=int)
    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(cost_matrix[b])
        for i, j in zip(row_ind, col_ind):
            pred[b, i] = j

    return pred


def oracle_perfect_assignment(identity_labels):
    return identity_labels.copy()


def swap_features(fut_feat):
    if fut_feat is None:
        return None
    swapped = fut_feat.copy()
    if swapped.ndim == 4:
        swapped[:, :, 0, :], swapped[:, :, 1, :] = fut_feat[:, :, 1, :].copy(), fut_feat[:, :, 0, :].copy()
    elif swapped.ndim == 3:
        swapped[:, 0, :], swapped[:, 1, :] = fut_feat[:, 1, :].copy(), fut_feat[:, 0, :].copy()
    return swapped


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


def combined_oracle_predict(feat_pred, oracle_pred, method="agreement"):
    B = feat_pred.shape[0]
    N = feat_pred.shape[1]

    if method == "agreement":
        agree = np.all(feat_pred == oracle_pred, axis=1)
        combined = np.where(agree[:, np.newaxis], feat_pred, oracle_pred)
        return combined
    elif method == "always_oracle":
        return oracle_pred
    elif method == "always_feat":
        return feat_pred


def main():
    seed = 42
    print("Generating datasets...")
    eval_ds = generate_v3_dataset(n_train=1000, n_test=200, num_objects=2, feature_dim=2,
        feature_mode="feature_bearing", force_train_type="attractor", force_test_type="vortex",
        randomize_object_order=True, disjoint_init_split=True, seed=seed)
    train_data = gen_train(n=1000, seed=seed)
    swap_test = eval_ds["identity_test_swap_only"]
    clean_test = eval_ds["clean_test_id"]

    print("\n" + "="*70)
    print("ORACLE TRAJECTORY ANALYSIS")
    print("="*70)

    obs_pos_s = swap_test["observed_positions"]
    obs_feat_s = swap_test.get("object_features_obs")
    fut_pos_s = swap_test["future_positions"]
    fut_feat_s = swap_test.get("object_features_fut")
    true_id_s = swap_test["identity_labels"]

    print("\n--- Oracle Trajectory Scorer (linear extrapolation) ---")
    oracle_traj_pred = oracle_trajectory_assignment(obs_pos_s, fut_pos_s, true_id_s)
    bd_oracle = compute_identity_breakdown(oracle_traj_pred, true_id_s)
    print(f"  Oracle trajectory accuracy (swap): {bd_oracle['identity_swap_only']:.4f}")

    fut_feat_conflict = swap_features(fut_feat_s)
    oracle_traj_conflict = oracle_trajectory_assignment(obs_pos_s, fut_pos_s, true_id_s)
    bd_oracle_c = compute_identity_breakdown(oracle_traj_conflict, true_id_s)
    print(f"  Oracle trajectory accuracy (conflict): {bd_oracle_c['identity_swap_only']:.4f}")

    print("\n--- Oracle Perfect Scorer (ground truth identity) ---")
    oracle_perfect = oracle_perfect_assignment(true_id_s)
    bd_perfect = compute_identity_breakdown(oracle_perfect, true_id_s)
    print(f"  Perfect scorer accuracy (swap): {bd_perfect['identity_swap_only']:.4f}")

    print("\n--- Feature Scorer (from trained model) ---")
    model = DualPathwayObjectFile(
        num_objects=2, feature_dim=2, hidden_dim=128, slot_dim=64,
        t_obs=10, t_pred=20, conflict_switch_temp=0.1)
    train_model(model, train_data, epochs=60, p_conflict=0.4)

    feat_pred_swap = model.predict_identity(
        obs_pos_s, obs_feat_s, future_positions=fut_pos_s,
        future_features=fut_feat_s, method="feature_only")
    if isinstance(feat_pred_swap, torch.Tensor):
        feat_pred_swap = feat_pred_swap.cpu().numpy()
    bd_feat = compute_identity_breakdown(feat_pred_swap, true_id_s)
    print(f"  Feature scorer accuracy (swap): {bd_feat['identity_swap_only']:.4f}")

    feat_pred_conflict = model.predict_identity(
        obs_pos_s, obs_feat_s, future_positions=fut_pos_s,
        future_features=fut_feat_conflict, method="feature_only")
    if isinstance(feat_pred_conflict, torch.Tensor):
        feat_pred_conflict = feat_pred_conflict.cpu().numpy()
    bd_feat_c = compute_identity_breakdown(feat_pred_conflict, true_id_s)
    print(f"  Feature scorer accuracy (conflict): {bd_feat_c['identity_swap_only']:.4f}")

    print("\n--- Combined: Feature + Oracle Trajectory ---")
    for method in ["agreement", "always_oracle"]:
        combined_swap = combined_oracle_predict(feat_pred_swap, oracle_traj_pred, method=method)
        bd_comb = compute_identity_breakdown(combined_swap, true_id_s)

        combined_conflict = combined_oracle_predict(feat_pred_conflict, oracle_traj_conflict, method=method)
        bd_comb_c = compute_identity_breakdown(combined_conflict, true_id_s)

        print(f"  {method}: swap={bd_comb['identity_swap_only']:.4f}, conflict={bd_comb_c['identity_swap_only']:.4f}")

    print("\n--- Combined: Feature + Oracle Perfect ---")
    for method in ["agreement", "always_oracle"]:
        combined_swap = combined_oracle_predict(feat_pred_swap, oracle_perfect, method=method)
        bd_comb = compute_identity_breakdown(combined_swap, true_id_s)

        combined_conflict = combined_oracle_predict(feat_pred_conflict, oracle_perfect, method=method)
        bd_comb_c = compute_identity_breakdown(combined_conflict, true_id_s)

        print(f"  {method}: swap={bd_comb['identity_swap_only']:.4f}, conflict={bd_comb_c['identity_swap_only']:.4f}")

    print("\n--- Agreement Analysis ---")
    agree_feat_oracle = np.all(feat_pred_swap == oracle_traj_pred, axis=1)
    print(f"  Feature-Oracle agreement (swap): {agree_feat_oracle.mean():.4f}")

    agree_feat_oracle_c = np.all(feat_pred_conflict == oracle_traj_conflict, axis=1)
    print(f"  Feature-Oracle agreement (conflict): {agree_feat_oracle_c.mean():.4f}")

    print("\n--- Error Decomposition ---")
    learned_traj_pred = model.predict_identity(
        obs_pos_s, obs_feat_s, future_positions=fut_pos_s,
        future_features=fut_feat_s, method="trajectory_only")
    if isinstance(learned_traj_pred, torch.Tensor):
        learned_traj_pred = learned_traj_pred.cpu().numpy()

    learned_correct = np.all(learned_traj_pred == true_id_s, axis=1)
    oracle_correct = np.all(oracle_traj_pred == true_id_s, axis=1)

    both_correct = learned_correct & oracle_correct
    learned_only = learned_correct & ~oracle_correct
    oracle_only = oracle_correct & ~learned_correct
    both_wrong = ~learned_correct & ~oracle_correct

    print(f"  Both correct:      {both_correct.mean():.4f}")
    print(f"  Learned only:      {learned_only.mean():.4f}")
    print(f"  Oracle only:       {oracle_only.mean():.4f}")
    print(f"  Both wrong:        {both_wrong.mean():.4f}")
    print(f"  Oracle ceiling:    {oracle_correct.mean():.4f}")
    print(f"  Learned traj acc:  {learned_correct.mean():.4f}")
    print(f"  Gap to oracle:     {oracle_correct.mean() - learned_correct.mean():.4f}")

    print("\n" + "="*70)
    print("v18g CEILING ANALYSIS")
    print("="*70)
    print(f"  Current dual-pathway (learned traj):  swap=0.88, conflict=0.88")
    print(f"  Oracle trajectory ceiling:            swap={bd_oracle['identity_swap_only']:.4f}, conflict={bd_oracle_c['identity_swap_only']:.4f}")
    print(f"  Perfect trajectory ceiling:           swap=1.000, conflict=1.000")
    print(f"  Gap to oracle:                        {oracle_correct.mean() - learned_correct.mean():.4f}")
    print(f"  => Improving trajectory encoder can close {(oracle_correct.mean() - learned_correct.mean()) / (1.0 - learned_correct.mean()) * 100:.1f}% of the gap to perfect")


if __name__ == "__main__":
    main()
