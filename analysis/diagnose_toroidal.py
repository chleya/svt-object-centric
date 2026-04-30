import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from metrics.identity_metrics import velocity_continuity_identity, compute_identity_accuracy

d = np.load('data_toroidal/identity_test.npz')
obs = d['observed_positions']
fut = d['future_positions']
ids = d['identity_labels']

swap_mask = ids[:, 0] == 1
n_swap = swap_mask.sum()
n_noswap = (~swap_mask).sum()
print(f"Swapped: {n_swap}, Not swapped: {n_noswap}")

vel_ids = velocity_continuity_identity(
    obs, future_positions=fut,
    toroidal=True, width=64.0, height=64.0,
)
vel_acc = compute_identity_accuracy(vel_ids, ids)
print(f"Overall Vel-ID accuracy: {vel_acc:.3f}")

swap_acc = compute_identity_accuracy(vel_ids[swap_mask], ids[swap_mask])
noswap_acc = compute_identity_accuracy(vel_ids[~swap_mask], ids[~swap_mask])
print(f"  On swap episodes: {swap_acc:.3f}")
print(f"  On no-swap episodes: {noswap_acc:.3f}")

last_obs_vel = obs[:, -1] - obs[:, -2]
raw_diff = fut[:, 0] - obs[:, -1]
first_fut_vel = np.stack([
    np.where(raw_diff[:, :, 0] > 32, raw_diff[:, :, 0] - 64,
             np.where(raw_diff[:, :, 0] < -32, raw_diff[:, :, 0] + 64, raw_diff[:, :, 0])),
    np.where(raw_diff[:, :, 1] > 32, raw_diff[:, :, 1] - 64,
             np.where(raw_diff[:, :, 1] < -32, raw_diff[:, :, 1] + 64, raw_diff[:, :, 1])),
], axis=-1)

for i in range(min(10, len(obs))):
    is_swap = ids[i, 0] == 1
    d_ns = (np.linalg.norm(last_obs_vel[i, 0] - first_fut_vel[i, 0]) +
            np.linalg.norm(last_obs_vel[i, 1] - first_fut_vel[i, 1]))
    d_s = (np.linalg.norm(last_obs_vel[i, 0] - first_fut_vel[i, 1]) +
           np.linalg.norm(last_obs_vel[i, 1] - first_fut_vel[i, 0]))
    predicted_swap = d_s < d_ns
    correct = (predicted_swap == is_swap)
    print(f"  Ep {i}: swap={is_swap}, d_ns={d_ns:.3f}, d_s={d_s:.3f}, pred_swap={predicted_swap}, correct={correct}")
