import numpy as np
from typing import Dict, Optional


def _toroidal_delta(delta: np.ndarray, size: float) -> np.ndarray:
    return np.where(delta > size / 2, delta - size,
                    np.where(delta < -size / 2, delta + size, delta))


def compute_identity_accuracy(pred_ids: np.ndarray, true_ids: np.ndarray) -> float:
    if pred_ids.shape != true_ids.shape:
        if pred_ids.ndim == 1 and true_ids.ndim == 2:
            pred_ids = np.tile(pred_ids, (true_ids.shape[0], 1))
    return float(np.mean(pred_ids == true_ids))


def compute_identity_skill(
    pred_accuracy: float, random_accuracy: float
) -> float:
    if random_accuracy >= 1.0:
        return 0.0
    return (pred_accuracy - random_accuracy) / (1.0 - random_accuracy)


def velocity_continuity_identity(
    observed_positions: np.ndarray,
    future_positions: Optional[np.ndarray] = None,
    observed_velocities: Optional[np.ndarray] = None,
    future_velocities: Optional[np.ndarray] = None,
    toroidal: bool = False,
    width: float = 64.0,
    height: float = 64.0,
) -> np.ndarray:
    if observed_positions.ndim == 3:
        observed_positions = observed_positions[np.newaxis]
        if future_positions is not None:
            future_positions = future_positions[np.newaxis]
    B, T_obs, N, D = observed_positions.shape
    pred_ids = np.tile(np.arange(N), (B, 1))

    if N != 2:
        return pred_ids

    last_obs_vel = observed_positions[:, -1] - observed_positions[:, -2]

    if future_positions is not None:
        if toroidal:
            raw_diff = future_positions[:, 0] - observed_positions[:, -1]
            first_fut_vel = np.stack([
                _toroidal_delta(raw_diff[:, :, 0], width),
                _toroidal_delta(raw_diff[:, :, 1], height),
            ], axis=-1)
        else:
            first_fut_vel = future_positions[:, 0] - observed_positions[:, -1]
    elif future_velocities is not None:
        if future_velocities.ndim == 3:
            future_velocities = future_velocities[np.newaxis]
        first_fut_vel = future_velocities[:, 0]
    else:
        return pred_ids

    for i in range(B):
        dist_no_swap = (np.linalg.norm(last_obs_vel[i, 0] - first_fut_vel[i, 0]) +
                        np.linalg.norm(last_obs_vel[i, 1] - first_fut_vel[i, 1]))
        dist_swap = (np.linalg.norm(last_obs_vel[i, 0] - first_fut_vel[i, 1]) +
                     np.linalg.norm(last_obs_vel[i, 1] - first_fut_vel[i, 0]))

        if dist_swap < dist_no_swap:
            pred_ids[i] = np.array([1, 0])

    return pred_ids


def nearest_neighbor_identity(
    observed_positions: np.ndarray,
    observed_velocities: np.ndarray,
    future_positions: np.ndarray,
) -> np.ndarray:
    T_obs, N, _ = observed_positions.shape
    last_pos = observed_positions[-1]
    first_future_pos = future_positions[0]

    predicted_ids = np.arange(N)

    if N == 2:
        dist_no_swap = np.linalg.norm(last_pos[0] - first_future_pos[0]) + np.linalg.norm(
            last_pos[1] - first_future_pos[1]
        )
        dist_swap = np.linalg.norm(last_pos[0] - first_future_pos[1]) + np.linalg.norm(
            last_pos[1] - first_future_pos[0]
        )

        if dist_swap < dist_no_swap:
            predicted_ids = np.array([1, 0])

    return predicted_ids


def velocity_extrapolation_identity(
    observed_velocities: np.ndarray,
    future_velocities: np.ndarray,
) -> np.ndarray:
    T_obs, N, _ = observed_velocities.shape
    predicted_ids = np.arange(N)

    if N == 2:
        last_vel = observed_velocities[-1]
        first_fut_vel = future_velocities[0]

        dist_no_swap = np.linalg.norm(last_vel[0] - first_fut_vel[0]) + np.linalg.norm(
            last_vel[1] - first_fut_vel[1]
        )
        dist_swap = np.linalg.norm(last_vel[0] - first_fut_vel[1]) + np.linalg.norm(
            last_vel[1] - first_fut_vel[0]
        )

        if dist_swap < dist_no_swap:
            predicted_ids = np.array([1, 0])

    return predicted_ids
