import numpy as np
from typing import Optional


def predict_future_oracle(
    observed_positions: np.ndarray,
    observed_velocities: np.ndarray,
    t_pred: int = 10,
    dt: float = 1.0,
    width: float = 64.0,
    height: float = 64.0,
    object_radius: float = 1.5,
    gravity: float = 0.0,
    friction: float = 0.0,
    acceleration_noise: float = 0.0,
    toroidal: bool = False,
):
    T_obs, N, _ = observed_positions.shape
    future_positions = np.zeros((t_pred, N, 2))
    future_velocities = np.zeros((t_pred, N, 2))

    pos = observed_positions[-1].copy()
    vel = observed_velocities[-1].copy()

    for t in range(t_pred):
        if friction > 0:
            vel *= (1.0 - friction * dt)

        vel[:, 1] += gravity * dt

        new_pos = pos + vel * dt

        if toroidal:
            new_pos[:, 0] = new_pos[:, 0] % width
            new_pos[:, 1] = new_pos[:, 1] % height
        else:
            for i in range(N):
                for dim, limit in [(0, width), (1, height)]:
                    if new_pos[i, dim] < object_radius:
                        new_pos[i, dim] = object_radius
                        vel[i, dim] = abs(vel[i, dim]) * 0.9
                    elif new_pos[i, dim] > limit - object_radius:
                        new_pos[i, dim] = limit - object_radius
                        vel[i, dim] = -abs(vel[i, dim]) * 0.9

        future_positions[t] = new_pos
        future_velocities[t] = vel.copy()
        pos = new_pos

    return future_positions, future_velocities


def predict_identity_oracle(
    observed_positions: np.ndarray,
    observed_velocities: np.ndarray,
    true_identity_labels: Optional[np.ndarray] = None,
):
    if true_identity_labels is not None:
        return true_identity_labels

    if observed_positions.ndim == 3:
        N = observed_positions.shape[1]
        return np.arange(N)
    else:
        B = observed_positions.shape[0]
        N = observed_positions.shape[2]
        return np.tile(np.arange(N), (B, 1))
