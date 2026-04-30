"""
SVT-v3 Nonlinear Force World

2D arena with configurable force fields (attractor, repulsor, vortex, gravity_well).
Supports feature-bearing objects, occlusion, crossing, swap, and OOD force transfer.
"""

import numpy as np
from typing import Optional, Dict


FORCE_TYPES = ["attractor", "repulsor", "vortex", "gravity_well"]


def compute_force(
    position: np.ndarray,
    velocity: np.ndarray,
    force_type: str,
    field_center: np.ndarray,
    field_strength: float,
    softening: float = 1.0,
) -> np.ndarray:
    delta = field_center - position
    dist = np.linalg.norm(delta) + softening
    direction = delta / dist

    if force_type == "attractor":
        return field_strength * direction / (dist ** 0.5)
    elif force_type == "repulsor":
        return -field_strength * direction / (dist ** 0.5)
    elif force_type == "vortex":
        perp = np.array([-direction[1], direction[0]])
        return field_strength * perp / (dist ** 0.5)
    elif force_type == "gravity_well":
        return field_strength * direction / (dist ** 1.5)
    else:
        return np.zeros(2)


def simulate_nonlinear_episode(
    num_objects: int = 2,
    width: float = 64.0,
    height: float = 64.0,
    t_obs: int = 10,
    t_pred: int = 20,
    dt: float = 1.0,
    velocity_range=(-2.0, 2.0),
    position_range=(5.0, 59.0),
    object_radius: float = 1.5,
    allow_occlusion: bool = True,
    allow_crossing: bool = True,
    force_type: str = "attractor",
    field_strength: float = 0.5,
    field_center: Optional[np.ndarray] = None,
    damping: float = 0.95,
    noise_std: float = 0.1,
    softening: float = 1.0,
    boundary_mode: str = "bounce",
    identity_test: bool = False,
    rng: Optional[np.random.RandomState] = None,
) -> Dict:
    if rng is None:
        rng = np.random.RandomState()

    if field_center is None:
        field_center = np.array([width / 2, height / 2])

    T = t_obs + t_pred
    N = num_objects

    positions = np.zeros((T, N, 2))
    velocities = np.zeros((T, N, 2))
    identity_labels = np.arange(N)

    for i in range(N):
        positions[0, i, 0] = rng.uniform(*position_range)
        positions[0, i, 1] = rng.uniform(*position_range)
        velocities[0, i, 0] = rng.uniform(*velocity_range)
        velocities[0, i, 1] = rng.uniform(*velocity_range)

    occlusion_mask = np.zeros((T, N), dtype=bool)
    crossing_flags = np.zeros(T, dtype=bool)

    occlusion_center = None
    occlusion_time_range = None

    if allow_occlusion and rng.random() < 0.5:
        mid_t = t_obs // 2
        oc_t = rng.randint(max(1, mid_t - 2), min(t_obs - 1, mid_t + 3))
        occlusion_time_range = (oc_t, oc_t + 2)
        cx = np.mean(positions[oc_t, :, 0])
        cy = np.mean(positions[oc_t, :, 1])
        occlusion_center = np.array([cx, cy])

    for t in range(1, T):
        for i in range(N):
            force = compute_force(
                positions[t - 1, i], velocities[t - 1, i],
                force_type, field_center, field_strength, softening,
            )

            new_vel = damping * velocities[t - 1, i] + force * dt
            if noise_std > 0:
                new_vel += rng.randn(2) * noise_std * np.sqrt(dt)

            new_pos = positions[t - 1, i] + new_vel * dt

            if boundary_mode == "toroidal":
                new_pos[0] = new_pos[0] % width
                new_pos[1] = new_pos[1] % height
            else:
                for dim, limit in [(0, width), (1, height)]:
                    if new_pos[dim] < object_radius:
                        new_pos[dim] = object_radius
                        new_vel[dim] = abs(new_vel[dim]) * 0.9
                    elif new_pos[dim] > limit - object_radius:
                        new_pos[dim] = limit - object_radius
                        new_vel[dim] = -abs(new_vel[dim]) * 0.9

            positions[t, i] = new_pos
            velocities[t, i] = new_vel

        if allow_crossing and N >= 2:
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.linalg.norm(positions[t, i] - positions[t, j])
                    if dist < object_radius * 2.5:
                        crossing_flags[t] = True
                        break

        if occlusion_time_range is not None:
            if occlusion_time_range[0] <= t <= occlusion_time_range[1]:
                occlusion_radius = 3.0
                for i in range(N):
                    dist_to_oc = np.linalg.norm(positions[t, i] - occlusion_center)
                    if dist_to_oc < occlusion_radius:
                        occlusion_mask[t, i] = True

    if identity_test and N >= 2 and allow_occlusion and occlusion_time_range is not None:
        if rng.random() < 0.5:
            swap_t = occlusion_time_range[1]
            if swap_t < T:
                swap_pair = rng.choice(N, size=2, replace=False)
                i, j = swap_pair
                pos_i = positions[swap_t:, i].copy()
                pos_j = positions[swap_t:, j].copy()
                positions[swap_t:, i] = pos_j
                positions[swap_t:, j] = pos_i
                identity_labels = identity_labels.copy()
                identity_labels[i], identity_labels[j] = identity_labels[j], identity_labels[i]

    observed = {
        "positions": positions[:t_obs],
        "velocities": velocities[:t_obs],
        "occlusion_mask": occlusion_mask[:t_obs],
    }
    future = {
        "positions": positions[t_obs:],
        "velocities": velocities[t_obs:],
        "occlusion_mask": occlusion_mask[t_obs:],
    }
    metadata = {
        "crossing_flags": crossing_flags,
        "occlusion_center": occlusion_center,
        "occlusion_time_range": occlusion_time_range,
        "identity_labels": identity_labels,
        "identity_test": identity_test,
        "force_type": force_type,
        "field_strength": field_strength,
        "field_center": field_center,
        "damping": damping,
        "noise_std": noise_std,
    }

    return observed, future, metadata
