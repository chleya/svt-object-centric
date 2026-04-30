import numpy as np
from typing import Optional


def simulate_episode(
    num_objects: int = 2,
    width: float = 64.0,
    height: float = 64.0,
    t_obs: int = 10,
    t_pred: int = 10,
    dt: float = 1.0,
    velocity_range=(-2.0, 2.0),
    position_range=(5.0, 59.0),
    object_radius: float = 1.5,
    allow_occlusion: bool = True,
    allow_crossing: bool = True,
    allow_hidden_perturbation: bool = True,
    occlusion_radius: float = 3.0,
    hidden_perturbation_strength: float = 0.5,
    gravity: float = 0.0,
    friction: float = 0.0,
    acceleration_noise: float = 0.0,
    toroidal: bool = False,
    rng: Optional[np.random.RandomState] = None,
    counterfactual: bool = False,
    cf_velocity_delta: Optional[np.ndarray] = None,
    identity_test: bool = False,
):
    if rng is None:
        rng = np.random.RandomState()

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
    hidden_perturbation_applied = np.zeros((T, N), dtype=bool)

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
            v = velocities[t - 1, i].copy()

            if counterfactual and cf_velocity_delta is not None:
                v += cf_velocity_delta[i]

            if friction > 0:
                v *= (1.0 - friction * dt)

            v[1] += gravity * dt

            if acceleration_noise > 0:
                v += rng.randn(2) * acceleration_noise * np.sqrt(dt)

            new_pos = positions[t - 1, i] + v * dt

            if toroidal:
                new_pos[0] = new_pos[0] % width
                new_pos[1] = new_pos[1] % height
            else:
                for dim, limit in [(0, width), (1, height)]:
                    if new_pos[dim] < object_radius:
                        new_pos[dim] = object_radius
                        v[dim] = abs(v[dim]) * 0.9
                    elif new_pos[dim] > limit - object_radius:
                        new_pos[dim] = limit - object_radius
                        v[dim] = -abs(v[dim]) * 0.9

            positions[t, i] = new_pos
            velocities[t, i] = v

        if allow_crossing and N >= 2:
            dist = np.linalg.norm(positions[t, 0] - positions[t, 1])
            if dist < object_radius * 2.5:
                crossing_flags[t] = True

        if occlusion_time_range is not None:
            if occlusion_time_range[0] <= t <= occlusion_time_range[1]:
                for i in range(N):
                    dist_to_oc = np.linalg.norm(positions[t, i] - occlusion_center)
                    if dist_to_oc < occlusion_radius:
                        occlusion_mask[t, i] = True

                        if (
                            allow_hidden_perturbation
                            and t == occlusion_time_range[0]
                            and not hidden_perturbation_applied[t, i]
                        ):
                            perturb = rng.randn(2) * hidden_perturbation_strength
                            velocities[t, i] += perturb
                            hidden_perturbation_applied[t, i] = True

    if identity_test and N >= 2 and allow_occlusion:
        if rng.random() < 0.5:
            swap_t = occlusion_time_range[1] if occlusion_time_range else t_obs - 1
            if swap_t < T:
                pos_0 = positions[swap_t:, 0].copy()
                pos_1 = positions[swap_t:, 1].copy()
                positions[swap_t:, 0] = pos_1
                positions[swap_t:, 1] = pos_0
                identity_labels = np.array([1, 0])

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
        "hidden_perturbation_applied": hidden_perturbation_applied,
        "occlusion_center": occlusion_center,
        "occlusion_time_range": occlusion_time_range,
        "identity_labels": identity_labels,
        "counterfactual": counterfactual,
        "identity_test": identity_test,
    }

    return observed, future, metadata
