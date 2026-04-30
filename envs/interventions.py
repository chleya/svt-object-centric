import numpy as np
from typing import Optional


def apply_counterfactual_intervention(
    observed: dict,
    future: dict,
    metadata: dict,
    velocity_delta: Optional[np.ndarray] = None,
    rng: Optional[np.random.RandomState] = None,
):
    if rng is None:
        rng = np.random.RandomState()
    if velocity_delta is None:
        N = observed["positions"].shape[1]
        velocity_delta = rng.randn(N, 2) * 1.0

    from envs.motion_world import simulate_episode

    return velocity_delta


def apply_compositional_intervention(
    observed: dict,
    future: dict,
    metadata: dict,
    swap_objects: bool = True,
):
    new_future = {
        "positions": future["positions"].copy(),
        "velocities": future["velocities"].copy(),
        "occlusion_mask": future["occlusion_mask"].copy(),
    }
    if swap_objects and new_future["positions"].shape[1] >= 2:
        new_future["positions"][:, [0, 1]] = new_future["positions"][:, [1, 0]]
        new_future["velocities"][:, [0, 1]] = new_future["velocities"][:, [1, 0]]
    return new_future
