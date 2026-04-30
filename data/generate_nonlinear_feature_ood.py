"""
SVT-v3 Data Generator: Nonlinear Feature-Bearing OOD Benchmark

Generates datasets with:
- Nonlinear force fields (attractor, repulsor, vortex, gravity_well)
- Feature-bearing / featureless objects
- Swap-only / no-swap-only / mixed identity splits
- OOD force transfer test
- Crossing/occlusion test
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.nonlinear_force_world import simulate_nonlinear_episode


def _generate_single_episode(
    t_obs=10, t_pred=20, num_objects=2, arena_size=64.0,
    feature_mode="feature_bearing",
    feature_dim=2,
    randomize_object_order=True,
    identity_test=False,
    swap_probability=0.5,
    force_type="attractor",
    field_strength=0.5,
    field_center=None,
    damping=0.95,
    noise_std=0.1,
    softening=1.0,
    boundary_mode="bounce",
    allow_occlusion=True,
    allow_crossing=True,
    force_crossing_occlusion=False,
    rng=None,
    max_retries=20,
):
    if rng is None:
        rng = np.random.RandomState()

    force_swap = identity_test and swap_probability >= 1.0
    force_no_swap = identity_test and swap_probability <= 0.0

    for _attempt in range(max_retries + 1):
        observed, future, metadata = simulate_nonlinear_episode(
            num_objects=num_objects,
            width=arena_size, height=arena_size,
            t_obs=t_obs, t_pred=t_pred,
            dt=1.0,
            velocity_range=(-2.5, 2.5),
            position_range=(5.0, arena_size - 5.0),
            object_radius=1.5,
            allow_occlusion=allow_occlusion,
            allow_crossing=allow_crossing,
            force_type=force_type,
            field_strength=field_strength,
            field_center=field_center,
            damping=damping,
            noise_std=noise_std,
            softening=softening,
            boundary_mode=boundary_mode,
            identity_test=identity_test,
            rng=rng,
        )

        identity_labels = metadata["identity_labels"]
        swap_happened = not np.array_equal(identity_labels, np.arange(num_objects))

        if force_swap and not swap_happened:
            continue
        if force_no_swap and swap_happened:
            continue
        if force_crossing_occlusion:
            has_crossing = bool(metadata["crossing_flags"].any())
            has_occlusion = metadata["occlusion_time_range"] is not None
            if not (has_crossing or has_occlusion):
                continue
        break

    occlusion_time_range = metadata.get("occlusion_time_range")
    has_occlusion = occlusion_time_range is not None
    has_crossing = bool(metadata["crossing_flags"].any())

    if feature_mode == "feature_bearing":
        T_obs = t_obs
        T_pred = t_pred
        Fd = feature_dim
        object_features_obs = np.zeros((T_obs, num_objects, Fd))
        object_features_fut = np.zeros((T_pred, num_objects, Fd))

        if Fd == num_objects:
            if swap_happened:
                swap_t = occlusion_time_range[1] if occlusion_time_range else t_obs - 1
                swap_t = min(swap_t, T_obs)

                for obj_idx in range(num_objects):
                    one_hot_init = np.zeros(Fd)
                    one_hot_init[obj_idx] = 1.0
                    object_features_obs[:swap_t, obj_idx, :] = one_hot_init

                for obj_idx in range(num_objects):
                    one_hot_swapped = np.zeros(Fd)
                    one_hot_swapped[identity_labels[obj_idx]] = 1.0
                    object_features_obs[swap_t:, obj_idx, :] = one_hot_swapped

                for obj_idx in range(num_objects):
                    one_hot_swapped = np.zeros(Fd)
                    one_hot_swapped[identity_labels[obj_idx]] = 1.0
                    object_features_fut[:, obj_idx, :] = one_hot_swapped
            else:
                for obj_idx in range(num_objects):
                    one_hot = np.zeros(Fd)
                    one_hot[obj_idx] = 1.0
                    object_features_obs[:, obj_idx, :] = one_hot
                    object_features_fut[:, obj_idx, :] = one_hot
        else:
            base_features = rng.randn(num_objects, Fd)
            base_features = base_features / (np.linalg.norm(base_features, axis=1, keepdims=True) + 1e-8)

            if swap_happened:
                swap_t = occlusion_time_range[1] if occlusion_time_range else t_obs - 1
                swap_t = min(swap_t, T_obs)

                for obj_idx in range(num_objects):
                    object_features_obs[:swap_t, obj_idx, :] = base_features[obj_idx]

                for obj_idx in range(num_objects):
                    object_features_obs[swap_t:, obj_idx, :] = base_features[identity_labels[obj_idx]]

                for obj_idx in range(num_objects):
                    object_features_fut[:, obj_idx, :] = base_features[identity_labels[obj_idx]]
            else:
                for obj_idx in range(num_objects):
                    object_features_obs[:, obj_idx, :] = base_features[obj_idx]
                    object_features_fut[:, obj_idx, :] = base_features[obj_idx]
    else:
        object_features_obs = None
        object_features_fut = None

    if randomize_object_order:
        perm = rng.permutation(num_objects)
        perm_inverse = np.argsort(perm)

        observed["positions"] = observed["positions"][:, perm, :]
        observed["velocities"] = observed["velocities"][:, perm, :]
        future["positions"] = future["positions"][:, perm, :]
        future["velocities"] = future["velocities"][:, perm, :]
        identity_labels = perm_inverse[identity_labels[perm]]

        if object_features_obs is not None:
            object_features_obs = object_features_obs[:, perm, :]
            object_features_fut = object_features_fut[:, perm, :]

    return {
        "observed_positions": observed["positions"],
        "observed_velocities": observed["velocities"],
        "future_positions": future["positions"],
        "future_velocities": future["velocities"],
        "identity_labels": identity_labels,
        "object_features_obs": object_features_obs,
        "object_features_fut": object_features_fut,
        "is_swap": swap_happened,
        "has_occlusion": has_occlusion,
        "has_crossing": has_crossing,
        "force_type": force_type,
        "field_strength": field_strength,
        "damping": damping,
        "noise_std": noise_std,
    }


def _stack_episodes(episodes, feature_mode):
    data = {
        "observed_positions": np.stack([e["observed_positions"] for e in episodes]),
        "observed_velocities": np.stack([e["observed_velocities"] for e in episodes]),
        "future_positions": np.stack([e["future_positions"] for e in episodes]),
        "future_velocities": np.stack([e["future_velocities"] for e in episodes]),
        "identity_labels": np.stack([e["identity_labels"] for e in episodes]),
        "is_swap": np.array([e["is_swap"] for e in episodes]),
        "has_occlusion": np.array([e["has_occlusion"] for e in episodes]),
        "has_crossing": np.array([e["has_crossing"] for e in episodes]),
    }
    if feature_mode == "feature_bearing":
        data["object_features_obs"] = np.stack([e["object_features_obs"] for e in episodes])
        data["object_features_fut"] = np.stack([e["object_features_fut"] for e in episodes])
    return data


def generate_v3_dataset(
    n_train=1000,
    n_test=200,
    t_obs=10,
    t_pred=20,
    feature_mode="feature_bearing",
    feature_dim=2,
    force_train_type="attractor",
    force_test_type="vortex",
    force_train_params=None,
    force_test_params=None,
    randomize_object_order=True,
    disjoint_init_split=True,
    allow_occlusion=True,
    allow_crossing=True,
    swap_probability=0.5,
    seed=0,
    arena_size=64.0,
    num_objects=2,
    field_strength=0.5,
    damping=0.95,
    noise_std=0.1,
    softening=1.0,
    boundary_mode="bounce",
):
    rng = np.random.RandomState(seed)

    if force_train_params is None:
        force_train_params = {}
    if force_test_params is None:
        force_test_params = {}

    train_ft = force_train_params.get("field_strength", field_strength)
    train_damp = force_train_params.get("damping", damping)
    train_noise = force_train_params.get("noise_std", noise_std)

    test_ft = force_test_params.get("field_strength", field_strength)
    test_damp = force_test_params.get("damping", damping)
    test_noise = force_test_params.get("noise_std", noise_std)

    splits_config = {
        "train_id": (n_train, False, False, 0.5, force_train_type, train_ft, train_damp, train_noise, False),
        "clean_test_id": (n_test, False, False, 0.5, force_train_type, train_ft, train_damp, train_noise, False),
        "identity_test_mixed": (n_test, True, True, 0.5, force_train_type, train_ft, train_damp, train_noise, False),
        "identity_test_swap_only": (n_test, True, True, 1.0, force_train_type, train_ft, train_damp, train_noise, False),
        "identity_test_no_swap_only": (n_test, True, True, 0.0, force_train_type, train_ft, train_damp, train_noise, False),
        "crossing_occlusion_test": (n_test, True, True, 0.5, force_train_type, train_ft, train_damp, train_noise, True),
        "ood_force_test": (n_test, True, True, 0.5, force_test_type, test_ft, test_damp, test_noise, False),
        "featureless_control_test": (n_test, True, True, 0.5, force_train_type, train_ft, train_damp, train_noise, False),
    }

    all_splits = {}

    for split_name, (n_ep, id_test_flag, _, sp, ft, fs, dmp, ns, force_co) in splits_config.items():
        fm = "featureless" if split_name == "featureless_control_test" else feature_mode
        episodes = []
        for _ in range(n_ep):
            ep = _generate_single_episode(
                t_obs=t_obs, t_pred=t_pred,
                num_objects=num_objects, arena_size=arena_size,
                feature_mode=fm,
                feature_dim=feature_dim,
                randomize_object_order=randomize_object_order,
                identity_test=id_test_flag,
                swap_probability=sp,
                force_type=ft,
                field_strength=fs,
                damping=dmp,
                noise_std=ns,
                softening=softening,
                boundary_mode=boundary_mode,
                allow_occlusion=allow_occlusion,
                allow_crossing=allow_crossing,
                force_crossing_occlusion=force_co,
                rng=rng,
            )
            episodes.append(ep)

        all_splits[split_name] = _stack_episodes(episodes, fm)

    if disjoint_init_split:
        train_init_x = all_splits["train_id"]["observed_positions"][:, 0, :, 0].mean(axis=1)
        median_x = np.median(train_init_x)
        for split_name in all_splits:
            if split_name == "train_id":
                continue
            split_init_x = all_splits[split_name]["observed_positions"][:, 0, :, 0].mean(axis=1)
            keep = split_init_x >= median_x
            if keep.sum() < 10:
                keep = np.ones(len(split_init_x), dtype=bool)
            for key in all_splits[split_name]:
                if isinstance(all_splits[split_name][key], np.ndarray):
                    all_splits[split_name][key] = all_splits[split_name][key][keep]

    return all_splits
