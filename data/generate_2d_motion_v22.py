"""
SVT-v2.2 Dataset Generator
- randomize_object_order=True by default
- feature_mode: "featureless" | "feature_bearing"
- disjoint_init_split support
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.motion_world import simulate_episode


def generate_episode_v22(
    t_obs=10, t_pred=20, num_objects=2, arena_size=64.0,
    feature_mode="featureless",
    randomize_object_order=True,
    identity_test=False,
    swap_probability=0.5,
    rng=None,
    max_retries=20,
    **kwargs,
):
    if rng is None:
        rng = np.random.RandomState()

    force_swap = identity_test and swap_probability >= 1.0
    force_no_swap = identity_test and swap_probability <= 0.0

    for _attempt in range(max_retries + 1):
        observed, future, metadata = simulate_episode(
            num_objects=num_objects,
            width=arena_size, height=arena_size,
            t_obs=t_obs, t_pred=t_pred,
            dt=1.0,
            velocity_range=(-2.5, 2.5),
            position_range=(5.0, arena_size - 5.0),
            object_radius=1.5,
            allow_occlusion=True,
            allow_crossing=True,
            allow_hidden_perturbation=True,
            occlusion_radius=3.0,
            hidden_perturbation_strength=0.5,
            gravity=kwargs.get("gravity", 0.3),
            friction=kwargs.get("friction", 0.02),
            acceleration_noise=kwargs.get("acceleration_noise", 0.15),
            rng=rng,
            identity_test=identity_test,
        )

        identity_labels = metadata["identity_labels"]
        swap_happened = not np.array_equal(identity_labels, np.arange(num_objects))

        if force_swap and not swap_happened:
            continue
        if force_no_swap and swap_happened:
            continue
        break

    occlusion_time_range = metadata.get("occlusion_time_range")

    if feature_mode == "feature_bearing":
        T_obs = t_obs
        T_pred = t_pred
        object_features_obs = np.zeros((T_obs, num_objects, 2))
        object_features_fut = np.zeros((T_pred, num_objects, 2))

        if swap_happened and num_objects == 2:
            swap_t = occlusion_time_range[1] if occlusion_time_range else t_obs - 1
            swap_t = min(swap_t, T_obs)

            for obj_idx in range(num_objects):
                one_hot_init = np.zeros(2)
                one_hot_init[obj_idx] = 1.0
                object_features_obs[:swap_t, obj_idx, :] = one_hot_init

            for obj_idx in range(num_objects):
                one_hot_swapped = np.zeros(2)
                one_hot_swapped[1 - obj_idx] = 1.0
                object_features_obs[swap_t:, obj_idx, :] = one_hot_swapped

            for obj_idx in range(num_objects):
                one_hot_swapped = np.zeros(2)
                one_hot_swapped[1 - obj_idx] = 1.0
                object_features_fut[:, obj_idx, :] = one_hot_swapped
        else:
            for obj_idx in range(num_objects):
                one_hot = np.zeros(2)
                one_hot[obj_idx] = 1.0
                object_features_obs[:, obj_idx, :] = one_hot
                object_features_fut[:, obj_idx, :] = one_hot
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
    }


def generate_dataset_v22(
    n_train=1000,
    n_clean_test=200,
    n_counterfactual_test=200,
    n_compositional_test=200,
    n_identity_test=200,
    n_identity_test_swap_only=200,
    n_identity_test_no_swap_only=200,
    t_obs=10,
    t_pred=20,
    num_objects=2,
    arena_size=64.0,
    feature_mode="featureless",
    randomize_object_order=True,
    independent_train_test_order=True,
    disjoint_init_split=True,
    swap_probability=0.5,
    seed=0,
    **kwargs,
):
    rng = np.random.RandomState(seed)

    splits_config = {
        "train": (n_train, False, False, 0.5),
        "clean_test": (n_clean_test, False, False, 0.5),
        "counterfactual_test": (n_counterfactual_test, True, False, 0.5),
        "compositional_test": (n_compositional_test, False, False, 0.5),
        "identity_test_mixed": (n_identity_test, False, True, 0.5),
        "identity_test_swap_only": (n_identity_test_swap_only, False, True, 1.0),
        "identity_test_no_swap_only": (n_identity_test_no_swap_only, False, True, 0.0),
    }

    all_splits = {}

    for split_name, (n_episodes, cf, id_test, sp) in splits_config.items():
        episodes = []
        for _ in range(n_episodes):
            ep = generate_episode_v22(
                t_obs=t_obs, t_pred=t_pred,
                num_objects=num_objects, arena_size=arena_size,
                feature_mode=feature_mode,
                randomize_object_order=randomize_object_order,
                identity_test=id_test,
                swap_probability=sp,
                rng=rng,
                **kwargs,
            )
            episodes.append(ep)

        split_data = {
            "observed_positions": np.stack([e["observed_positions"] for e in episodes]),
            "observed_velocities": np.stack([e["observed_velocities"] for e in episodes]),
            "future_positions": np.stack([e["future_positions"] for e in episodes]),
            "future_velocities": np.stack([e["future_velocities"] for e in episodes]),
            "identity_labels": np.stack([e["identity_labels"] for e in episodes]),
        }

        if feature_mode == "feature_bearing":
            split_data["object_features_obs"] = np.stack([e["object_features_obs"] for e in episodes])
            split_data["object_features_fut"] = np.stack([e["object_features_fut"] for e in episodes])

        all_splits[split_name] = split_data

    if disjoint_init_split:
        train_init_x = all_splits["train"]["observed_positions"][:, 0, :, 0].mean(axis=1)
        median_x = np.median(train_init_x)
        for split_name in all_splits:
            if split_name == "train":
                continue
            split_init_x = all_splits[split_name]["observed_positions"][:, 0, :, 0].mean(axis=1)
            keep = split_init_x >= median_x
            if keep.sum() < 10:
                keep = np.ones(len(split_init_x), dtype=bool)
            for key in all_splits[split_name]:
                if isinstance(all_splits[split_name][key], np.ndarray):
                    all_splits[split_name][key] = all_splits[split_name][key][keep]

    return all_splits


def save_dataset_v22(all_splits, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for split_name, data in all_splits.items():
        np.savez_compressed(
            os.path.join(save_dir, f"{split_name}.npz"),
            **data,
        )
    print(f"Saved dataset to {save_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-mode", type=str, default="featureless")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    suffix = "featureless" if args.feature_mode == "featureless" else "feature_bearing"
    save_dir = f"data_v22_{suffix}"

    dataset = generate_dataset_v22(
        feature_mode=args.feature_mode,
        randomize_object_order=True,
        disjoint_init_split=True,
        seed=args.seed,
    )
    save_dataset_v22(dataset, save_dir)
