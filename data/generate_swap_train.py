import sys
import os
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.motion_world import simulate_episode


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_swap_augmented_data(cfg, swap_fraction=0.3, seed=42):
    world_cfg = cfg["world"]
    data_cfg = cfg["data"]
    rng = np.random.RandomState(seed)

    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        data_cfg["save_dir"] + "_swaptrain",
    )
    os.makedirs(save_dir, exist_ok=True)

    n_train = data_cfg["n_train"]
    n_swap = int(n_train * swap_fraction)
    n_normal = n_train - n_swap

    print(f"Generating {n_normal} normal + {n_swap} swap episodes for training")

    all_obs_pos = []
    all_obs_vel = []
    all_fut_pos = []
    all_fut_vel = []
    all_identity = []

    for i in range(n_train):
        is_identity_test = (i >= n_normal)

        observed, future, metadata = simulate_episode(
            num_objects=world_cfg["num_objects"],
            width=world_cfg["width"],
            height=world_cfg["height"],
            t_obs=world_cfg["t_obs"],
            t_pred=world_cfg["t_pred"],
            dt=world_cfg["dt"],
            velocity_range=tuple(data_cfg["velocity_range"]),
            position_range=tuple(data_cfg["position_range"]),
            object_radius=world_cfg["object_radius"],
            allow_occlusion=world_cfg["allow_occlusion"],
            allow_crossing=world_cfg["allow_crossing"],
            allow_hidden_perturbation=world_cfg["allow_hidden_perturbation"],
            occlusion_radius=world_cfg["occlusion_radius"],
            hidden_perturbation_strength=world_cfg["hidden_perturbation_strength"],
            gravity=world_cfg.get("gravity", 0.0),
            friction=world_cfg.get("friction", 0.0),
            acceleration_noise=world_cfg.get("acceleration_noise", 0.0),
            rng=rng,
            identity_test=is_identity_test,
        )

        all_obs_pos.append(observed["positions"])
        all_obs_vel.append(observed["velocities"])
        all_fut_pos.append(future["positions"])
        all_fut_vel.append(future["velocities"])
        all_identity.append(metadata["identity_labels"])

    obs_pos = np.stack(all_obs_pos)
    obs_vel = np.stack(all_obs_vel)
    fut_pos = np.stack(all_fut_pos)
    fut_vel = np.stack(all_fut_vel)
    identity = np.stack(all_identity)

    swap_count = np.sum(identity[:, 0] == 1)
    print(f"Training set: {len(identity)} episodes, {swap_count} with swaps ({swap_count/len(identity)*100:.1f}%)")

    save_path = os.path.join(save_dir, "train.npz")
    np.savez_compressed(
        save_path,
        observed_positions=obs_pos,
        observed_velocities=obs_vel,
        future_positions=fut_pos,
        future_velocities=fut_vel,
        identity_labels=identity,
    )
    print(f"Saved train -> {save_path}")

    for split_name in ["clean_test", "counterfactual_test", "compositional_test", "identity_test"]:
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            data_cfg["save_dir"],
            f"{split_name}.npz",
        )
        if os.path.exists(src_path):
            import shutil
            dst_path = os.path.join(save_dir, f"{split_name}.npz")
            shutil.copy2(src_path, dst_path)
            print(f"Copied {split_name} -> {dst_path}")

    return save_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke_hard.yaml")
    parser.add_argument("--swap-fraction", type=float, default=0.3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    save_dir = generate_swap_augmented_data(cfg, swap_fraction=args.swap_fraction)
    print(f"\nDone! Swap-augmented data saved to: {save_dir}")
