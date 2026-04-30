import sys
import os
import yaml
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.motion_world import simulate_episode


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_split(
    cfg: dict,
    split_name: str,
    n_samples: int,
    rng: np.random.RandomState,
    counterfactual: bool = False,
    identity_test: bool = False,
):
    world_cfg = cfg["world"]
    data_cfg = cfg["data"]

    episodes = []
    for _ in tqdm(range(n_samples), desc=f"Generating {split_name}"):
        cf_delta = None
        if counterfactual:
            N = world_cfg["num_objects"]
            cf_delta = rng.randn(N, 2) * 1.5

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
            toroidal=world_cfg.get("toroidal", False),
            rng=rng,
            counterfactual=counterfactual,
            cf_velocity_delta=cf_delta,
            identity_test=identity_test,
        )

        episode = {
            "observed_positions": observed["positions"],
            "observed_velocities": observed["velocities"],
            "observed_occlusion": observed["occlusion_mask"],
            "future_positions": future["positions"],
            "future_velocities": future["velocities"],
            "future_occlusion": future["occlusion_mask"],
            "identity_labels": metadata["identity_labels"],
            "crossing_flags": metadata["crossing_flags"],
            "hidden_perturbation_applied": metadata["hidden_perturbation_applied"],
            "occlusion_center": metadata["occlusion_center"],
            "occlusion_time_range": metadata["occlusion_time_range"],
            "counterfactual": metadata["counterfactual"],
            "identity_test": metadata["identity_test"],
            "split": split_name,
        }
        episodes.append(episode)

    return episodes


def generate_dataset(cfg: dict):
    seed = cfg.get("seed", 42)
    rng = np.random.RandomState(seed)
    data_cfg = cfg["data"]
    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        data_cfg["save_dir"],
    )
    os.makedirs(save_dir, exist_ok=True)

    splits = {
        "train": (data_cfg["n_train"], False, False),
        "clean_test": (data_cfg["n_clean_test"], False, False),
        "counterfactual_test": (data_cfg["n_counterfactual_test"], True, False),
        "compositional_test": (data_cfg["n_compositional_test"], False, False),
        "identity_test": (data_cfg["n_identity_test"], False, True),
    }

    all_episodes = {}
    for split_name, (n_samples, cf, id_test) in splits.items():
        episodes = generate_split(cfg, split_name, n_samples, rng, cf, id_test)
        all_episodes[split_name] = episodes

        observed_pos = np.stack([e["observed_positions"] for e in episodes])
        observed_vel = np.stack([e["observed_velocities"] for e in episodes])
        future_pos = np.stack([e["future_positions"] for e in episodes])
        future_vel = np.stack([e["future_velocities"] for e in episodes])
        identity = np.stack([e["identity_labels"] for e in episodes])

        save_path = os.path.join(save_dir, f"{split_name}.npz")
        np.savez_compressed(
            save_path,
            observed_positions=observed_pos,
            observed_velocities=observed_vel,
            future_positions=future_pos,
            future_velocities=future_vel,
            identity_labels=identity,
        )
        print(f"Saved {split_name}: {observed_pos.shape[0]} episodes -> {save_path}")

    return all_episodes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_dataset(cfg)
