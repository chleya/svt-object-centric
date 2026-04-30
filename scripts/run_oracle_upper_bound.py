import sys
import os
import yaml
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.physics_oracle import predict_future_oracle, predict_identity_oracle
from metrics.prediction_metrics import compute_prediction_metrics
from metrics.identity_metrics import compute_identity_accuracy


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split(data_dir, split_name):
    path = os.path.join(data_dir, f"{split_name}.npz")
    data = np.load(path)
    return {
        "observed_positions": data["observed_positions"],
        "observed_velocities": data["observed_velocities"],
        "future_positions": data["future_positions"],
        "future_velocities": data["future_velocities"],
        "identity_labels": data["identity_labels"],
    }


def run_oracle(cfg):
    data_dir = cfg["data"]["save_dir"]
    world_cfg = cfg["world"]

    splits = ["clean_test", "counterfactual_test", "compositional_test", "identity_test"]
    all_results = {}

    for split_name in splits:
        print(f"\nEvaluating Oracle on {split_name}...")
        data = load_split(data_dir, split_name)

        n_samples = data["observed_positions"].shape[0]
        pred_positions = np.zeros_like(data["future_positions"])

        for i in range(n_samples):
            pred_pos, _ = predict_future_oracle(
                data["observed_positions"][i],
                data["observed_velocities"][i],
                t_pred=world_cfg["t_pred"],
                dt=world_cfg["dt"],
                width=world_cfg["width"],
                height=world_cfg["height"],
                object_radius=world_cfg["object_radius"],
                gravity=world_cfg.get("gravity", 0.0),
                friction=world_cfg.get("friction", 0.0),
                acceleration_noise=world_cfg.get("acceleration_noise", 0.0),
                toroidal=world_cfg.get("toroidal", False),
            )
            pred_positions[i] = pred_pos

        pred_metrics = compute_prediction_metrics(pred_positions, data["future_positions"])
        oracle_ids = predict_identity_oracle(
            data["observed_positions"], data["observed_velocities"],
            true_identity_labels=data["identity_labels"]
        )
        id_acc = compute_identity_accuracy(oracle_ids, data["identity_labels"])

        result = {
            "mse": pred_metrics["mse"],
            "mean_predictor_mse": pred_metrics["mean_predictor_mse"],
            "skill_score": pred_metrics["skill_score"],
            "normalized_mse": pred_metrics["normalized_mse"],
            "identity_accuracy": id_acc,
        }
        all_results[split_name] = result
        print(f"  MSE: {result['mse']:.6f}, Skill: {result['skill_score']:.4f}, Identity: {id_acc:.4f}")

    return all_results


def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "oracle_upper_bound.json"), "w") as f:
        json.dump(results, f, indent=2)

    import csv
    with open(os.path.join(output_dir, "oracle_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "mse", "mean_predictor_mse", "skill_score", "identity_accuracy"])
        for split, res in results.items():
            writer.writerow([split, res["mse"], res["mean_predictor_mse"], res["skill_score"], res["identity_accuracy"]])

    print(f"\nOracle results saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke.yaml")
    parser.add_argument("--output", type=str, default="results/oracle")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = run_oracle(cfg)
    save_results(results, args.output)
