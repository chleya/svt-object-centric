import sys
import os
import yaml
import numpy as np
import json
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.knn_retriever import KNN_REGISTRY
from metrics.prediction_metrics import compute_prediction_metrics
from metrics.identity_metrics import compute_identity_accuracy, velocity_continuity_identity
from metrics.gated_svt_score import compute_gated_svt_score, compute_old_smss


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


def run_knn_attack(cfg):
    data_dir = cfg["data"]["save_dir"]
    knn_cfg = cfg["knn"]
    metrics_cfg = cfg["metrics"]

    train_data = load_split(data_dir, "train")
    clean_data = load_split(data_dir, "clean_test")
    cf_data = load_split(data_dir, "counterfactual_test")
    comp_data = load_split(data_dir, "compositional_test")
    identity_data = load_split(data_dir, "identity_test")

    max_train = knn_cfg.get("max_train", len(train_data["observed_positions"]))
    train_obs = train_data["observed_positions"][:max_train]
    train_future = train_data["future_positions"][:max_train]
    train_identity = train_data["identity_labels"][:max_train]

    results = []

    for model_name, ModelClass in KNN_REGISTRY.items():
        for k in knn_cfg["k_values"]:
            print(f"\nRunning {model_name} (k={k})...")
            model = ModelClass(k=k, weighting=knn_cfg["weighting"])
            model.fit(train_obs, train_future, train_identity)

            splits_results = {}
            for split_name, split_data in [
                ("clean", clean_data),
                ("counterfactual", cf_data),
                ("compositional", comp_data),
                ("identity", identity_data),
            ]:
                pred_future = model.predict_future(split_data["observed_positions"])
                pred_metrics = compute_prediction_metrics(
                    pred_future, split_data["future_positions"]
                )

                pred_identity = model.predict_identity(
                    split_data["observed_positions"],
                    test_future=split_data["future_positions"]
                )
                id_acc = compute_identity_accuracy(
                    pred_identity, split_data["identity_labels"]
                )

                splits_results[split_name] = {
                    "mse": pred_metrics["mse"],
                    "mean_predictor_mse": pred_metrics["mean_predictor_mse"],
                    "skill_score": pred_metrics["skill_score"],
                    "normalized_mse": pred_metrics["normalized_mse"],
                    "identity_accuracy": id_acc,
                }

            clean = splits_results["clean"]
            cf = splits_results["counterfactual"]
            comp = splits_results["compositional"]
            identity = splits_results["identity"]

            gated = compute_gated_svt_score(
                clean["skill_score"],
                cf["skill_score"],
                comp["skill_score"],
                identity["identity_accuracy"],
                clean_skill_threshold=metrics_cfg["clean_skill_threshold"],
            )

            old_smss = compute_old_smss(
                clean["mse"], cf["mse"], comp["mse"], identity["identity_accuracy"]
            )

            result = {
                "model": model_name,
                "k": k,
                "clean_mse": clean["mse"],
                "cf_mse": cf["mse"],
                "comp_mse": comp["mse"],
                "clean_skill": clean["skill_score"],
                "cf_skill": cf["skill_score"],
                "comp_skill": comp["skill_score"],
                "identity_acc": identity["identity_accuracy"],
                "gated_svt_score": gated["gated_svt_score"],
                "gate_passed": gated["gate_passed"],
                "old_smss": old_smss,
            }
            results.append(result)
            print(f"  Clean skill: {clean['skill_score']:.3f}, Identity: {identity['identity_accuracy']:.3f}, Gated: {gated['gated_svt_score']:.3f}")

    # Velocity continuity identity baseline
    print("\nRunning VelocityContinuity identity baseline...")
    is_toroidal = cfg["world"].get("toroidal", False)
    w_width = cfg["world"]["width"]
    w_height = cfg["world"]["height"]
    for split_name, split_data in [
        ("clean", clean_data),
        ("identity", identity_data),
    ]:
        vel_ids = velocity_continuity_identity(
            split_data["observed_positions"],
            future_positions=split_data["future_positions"],
            toroidal=is_toroidal,
            width=w_width,
            height=w_height,
        )
        vel_acc = compute_identity_accuracy(vel_ids, split_data["identity_labels"])
        print(f"  VelocityContinuity on {split_name}: Identity={vel_acc:.3f}")

    return results


def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    import csv
    keys = list(results[0].keys())
    with open(os.path.join(output_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    with open(os.path.join(output_dir, "full_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke.yaml")
    parser.add_argument("--output", type=str, default="results/knn_attack")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = run_knn_attack(cfg)
    save_results(results, args.output)
