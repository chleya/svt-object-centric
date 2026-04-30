import sys
import os
import yaml
import numpy as np
import json
import csv
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.knn_retriever_v2 import KNN_V2_REGISTRY, KNNLastVelocityBlend, TranslationNormalizedDeltaKNN
from metrics.prediction_metrics import compute_prediction_metrics
from metrics.identity_metrics import compute_identity_accuracy
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


def evaluate_model(model, split_data):
    pred_future = model.predict_future(split_data["observed_positions"])
    pred_metrics = compute_prediction_metrics(pred_future, split_data["future_positions"])

    pred_identity = model.predict_identity(
        split_data["observed_positions"],
        test_future=split_data["future_positions"]
    )
    id_acc = compute_identity_accuracy(pred_identity, split_data["identity_labels"])

    return {
        "mse": pred_metrics["mse"],
        "mean_predictor_mse": pred_metrics["mean_predictor_mse"],
        "skill_score": pred_metrics["skill_score"],
        "normalized_mse": pred_metrics["normalized_mse"],
        "identity_accuracy": id_acc,
    }


def run_knn_v2_attack(cfg):
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

    for model_name, ModelClass in KNN_V2_REGISTRY.items():
        for k in knn_cfg["k_values"]:
            print(f"\nRunning {model_name} (k={k})...")

            if model_name == "LastVelocityBaseline":
                model = ModelClass()
            else:
                model = ModelClass(k=k, weighting=knn_cfg["weighting"])

            model.fit(train_obs, train_future, train_identity)

            splits_results = {}
            for split_name, split_data in [
                ("clean", clean_data),
                ("counterfactual", cf_data),
                ("compositional", comp_data),
                ("identity", identity_data),
            ]:
                splits_results[split_name] = evaluate_model(model, split_data)

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
                "version": "v2_delta",
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

    # Blend models: TranslationNormalizedDeltaKNN + LastVelocity
    print("\nRunning Blend models...")
    for k in knn_cfg["k_values"]:
        for alpha in [0.3, 0.5, 0.7]:
            knn_core = TranslationNormalizedDeltaKNN(k=k, weighting=knn_cfg["weighting"])
            knn_core.fit(train_obs, train_future, train_identity)
            blend = KNNLastVelocityBlend(knn_core, alpha=alpha)
            blend.fit(train_obs, train_future, train_identity)

            splits_results = {}
            for split_name, split_data in [
                ("clean", clean_data),
                ("counterfactual", cf_data),
                ("compositional", comp_data),
                ("identity", identity_data),
            ]:
                splits_results[split_name] = evaluate_model(blend, split_data)

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
                "model": f"Blend_TransNorm_LastVel_a{alpha}",
                "k": k,
                "version": "v2_blend",
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
            print(f"  Blend(a={alpha},k={k}) Clean skill: {clean['skill_score']:.3f}, Identity: {identity['identity_accuracy']:.3f}, Gated: {gated['gated_svt_score']:.3f}")

    return results


def run_scale_sweep(cfg):
    data_dir = cfg["data"]["save_dir"]
    knn_cfg = cfg["knn"]
    metrics_cfg = cfg["metrics"]

    train_data = load_split(data_dir, "train")
    clean_data = load_split(data_dir, "clean_test")
    identity_data = load_split(data_dir, "identity_test")

    scale_sizes = [50, 100, 250, 500, 1000, min(5000, len(train_data["observed_positions"]))]
    scale_sizes = sorted(set(s for s in scale_sizes if s <= len(train_data["observed_positions"])))

    results = []

    for n_train in scale_sizes:
        print(f"\nScale sweep: n_train={n_train}")
        train_obs = train_data["observed_positions"][:n_train]
        train_future = train_data["future_positions"][:n_train]
        train_identity = train_data["identity_labels"][:n_train]

        for model_name in ["TranslationNormalizedDeltaKNN", "RawDeltaKNN", "LastVelocityBaseline"]:
            ModelClass = KNN_V2_REGISTRY[model_name]

            if model_name == "LastVelocityBaseline":
                model = ModelClass()
                k_val = 0
            else:
                k_val = min(5, n_train)
                model = ModelClass(k=k_val, weighting=knn_cfg["weighting"])

            model.fit(train_obs, train_future, train_identity)

            clean_res = evaluate_model(model, clean_data)
            id_res = evaluate_model(model, identity_data)

            gated = compute_gated_svt_score(
                clean_res["skill_score"], 0.0, 0.0,
                id_res["identity_accuracy"],
                clean_skill_threshold=metrics_cfg["clean_skill_threshold"],
            )

            result = {
                "model": model_name,
                "n_train": n_train,
                "k": k_val,
                "clean_skill": clean_res["skill_score"],
                "clean_mse": clean_res["mse"],
                "identity_acc": id_res["identity_accuracy"],
                "gated_svt_score": gated["gated_svt_score"],
            }
            results.append(result)
            print(f"  {model_name} (n={n_train}): Clean skill={clean_res['skill_score']:.3f}, Identity={id_res['identity_accuracy']:.3f}")

    return results


def save_results(results, output_dir, filename="summary"):
    os.makedirs(output_dir, exist_ok=True)

    keys = list(results[0].keys())
    with open(os.path.join(output_dir, f"{filename}.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    with open(os.path.join(output_dir, f"{filename}_full.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/{filename}.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke.yaml")
    parser.add_argument("--output", type=str, default="results/knn_attack_v2")
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("=" * 60)
    print("k-NN v2 Attack (Delta-output)")
    print("=" * 60)
    results = run_knn_v2_attack(cfg)
    save_results(results, args.output, "v2_summary")

    if not args.skip_sweep:
        print("\n" + "=" * 60)
        print("Scale Sweep")
        print("=" * 60)
        sweep_results = run_scale_sweep(cfg)
        save_results(sweep_results, args.output, "scale_sweep")
