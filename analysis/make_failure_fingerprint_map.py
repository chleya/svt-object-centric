import sys
import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(csv_path):
    results = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                try:
                    row[key] = float(row[key])
                except:
                    pass
            results.append(row)
    return results


def make_fingerprint_map(results, output_path):
    models = sorted(set(r["model"] for r in results))
    gates = ["clean_skill", "cf_skill", "comp_skill", "identity_acc"]
    gate_labels = ["Clean\nPrediction", "Counterfactual", "Compositional", "Identity"]

    data = np.zeros((len(models), len(gates)))
    passes = np.zeros((len(models), len(gates)), dtype=bool)

    for i, model in enumerate(models):
        model_results = [r for r in results if r["model"] == model]
        if not model_results:
            continue
        best = max(model_results, key=lambda r: r.get("gated_svt_score", 0))
        for j, gate in enumerate(gates):
            val = best.get(gate, 0)
            data[i, j] = val
            passes[i, j] = val >= 0.5

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(gates)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(gate_labels)
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(gates)):
            val = data[i, j]
            symbol = "✓" if passes[i, j] else "✗"
            text_color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{symbol}\n{val:.2f}", ha="center", va="center",
                    color=text_color, fontsize=9, fontweight="bold")

    ax.set_title("SVT-v2 Failure Fingerprint Map\n(Model × Gate Performance)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Failure Fingerprint Map saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/knn_attack/summary.csv")
    parser.add_argument("--output", type=str, default="results/knn_attack/failure_fingerprint.png")
    args = parser.parse_args()

    results = load_results(args.input)
    make_fingerprint_map(results, args.output)
