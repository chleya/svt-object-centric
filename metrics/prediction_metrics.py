import numpy as np
from typing import Dict, Optional


def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def compute_mean_predictor_mse(target: np.ndarray) -> float:
    mean_future = np.mean(target, axis=(0, 1), keepdims=True)
    return compute_mse(np.broadcast_to(mean_future, target.shape), target)


def compute_skill_score(model_mse: float, mean_predictor_mse: float) -> float:
    if mean_predictor_mse < 1e-10:
        return 0.0
    return 1.0 - model_mse / mean_predictor_mse


def compute_normalized_mse(model_mse: float, mean_predictor_mse: float) -> float:
    if mean_predictor_mse < 1e-10:
        return float("inf")
    return model_mse / mean_predictor_mse


def compute_identity_accuracy(pred_ids: np.ndarray, true_ids: np.ndarray) -> float:
    return float(np.mean(pred_ids == true_ids))


def compute_prediction_metrics(
    pred_positions: np.ndarray,
    true_positions: np.ndarray,
) -> Dict[str, float]:
    mse = compute_mse(pred_positions, true_positions)
    mean_pred_mse = compute_mean_predictor_mse(true_positions)
    skill = compute_skill_score(mse, mean_pred_mse)
    norm_mse = compute_normalized_mse(mse, mean_pred_mse)

    per_object_mse = []
    for i in range(pred_positions.shape[1]):
        per_object_mse.append(compute_mse(pred_positions[:, i], true_positions[:, i]))

    return {
        "mse": mse,
        "mean_predictor_mse": mean_pred_mse,
        "skill_score": skill,
        "normalized_mse": norm_mse,
        "per_object_mse": per_object_mse,
    }
