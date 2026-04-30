"""
SVT-v3 OOD Metrics
"""

import numpy as np
from typing import Dict


def compute_ood_skill(
    id_predictions: np.ndarray,
    id_targets: np.ndarray,
    ood_predictions: np.ndarray,
    ood_targets: np.ndarray,
) -> Dict[str, float]:
    id_mse = float(np.mean((id_predictions - id_targets) ** 2))
    ood_mse = float(np.mean((ood_predictions - ood_targets) ** 2))

    id_mean_mse = float(np.mean((id_targets - id_targets.mean(axis=(0, 1), keepdims=True)) ** 2))
    ood_mean_mse = float(np.mean((ood_targets - ood_targets.mean(axis=(0, 1), keepdims=True)) ** 2))

    id_skill = 1.0 - id_mse / id_mean_mse if id_mean_mse > 1e-10 else 0.0
    ood_skill = 1.0 - ood_mse / ood_mean_mse if ood_mean_mse > 1e-10 else 0.0

    skill_drop = id_skill - ood_skill

    return {
        "id_mse": id_mse,
        "ood_mse": ood_mse,
        "id_skill": id_skill,
        "ood_skill": ood_skill,
        "ood_skill_drop": skill_drop,
    }


def compute_crossing_occlusion_skill(
    all_predictions: np.ndarray,
    all_targets: np.ndarray,
    has_crossing: np.ndarray,
    has_occlusion: np.ndarray,
) -> Dict[str, float]:
    B = all_predictions.shape[0]
    all_mse = np.mean((all_predictions - all_targets) ** 2, axis=(1, 2, 3))

    crossing_mask = has_crossing.astype(bool)
    occlusion_mask = has_occlusion.astype(bool)
    both_mask = crossing_mask & occlusion_mask
    neither_mask = ~crossing_mask & ~occlusion_mask

    results = {
        "overall_mse": float(all_mse.mean()),
        "overall_skill": None,
    }

    mean_pred_mse = float(np.mean((all_targets - all_targets.mean(axis=(0, 1), keepdims=True)) ** 2))
    if mean_pred_mse > 1e-10:
        results["overall_skill"] = float(1.0 - all_mse.mean() / mean_pred_mse)

    if crossing_mask.sum() > 0:
        results["crossing_mse"] = float(all_mse[crossing_mask].mean())
    else:
        results["crossing_mse"] = float("nan")

    if occlusion_mask.sum() > 0:
        results["occlusion_mse"] = float(all_mse[occlusion_mask].mean())
    else:
        results["occlusion_mse"] = float("nan")

    if both_mask.sum() > 0:
        results["crossing_and_occlusion_mse"] = float(all_mse[both_mask].mean())
    else:
        results["crossing_and_occlusion_mse"] = float("nan")

    if neither_mask.sum() > 0:
        results["neither_mse"] = float(all_mse[neither_mask].mean())
    else:
        results["neither_mse"] = float("nan")

    return results
