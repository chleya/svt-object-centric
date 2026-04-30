"""
SVT-v2.3 Identity Breakdown Metrics

Breaks down identity accuracy into swap-only, no-swap, and balanced components.
"""

import numpy as np
from typing import Dict


def compute_identity_breakdown(
    pred_identity: np.ndarray,
    true_identity: np.ndarray,
) -> Dict[str, float]:
    if pred_identity.shape != true_identity.shape:
        if pred_identity.ndim == 1 and true_identity.ndim == 2:
            pred_identity = np.tile(pred_identity, (true_identity.shape[0], 1))

    B = true_identity.shape[0]
    N = true_identity.shape[1]

    is_swap = true_identity[:, 0] != 0
    is_no_swap = ~is_swap

    n_swap = int(is_swap.sum())
    n_no_swap = int(is_no_swap.sum())

    correct = pred_identity == true_identity
    per_episode_correct = correct.all(axis=1).astype(float)

    identity_overall = float(per_episode_correct.mean())

    if n_no_swap > 0:
        identity_no_swap = float(per_episode_correct[is_no_swap].mean())
    else:
        identity_no_swap = float("nan")

    if n_swap > 0:
        identity_swap_only = float(per_episode_correct[is_swap].mean())
    else:
        identity_swap_only = float("nan")

    pred_is_swap = pred_identity[:, 0] != 0

    if n_swap > 0:
        swap_detect_recall = float(pred_is_swap[is_swap].sum() / n_swap)
    else:
        swap_detect_recall = float("nan")

    if n_no_swap > 0:
        swap_false_positive_rate = float(pred_is_swap[is_no_swap].sum() / n_no_swap)
    else:
        swap_false_positive_rate = float("nan")

    if not np.isnan(identity_no_swap) and not np.isnan(identity_swap_only):
        balanced_identity = (identity_no_swap + identity_swap_only) / 2.0
    else:
        balanced_identity = float("nan")

    return {
        "identity_overall": identity_overall,
        "identity_no_swap": identity_no_swap,
        "identity_swap_only": identity_swap_only,
        "swap_detect_recall": swap_detect_recall,
        "swap_false_positive_rate": swap_false_positive_rate,
        "balanced_identity": balanced_identity,
    }
