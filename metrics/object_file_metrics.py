"""
SVT-v4.1 Object-File Metrics

Confidence calibration and conflict gate evaluation metrics.
"""

import numpy as np


def compute_confidence_calibration(pred_identity, true_identity, confidences):
    if isinstance(confidences, list):
        confidences = np.array(confidences)

    correct = (pred_identity == true_identity).all(axis=1)

    if len(confidences.shape) > 1:
        per_episode_conf = confidences.max(axis=1)
    else:
        per_episode_conf = confidences

    n_correct = int(correct.sum())
    n_incorrect = int((~correct).sum())

    avg_conf_correct = float(per_episode_conf[correct].mean()) if n_correct > 0 else float('nan')
    avg_conf_incorrect = float(per_episode_conf[~correct].mean()) if n_incorrect > 0 else float('nan')

    if n_correct > 0 and n_incorrect > 0:
        calibration_error = abs(avg_conf_correct - avg_conf_incorrect)
    else:
        calibration_error = float('nan')

    return {
        'avg_confidence_correct': avg_conf_correct,
        'avg_confidence_incorrect': avg_conf_incorrect,
        'confidence_calibration_error': calibration_error,
        'n_correct': n_correct,
        'n_incorrect': n_incorrect,
    }


def compute_conflict_gate_stats(chosen_sources, pred_identity, true_identity):
    if not chosen_sources:
        return {
            'chosen_source_feature_rate': float('nan'),
            'chosen_source_trajectory_rate': float('nan'),
            'chosen_source_uncertain_rate': float('nan'),
        }

    flat_sources = []
    for ep_sources in chosen_sources:
        if isinstance(ep_sources, list):
            flat_sources.extend(ep_sources)
        else:
            flat_sources.append(ep_sources)

    n_total = len(flat_sources)
    n_feature = sum(1 for s in flat_sources if s == "feature")
    n_trajectory = sum(1 for s in flat_sources if s in ("trajectory", "trajectory_fallback", "trajectory_occlusion"))
    n_uncertain = sum(1 for s in flat_sources if s == "uncertain")
    n_agreement = sum(1 for s in flat_sources if s == "agreement")

    return {
        'chosen_source_feature_rate': n_feature / n_total if n_total > 0 else float('nan'),
        'chosen_source_trajectory_rate': n_trajectory / n_total if n_total > 0 else float('nan'),
        'chosen_source_uncertain_rate': n_uncertain / n_total if n_total > 0 else float('nan'),
        'chosen_source_agreement_rate': n_agreement / n_total if n_total > 0 else float('nan'),
    }


def compute_abstention_metrics(pred_identity, true_identity, abstain_flags):
    if isinstance(abstain_flags, list):
        abstain_flags = np.array(abstain_flags)

    correct = (pred_identity == true_identity).all(axis=1)

    n_abstain = int(abstain_flags.sum())
    n_total = len(abstain_flags)
    abstention_rate = n_abstain / n_total if n_total > 0 else 0.0

    not_abstain = ~abstain_flags
    n_not_abstain = int(not_abstain.sum())

    if n_not_abstain > 0:
        accuracy_when_not_abstaining = float(correct[not_abstain].mean())
    else:
        accuracy_when_not_abstaining = float('nan')

    if n_abstain > 0:
        accuracy_when_abstaining = float(correct[abstain_flags].mean())
    else:
        accuracy_when_abstaining = float('nan')

    return {
        'abstention_rate': abstention_rate,
        'accuracy_when_not_abstaining': accuracy_when_not_abstaining,
        'accuracy_when_abstaining': accuracy_when_abstaining,
        'n_abstain': n_abstain,
        'n_total': n_total,
    }


def compute_conflict_detection_accuracy(sources, true_is_conflict):
    if not sources or len(true_is_conflict) == 0:
        return {
            'conflict_detection_accuracy': float('nan'),
            'conflict_precision': float('nan'),
            'conflict_recall': float('nan'),
        }

    if isinstance(sources, list):
        sources = np.array(sources)
    if isinstance(true_is_conflict, list):
        true_is_conflict = np.array(true_is_conflict)

    detected = np.array([s != "agreement" for s in sources])

    n = len(detected)
    if n != len(true_is_conflict):
        n = min(n, len(true_is_conflict))
        detected = detected[:n]
        true_is_conflict = true_is_conflict[:n]

    tp = int((detected & true_is_conflict).sum())
    fp = int((detected & ~true_is_conflict).sum())
    fn = int((~detected & true_is_conflict).sum())
    tn = int((~detected & ~true_is_conflict).sum())

    accuracy = (tp + tn) / n if n > 0 else float('nan')
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')

    return {
        'conflict_detection_accuracy': accuracy,
        'conflict_precision': precision,
        'conflict_recall': recall,
    }
