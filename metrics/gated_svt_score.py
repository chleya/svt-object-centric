import numpy as np
from typing import Dict


def compute_gated_svt_score(
    clean_skill: float,
    cf_skill: float,
    comp_skill: float,
    identity_accuracy: float,
    clean_skill_threshold: float = 0.5,
) -> Dict[str, float]:
    if clean_skill < clean_skill_threshold:
        gated_score = 0.0
    else:
        cf_component = max(0.0, cf_skill)
        comp_component = max(0.0, comp_skill)
        id_component = max(0.0, identity_accuracy)
        gated_score = float(clean_skill * cf_component * comp_component * id_component)
        gated_score = max(0.0, gated_score)

    return {
        "clean_skill": clean_skill,
        "cf_skill": cf_skill,
        "comp_skill": comp_skill,
        "identity_accuracy": identity_accuracy,
        "gated_svt_score": gated_score,
        "gate_passed": clean_skill >= clean_skill_threshold,
    }


def compute_old_smss(
    clean_mse: float,
    cf_mse: float,
    comp_mse: float,
    identity_accuracy: float,
) -> float:
    if clean_mse < 1e-10:
        return 0.0

    cf_ratio = cf_mse / clean_mse
    comp_ratio = comp_mse / clean_mse

    if cf_ratio > 1.5 or comp_ratio > 1.5:
        return 0.0

    smss = identity_accuracy * (1.0 - abs(cf_ratio - 1.0)) * (1.0 - abs(comp_ratio - 1.0))
    return max(0.0, min(1.0, smss))
