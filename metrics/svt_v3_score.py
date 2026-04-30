"""
SVT-v3 Gated Score

Primary identity metric: identity_swap_only
"""

import numpy as np
from typing import Dict


def compute_gated_svt_score_v3(
    clean_skill: float,
    cf_skill: float,
    comp_skill: float,
    identity_swap_only: float,
    identity_overall: float = None,
    clean_skill_threshold: float = 0.5,
) -> Dict[str, float]:
    if clean_skill < clean_skill_threshold:
        gated_swap = 0.0
        gated_overall = 0.0
    else:
        cf_c = max(0.0, cf_skill)
        comp_c = max(0.0, comp_skill)
        id_swap_c = max(0.0, identity_swap_only)
        gated_swap = float(clean_skill * cf_c * comp_c * id_swap_c)
        gated_swap = max(0.0, gated_swap)

        if identity_overall is not None:
            id_overall_c = max(0.0, identity_overall)
            gated_overall = float(clean_skill * cf_c * comp_c * id_overall_c)
            gated_overall = max(0.0, gated_overall)
        else:
            gated_overall = 0.0

    score_drop = gated_overall - gated_swap

    no_swap_bias_flag = False
    if identity_overall is not None and not np.isnan(identity_swap_only):
        no_swap_bias_flag = (identity_overall - identity_swap_only) > 0.1

    return {
        "clean_skill": clean_skill,
        "cf_skill": cf_skill,
        "comp_skill": comp_skill,
        "identity_swap_only": identity_swap_only,
        "identity_overall": identity_overall if identity_overall is not None else float("nan"),
        "gated_score_swap_only": gated_swap,
        "gated_score_overall_id": gated_overall,
        "score_drop": score_drop,
        "no_swap_bias_flag": no_swap_bias_flag,
    }
