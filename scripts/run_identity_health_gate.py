"""
SVT-v2.4 Identity Health Gate

Consolidates v2.1/v2.2/v2.3 checks into a single automated gate.
All future SVT-v3 experiments must pass this gate before proceeding.
"""

import sys
import os
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/svt_v2_4_identity_health_gate"

V21_AUDIT_SUMMARY = "results/svt_v2_1_audit/audit_summary.csv"
V22_DATASET_HEALTH = "results/svt_v2_2_dataset_fix/dataset_health.csv"
V22_FEATURE_BASELINE = "results/svt_v2_2_dataset_fix/feature_identity_baseline.csv"
V22_LABEL_PERMUTATION = "results/svt_v2_2_dataset_fix/label_permutation_health.csv"
V23_IDENTITY_BREAKDOWN = "results/svt_v2_3_swap_only_identity/identity_breakdown.csv"
V23_GATED_SCORE = "results/svt_v2_3_swap_only_identity/gated_score_v23.csv"


def read_csv(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_csv(results, filename, fieldnames):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, filename), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def check_label_permutation_sanity(v22_perm, v22_health):
    if v22_perm is not None:
        perm_vals = [float(r["permuted_identity_accuracy"]) for r in v22_perm]
        avg = np.mean(perm_vals)
        passed = all(r["status"] == "PASS" for r in v22_perm)
    elif v22_health is not None:
        lp_row = [r for r in v22_health if r["check_name"] == "label_permutation_sanity"]
        if lp_row:
            avg = float(lp_row[0]["value"])
            passed = lp_row[0]["status"] == "PASS"
        else:
            return "label_permutation_sanity", "MISSING", "N/A", "permuted identity around 0.5", "critical", "Run v2.2 label permutation check"
    else:
        return "label_permutation_sanity", "MISSING", "N/A", "permuted identity around 0.5", "critical", "Run v2.2 label permutation check"

    status = "PASS" if passed and abs(avg - 0.5) < 0.15 else "FAIL"
    severity = "none" if status == "PASS" else "critical"
    action = "none" if status == "PASS" else "Fix identity metric or pipeline"
    return "label_permutation_sanity", status, f"{avg:.4f}", "permuted identity around 0.5", severity, action


def check_featureless_identifiability(v22_feat, v22_health):
    if v22_feat is not None:
        fl_row = [r for r in v22_feat if r["feature_mode"] == "featureless"]
        if fl_row:
            val = float(fl_row[0]["identity_accuracy"])
            passed = fl_row[0]["status"] == "PASS"
        else:
            return "featureless_identifiability", "MISSING", "N/A", "FeatureAwareBaseline featureless around 0.5", "critical", "Run v2.2 feature identity baseline"
    elif v22_health is not None:
        fl_row = [r for r in v22_health if r["check_name"] == "featureless_identifiability"]
        if fl_row:
            val = float(fl_row[0]["value"])
            passed = fl_row[0]["status"] == "PASS"
        else:
            return "featureless_identifiability", "MISSING", "N/A", "FeatureAwareBaseline featureless around 0.5", "critical", "Run v2.2 feature identity baseline"
    else:
        return "featureless_identifiability", "MISSING", "N/A", "FeatureAwareBaseline featureless around 0.5", "critical", "Run v2.2 feature identity baseline"

    status = "PASS" if passed and 0.4 <= val <= 0.6 else "FAIL"
    severity = "none" if status == "PASS" else "serious"
    action = "none" if status == "PASS" else "Check feature pipeline for featureless mode"
    return "featureless_identifiability", status, f"{val:.4f}", "FeatureAwareBaseline featureless around 0.5", severity, action


def check_feature_bearing_identifiability(v22_feat, v22_health):
    if v22_feat is not None:
        fb_row = [r for r in v22_feat if r["feature_mode"] == "feature_bearing"]
        if fb_row:
            val = float(fb_row[0]["identity_accuracy"])
            passed = fb_row[0]["status"] == "PASS"
        else:
            return "feature_bearing_identifiability", "MISSING", "N/A", "FeatureAwareBaseline feature_bearing >= 0.95", "critical", "Run v2.2 feature identity baseline"
    elif v22_health is not None:
        fb_row = [r for r in v22_health if r["check_name"] == "feature_bearing_identifiability"]
        if fb_row:
            val = float(fb_row[0]["value"])
            passed = fb_row[0]["status"] == "PASS"
        else:
            return "feature_bearing_identifiability", "MISSING", "N/A", "FeatureAwareBaseline feature_bearing >= 0.95", "critical", "Run v2.2 feature identity baseline"
    else:
        return "feature_bearing_identifiability", "MISSING", "N/A", "FeatureAwareBaseline feature_bearing >= 0.95", "critical", "Run v2.2 feature identity baseline"

    status = "PASS" if passed and val >= 0.95 else "FAIL"
    severity = "none" if status == "PASS" else "critical"
    action = "none" if status == "PASS" else "Fix feature pipeline or identity labels"
    return "feature_bearing_identifiability", status, f"{val:.4f}", "FeatureAwareBaseline feature_bearing >= 0.95", severity, action


def check_object_order_leakage(v21_audit, v22_health):
    if v21_audit is not None:
        oo_row = [r for r in v21_audit if r["audit_name"] == "A_object_order"]
        if oo_row:
            severity_v21 = oo_row[0]["severity"]
            leakage_found = severity_v21 in ["serious", "critical"]
        else:
            leakage_found = False
    else:
        leakage_found = None

    if v22_health is not None:
        oo_row = [r for r in v22_health if r["check_name"] == "object_order_randomized_default"]
        if oo_row:
            randomized = oo_row[0]["status"] == "PASS"
        else:
            randomized = None
    else:
        randomized = None

    if leakage_found is None and randomized is None:
        return "object_order_leakage_checked", "MISSING", "N/A", "v2.1/v2.2 audit exists and order randomized", "critical", "Run v2.1 audit and v2.2 dataset fix"

    if leakage_found and randomized:
        return "object_order_leakage_checked", "PASS", "leakage_found_and_fixed", "v2.1/v2.2 audit exists and order randomized", "warning", "none (leakage was found in v2.1 but fixed in v2.2)"
    elif not leakage_found and randomized:
        return "object_order_leakage_checked", "PASS", "no_leakage_and_randomized", "v2.1/v2.2 audit exists and order randomized", "none", "none"
    elif leakage_found and not randomized:
        return "object_order_leakage_checked", "FAIL", "leakage_found_not_fixed", "v2.1/v2.2 audit exists and order randomized", "critical", "Fix object order randomization"
    else:
        return "object_order_leakage_checked", "PASS", "no_leakage", "v2.1/v2.2 audit exists and order randomized", "none", "none"


def check_swap_only_split_valid(v23_breakdown):
    if v23_breakdown is None:
        return "swap_only_split_valid", "MISSING", "N/A", "swap_only split contains 100% swap episodes", "critical", "Run v2.3 swap-only identity test"

    swap_rows = [r for r in v23_breakdown if r["split_name"] == "identity_test_swap_only"]
    if not swap_rows:
        return "swap_only_split_valid", "MISSING", "N/A", "swap_only split contains 100% swap episodes", "critical", "Run v2.3 with swap_only split"

    all_no_swap_nan = all(r.get("identity_no_swap", "nan") == "nan" for r in swap_rows)
    fab_fb = [r for r in swap_rows if r["model"] == "FeatureAwareIdentityBaseline" and r["feature_mode"] == "feature_bearing"]
    fab_fb_swap_id = float(fab_fb[0]["identity_swap_only"]) if fab_fb and fab_fb[0].get("identity_swap_only", "nan") != "nan" else None

    if all_no_swap_nan and fab_fb_swap_id is not None and fab_fb_swap_id >= 0.95:
        return "swap_only_split_valid", "PASS", "100% swap (no_swap=nan, FAB_fb_swap=1.0)", "swap_only split contains 100% swap episodes", "none", "none"
    elif all_no_swap_nan:
        return "swap_only_split_valid", "PASS", "100% swap (no_swap=nan)", "swap_only split contains 100% swap episodes", "none", "none"
    else:
        return "swap_only_split_valid", "FAIL", "contains no-swap episodes", "swap_only split contains 100% swap episodes", "critical", "Fix swap_only split generation to ensure 100% swap episodes"


def check_no_swap_only_split_valid(v23_breakdown):
    if v23_breakdown is None:
        return "no_swap_only_split_valid", "MISSING", "N/A", "no_swap_only split contains 0% swap episodes", "critical", "Run v2.3 swap-only identity test"

    noswap_rows = [r for r in v23_breakdown if r["split_name"] == "identity_test_no_swap_only"]
    if not noswap_rows:
        return "no_swap_only_split_valid", "MISSING", "N/A", "no_swap_only split contains 0% swap episodes", "critical", "Run v2.3 with no_swap_only split"

    all_swap_nan = all(r.get("identity_swap_only", "nan") == "nan" for r in noswap_rows)
    fab_fb = [r for r in noswap_rows if r["model"] == "FeatureAwareIdentityBaseline" and r["feature_mode"] == "feature_bearing"]
    fab_fb_noswap_id = float(fab_fb[0]["identity_no_swap"]) if fab_fb and fab_fb[0].get("identity_no_swap", "nan") != "nan" else None

    if all_swap_nan:
        return "no_swap_only_split_valid", "PASS", "0% swap (swap_only=nan)", "no_swap_only split contains 0% swap episodes", "none", "none"
    else:
        return "no_swap_only_split_valid", "FAIL", "contains swap episodes", "no_swap_only split contains 0% swap episodes", "critical", "Fix no_swap_only split generation to ensure 0% swap episodes"


def check_no_swap_bias(v23_breakdown):
    if v23_breakdown is None:
        return "no_swap_bias_detected", "MISSING", "N/A", "overall identity not much greater than swap_only identity", "critical", "Run v2.3 swap-only identity test"

    mixed_rows = [r for r in v23_breakdown if r["split_name"] == "identity_test_mixed"
                  and r["model"] in ["RawTrajectoryKNN", "TranslationNormalizedKNN"]]

    if not mixed_rows:
        return "no_swap_bias_detected", "MISSING", "N/A", "overall identity not much greater than swap_only identity", "warning", "No mixed split data for KNN models"

    bias_found = False
    max_gap = 0.0
    for r in mixed_rows:
        overall = float(r["identity_overall"])
        swap_only = float(r["identity_swap_only"]) if r["identity_swap_only"] != "nan" else float("nan")
        if not np.isnan(swap_only):
            gap = overall - swap_only
            max_gap = max(max_gap, gap)
            if gap > 0.1:
                bias_found = True

    if bias_found:
        return "no_swap_bias_detected", "WARNING", f"max_gap={max_gap:.3f}", "overall identity not much greater than swap_only identity", "warning", "Use identity_swap_only as primary metric; overall identity is inflated by no-swap episodes"
    else:
        return "no_swap_bias_detected", "PASS", f"max_gap={max_gap:.3f}", "overall identity not much greater than swap_only identity", "none", "none"


def check_identity_metric_policy(v23_breakdown):
    if v23_breakdown is None:
        return "identity_metric_policy", "MISSING", "N/A", "identity_swap_only present in v2.3 outputs", "critical", "Run v2.3 swap-only identity test"

    has_swap_only_col = any("identity_swap_only" in r and r["identity_swap_only"] not in ["", "nan"] for r in v23_breakdown)

    if has_swap_only_col:
        return "identity_metric_policy", "PASS", "identity_swap_only_available", "identity_swap_only present in v2.3 outputs", "none", "none"
    else:
        return "identity_metric_policy", "FAIL", "identity_swap_only_missing", "identity_swap_only present in v2.3 outputs", "critical", "Re-run v2.3 with identity breakdown metrics"


def build_gated_score_comparison(v23_gated, v23_breakdown):
    results = []

    if v23_gated is not None:
        for r in v23_gated:
            model = r["model"]
            fm = r["feature_mode"]
            clean_skill = r.get("clean_skill", "NaN")
            cf_skill = r.get("cf_skill", "NaN")
            comp_skill = r.get("comp_skill", "NaN")
            id_overall = r.get("identity_overall", "NaN")
            id_swap = r.get("identity_swap_only", "NaN")
            gated_overall = r.get("gated_score_overall_id", "NaN")
            gated_swap = r.get("gated_score_swap_only_id", "NaN")

            try:
                drop = float(gated_overall) - float(gated_swap)
            except (ValueError, TypeError):
                drop = "NaN"

            try:
                overall_val = float(id_overall)
                swap_val = float(id_swap)
                bias = overall_val - swap_val > 0.1
            except (ValueError, TypeError):
                bias = "NaN"

            results.append({
                "model": model,
                "feature_mode": fm,
                "clean_skill": clean_skill,
                "cf_skill": cf_skill,
                "comp_skill": comp_skill,
                "identity_overall": id_overall,
                "identity_swap_only": id_swap,
                "gated_score_overall_id": gated_overall,
                "gated_score_swap_only_id": gated_swap,
                "score_drop": f"{drop:.4f}" if isinstance(drop, float) else "NaN",
                "no_swap_bias_flag": str(bias),
            })

    if v23_breakdown is not None and not v23_gated:
        seen = set()
        for r in v23_breakdown:
            if r["split_name"] != "identity_test_mixed":
                continue
            key = (r["model"], r["feature_mode"])
            if key in seen:
                continue
            seen.add(key)

            id_overall = r.get("identity_overall", "NaN")
            id_swap = r.get("identity_swap_only", "NaN")

            try:
                overall_val = float(id_overall)
                swap_val = float(id_swap)
                bias = overall_val - swap_val > 0.1
            except (ValueError, TypeError):
                bias = "NaN"

            results.append({
                "model": r["model"],
                "feature_mode": r["feature_mode"],
                "clean_skill": "NaN",
                "cf_skill": "NaN",
                "comp_skill": "NaN",
                "identity_overall": id_overall,
                "identity_swap_only": id_swap,
                "gated_score_overall_id": "NaN",
                "gated_score_swap_only_id": "NaN",
                "score_drop": "NaN",
                "no_swap_bias_flag": str(bias),
            })

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SVT-v2.4 Identity Health Gate")
    print("=" * 60)

    # =========================================================================
    # Load all data sources
    # =========================================================================
    print("\n--- Loading data sources ---")

    v21_audit = read_csv(V21_AUDIT_SUMMARY)
    v22_health = read_csv(V22_DATASET_HEALTH)
    v22_feat = read_csv(V22_FEATURE_BASELINE)
    v22_perm = read_csv(V22_LABEL_PERMUTATION)
    v23_breakdown = read_csv(V23_IDENTITY_BREAKDOWN)
    v23_gated = read_csv(V23_GATED_SCORE)

    sources = {
        "v2.1 audit_summary": v21_audit is not None,
        "v2.2 dataset_health": v22_health is not None,
        "v2.2 feature_baseline": v22_feat is not None,
        "v2.2 label_permutation": v22_perm is not None,
        "v2.3 identity_breakdown": v23_breakdown is not None,
        "v2.3 gated_score": v23_gated is not None,
    }
    for name, ok in sources.items():
        print(f"  {name}: {'LOADED' if ok else 'MISSING'}")

    # =========================================================================
    # Run all health checks
    # =========================================================================
    print("\n--- Running health checks ---")

    checks = []

    checks.append(list(check_label_permutation_sanity(v22_perm, v22_health)))
    checks.append(list(check_featureless_identifiability(v22_feat, v22_health)))
    checks.append(list(check_feature_bearing_identifiability(v22_feat, v22_health)))
    checks.append(list(check_object_order_leakage(v21_audit, v22_health)))
    checks.append(list(check_swap_only_split_valid(v23_breakdown)))
    checks.append(list(check_no_swap_only_split_valid(v23_breakdown)))
    checks.append(list(check_no_swap_bias(v23_breakdown)))
    checks.append(list(check_identity_metric_policy(v23_breakdown)))

    gate_results = []
    for check in checks:
        gate_results.append({
            "check_name": check[0],
            "status": check[1],
            "value": check[2],
            "expected": check[3],
            "severity": check[4],
            "required_action": check[5],
        })
        print(f"  {check[0]}: {check[1]} (value={check[2]}, severity={check[4]})")

    save_csv(gate_results, "identity_health_gate.csv",
             ["check_name", "status", "value", "expected", "severity", "required_action"])

    # =========================================================================
    # Gated Score Comparison
    # =========================================================================
    print("\n--- Gated Score Comparison ---")

    comparison = build_gated_score_comparison(v23_gated, v23_breakdown)
    save_csv(comparison, "gated_score_comparison.csv",
             ["model", "feature_mode", "clean_skill", "cf_skill", "comp_skill",
              "identity_overall", "identity_swap_only",
              "gated_score_overall_id", "gated_score_swap_only_id",
              "score_drop", "no_swap_bias_flag"])

    for r in comparison:
        print(f"  {r['model']} ({r['feature_mode']}): overall={r['identity_overall']} swap={r['identity_swap_only']} bias={r['no_swap_bias_flag']}")

    # =========================================================================
    # Determine can_proceed_to_v3
    # =========================================================================
    critical_checks = [r for r in gate_results if r["severity"] == "critical"]
    missing_checks = [r for r in gate_results if r["status"] == "MISSING"]
    warning_checks = [r for r in gate_results if r["severity"] == "warning"]

    can_proceed = len(critical_checks) == 0 and len(missing_checks) == 0

    # =========================================================================
    # Identity Health Summary
    # =========================================================================
    if can_proceed:
        verdict = "Identity health gate passed. SVT-v3 may proceed, but all identity-based claims must use swap-only identity as the primary metric."
    else:
        verdict = "Identity health gate failed. Do not proceed to SVT-v3 until the listed issues are fixed."

    summary_lines = [
        f"# Identity Health Gate Summary\n",
        f"**Verdict**: {verdict}\n",
        f"**can_proceed_to_v3**: {can_proceed}\n",
        f"\n## Check Results\n",
    ]

    for r in gate_results:
        icon = "PASS" if r["status"] == "PASS" else ("WARN" if r["severity"] == "warning" else "FAIL")
        summary_lines.append(f"- [{icon}] {r['check_name']}: {r['status']} (value={r['value']}, severity={r['severity']})")

    if critical_checks:
        summary_lines.append(f"\n## Critical Issues\n")
        for r in critical_checks:
            summary_lines.append(f"- {r['check_name']}: {r['required_action']}")

    if missing_checks:
        summary_lines.append(f"\n## Missing Data\n")
        for r in missing_checks:
            summary_lines.append(f"- {r['check_name']}: {r['required_action']}")

    if warning_checks:
        summary_lines.append(f"\n## Warnings (do not block v3)\n")
        for r in warning_checks:
            summary_lines.append(f"- {r['check_name']}: {r['required_action']}")

    summary_path = os.path.join(OUTPUT_DIR, "identity_health_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # =========================================================================
    # README
    # =========================================================================
    readme_lines = [
        "# SVT-v2.4 Identity Health Gate Report\n",
        "## 1. Purpose\n",
        "The Identity Health Gate is a **mandatory pre-check** for all future SVT-v3 experiments. "
        "It consolidates the findings from v2.1 (Leakage Audit), v2.2 (Dataset Fix), and v2.3 (Swap-Only Stress Test) "
        "into a single automated gate.\n",
        "No SVT-v3 experiment should be run without this gate passing.\n",
        "## 2. Why Overall Identity Is Not Enough\n",
        "v2.3 demonstrated that **overall identity accuracy is inflated by no-swap episodes**:\n",
        "- RawKNN overall identity = 0.800, but swap-only identity = 0.689\n",
        "- RawDeltaKNN overall identity = 0.547, but swap-only identity = 0.189\n",
        "- No-swap episodes are trivially easy (just predict \"no swap\"), achieving >0.94 accuracy\n",
        "- Swap episodes are the hard part, and models perform much worse on them\n",
        "**Overall identity measures prediction quality on no-swap episodes, not identity tracking.**\n",
        "## 3. Required Policy\n",
        "All future SVT identity claims **must**:\n",
        "1. Report `identity_swap_only` as the **primary** identity metric\n",
        "2. Report `balanced_identity` as a secondary summary\n",
        "3. Report `swap_detect_recall` and `swap_false_positive_rate`\n",
        "4. Always specify `feature_mode` (featureless vs feature-bearing)\n",
        "5. Never use `identity_overall` alone as evidence of identity understanding\n",
        "6. Never conflate featureless near-random results with model failure\n",
        "7. Never conflate feature-bearing position-only KNN failure with task unsolvability\n",
        "## 4. Health Gate Result\n",
        f"**can_proceed_to_v3**: `{can_proceed}`\n",
    ]

    if not can_proceed:
        readme_lines.append("## 5. Issues That Must Be Fixed Before v3\n")
        for r in gate_results:
            if r["severity"] == "critical" or r["status"] == "MISSING":
                readme_lines.append(f"- **{r['check_name']}** (severity={r['severity']}): {r['required_action']}")
    else:
        readme_lines.append("## 5. Warnings (do not block v3)\n")
        for r in gate_results:
            if r["severity"] == "warning":
                readme_lines.append(f"- **{r['check_name']}**: {r['required_action']}")

    readme_lines.append("\n## 6. Check Details\n")
    readme_lines.append("| Check | Status | Value | Expected | Severity | Action |")
    readme_lines.append("|-------|--------|-------|----------|----------|--------|")
    for r in gate_results:
        readme_lines.append(f"| {r['check_name']} | {r['status']} | {r['value']} | {r['expected']} | {r['severity']} | {r['required_action']} |")

    readme_lines.append("\n## 7. Gated Score Comparison\n")
    readme_lines.append("| Model | Feature Mode | Identity Overall | Identity Swap-Only | Bias Flag |")
    readme_lines.append("|-------|-------------|-----------------|-------------------|-----------|")
    for r in comparison:
        readme_lines.append(f"| {r['model']} | {r['feature_mode']} | {r['identity_overall']} | {r['identity_swap_only']} | {r['no_swap_bias_flag']} |")

    readme_path = os.path.join(OUTPUT_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines))

    # =========================================================================
    # Final Output
    # =========================================================================
    print("\n" + "=" * 60)
    print("IDENTITY HEALTH GATE RESULT")
    print("=" * 60)
    for r in gate_results:
        icon = "OK" if r["status"] == "PASS" else ("!!" if r["severity"] in ["critical"] else "~~")
        print(f"  [{icon}] {r['check_name']}: {r['status']} (severity={r['severity']})")
    print(f"\ncan_proceed_to_v3: {can_proceed}")
    print(f"\n{verdict}")


if __name__ == "__main__":
    main()
