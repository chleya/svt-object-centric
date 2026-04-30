# SVT-v2.4 Identity Health Gate Report

## 1. Purpose

The Identity Health Gate is a **mandatory pre-check** for all future SVT-v3 experiments. It consolidates the findings from v2.1 (Leakage Audit), v2.2 (Dataset Fix), and v2.3 (Swap-Only Stress Test) into a single automated gate.

No SVT-v3 experiment should be run without this gate passing.

## 2. Why Overall Identity Is Not Enough

v2.3 demonstrated that **overall identity accuracy is inflated by no-swap episodes**:

- RawKNN overall identity = 0.800, but swap-only identity = 0.689

- RawDeltaKNN overall identity = 0.547, but swap-only identity = 0.189

- No-swap episodes are trivially easy (just predict "no swap"), achieving >0.94 accuracy

- Swap episodes are the hard part, and models perform much worse on them

**Overall identity measures prediction quality on no-swap episodes, not identity tracking.**

## 3. Required Policy

All future SVT identity claims **must**:

1. Report `identity_swap_only` as the **primary** identity metric

2. Report `balanced_identity` as a secondary summary

3. Report `swap_detect_recall` and `swap_false_positive_rate`

4. Always specify `feature_mode` (featureless vs feature-bearing)

5. Never use `identity_overall` alone as evidence of identity understanding

6. Never conflate featureless near-random results with model failure

7. Never conflate feature-bearing position-only KNN failure with task unsolvability

## 4. Health Gate Result

**can_proceed_to_v3**: `True`

## 5. Warnings (do not block v3)

- **object_order_leakage_checked**: none (leakage was found in v2.1 but fixed in v2.2)
- **no_swap_bias_detected**: Use identity_swap_only as primary metric; overall identity is inflated by no-swap episodes

## 6. Check Details

| Check | Status | Value | Expected | Severity | Action |
|-------|--------|-------|----------|----------|--------|
| label_permutation_sanity | PASS | 0.5232 | permuted identity around 0.5 | none | none |
| featureless_identifiability | PASS | 0.5000 | FeatureAwareBaseline featureless around 0.5 | none | none |
| feature_bearing_identifiability | PASS | 1.0000 | FeatureAwareBaseline feature_bearing >= 0.95 | none | none |
| object_order_leakage_checked | PASS | leakage_found_and_fixed | v2.1/v2.2 audit exists and order randomized | warning | none (leakage was found in v2.1 but fixed in v2.2) |
| swap_only_split_valid | PASS | 100% swap (no_swap=nan, FAB_fb_swap=1.0) | swap_only split contains 100% swap episodes | none | none |
| no_swap_only_split_valid | PASS | 0% swap (swap_only=nan) | no_swap_only split contains 0% swap episodes | none | none |
| no_swap_bias_detected | WARNING | max_gap=0.226 | overall identity not much greater than swap_only identity | warning | Use identity_swap_only as primary metric; overall identity is inflated by no-swap episodes |
| identity_metric_policy | PASS | identity_swap_only_available | identity_swap_only present in v2.3 outputs | none | none |

## 7. Gated Score Comparison

| Model | Feature Mode | Identity Overall | Identity Swap-Only | Bias Flag |
|-------|-------------|-----------------|-------------------|-----------|
| RawTrajectoryKNN | featureless | 0.8000 | 0.6383 | True |
| RawDeltaKNN | featureless | 0.5474 | 0.1277 | True |
| TranslationNormalizedKNN | featureless | 0.7579 | 0.5319 | True |
| RawTrajectoryKNN | feature_bearing | 0.8000 | 0.6383 | True |
| RawDeltaKNN | feature_bearing | 0.5474 | 0.1277 | True |
| TranslationNormalizedKNN | feature_bearing | 0.7579 | 0.5319 | True |