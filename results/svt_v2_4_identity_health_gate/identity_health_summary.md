# Identity Health Gate Summary

**Verdict**: Identity health gate passed. SVT-v3 may proceed, but all identity-based claims must use swap-only identity as the primary metric.

**can_proceed_to_v3**: True


## Check Results

- [PASS] label_permutation_sanity: PASS (value=0.5232, severity=none)
- [PASS] featureless_identifiability: PASS (value=0.5000, severity=none)
- [PASS] feature_bearing_identifiability: PASS (value=1.0000, severity=none)
- [PASS] object_order_leakage_checked: PASS (value=leakage_found_and_fixed, severity=warning)
- [PASS] swap_only_split_valid: PASS (value=100% swap (no_swap=nan, FAB_fb_swap=1.0), severity=none)
- [PASS] no_swap_only_split_valid: PASS (value=0% swap (swap_only=nan), severity=none)
- [WARN] no_swap_bias_detected: WARNING (value=max_gap=0.226, severity=warning)
- [PASS] identity_metric_policy: PASS (value=identity_swap_only_available, severity=none)

## Warnings (do not block v3)

- object_order_leakage_checked: none (leakage was found in v2.1 but fixed in v2.2)
- no_swap_bias_detected: Use identity_swap_only as primary metric; overall identity is inflated by no-swap episodes