# SVT-v2.3 Swap-Only Identity Stress Test Report

## 1. Purpose

SVT-v2.3 is an **identity metric correction**, not a new model experiment. The goal is to fix a critical flaw in how identity accuracy is measured:

**Overall identity accuracy is inflated by no-swap episodes and cannot serve as the primary identity metric.**

v2.2 showed RawKNN overall identity = 0.800, but breakdown revealed:
- No-swap episodes: 95.8% accuracy
- Swap episodes: 63.8% accuracy

The 0.800 overall is misleading — it reflects the model's ability to correctly predict "no swap happened" (trivial), not its ability to detect swaps (the hard part).

## 2. What Changed

### 2.1 Identity Breakdown Metric (`metrics/identity_breakdown.py`)

New function `compute_identity_breakdown()` that decomposes identity accuracy into:

| Metric | Definition |
|--------|-----------|
| `identity_overall` | Accuracy across all episodes |
| `identity_no_swap` | Accuracy on episodes where no swap occurred |
| `identity_swap_only` | Accuracy on episodes where a swap occurred |
| `swap_detect_recall` | Fraction of true swaps correctly detected |
| `swap_false_positive_rate` | Fraction of no-swap episodes falsely labeled as swap |
| `balanced_identity` | Average of no-swap and swap-only accuracy |

**Convention**: For 2-object episodes, `true_identity[:, 0] == 1` indicates a swap episode; `true_identity[:, 0] == 0` indicates no swap.

### 2.2 Swap-Only and No-Swap-Only Test Splits

Extended `generate_dataset_v22` with three identity test splits:

| Split | Swap Probability | Purpose |
|-------|-----------------|---------|
| `identity_test_mixed` | 0.5 | Standard mixed evaluation |
| `identity_test_swap_only` | 1.0 | All episodes have swaps |
| `identity_test_no_swap_only` | 0.0 | No episodes have swaps |

Implementation uses rejection sampling: episodes are regenerated until the swap state matches the target probability. This ensures:
- `identity_test_swap_only`: 100% swap episodes (90/90 in our run)
- `identity_test_no_swap_only`: 0% swap episodes (0/104 in our run)

### 2.3 Gated SVT Score v2.3

Two variants of the gated score:
- `gated_score_overall_id`: Uses `identity_overall` (legacy)
- `gated_score_swap_only_id`: Uses `identity_swap_only` (recommended)

## 3. Core Results

### 3.1 Identity Breakdown on Mixed Split (featureless)

| Model | Overall | No-Swap | Swap-Only | Swap Recall | FPR | Balanced |
|-------|---------|---------|-----------|-------------|-----|----------|
| RawTrajectoryKNN | **0.800** | 0.958 | **0.638** | 0.638 | 0.042 | 0.798 |
| RawDeltaKNN | 0.547 | 0.958 | **0.128** | 0.128 | 0.042 | 0.543 |
| TranslationNormKNN | 0.758 | 0.979 | **0.532** | 0.532 | 0.021 | 0.756 |
| RandomBaseline | 0.516 | 0.500 | 0.532 | 0.532 | 0.500 | 0.516 |
| FeatureAwareBaseline | 0.516 | 0.500 | 0.532 | 0.532 | 0.500 | 0.516 |

### 3.2 Identity on Swap-Only Split (featureless)

| Model | Swap-Only Identity | Status |
|-------|-------------------|--------|
| RawTrajectoryKNN | **0.689** | NO_SWAP_BIAS |
| RawDeltaKNN | 0.189 | PASS |
| TranslationNormKNN | **0.667** | NO_SWAP_BIAS |
| RandomBaseline | 0.533 | PASS |
| FeatureAwareBaseline | 0.533 | PASS |

### 3.3 Identity on Swap-Only Split (feature_bearing)

| Model | Swap-Only Identity | Status |
|-------|-------------------|--------|
| RawTrajectoryKNN | 0.689 | NO_SWAP_BIAS |
| RawDeltaKNN | 0.189 | PASS |
| TranslationNormKNN | 0.667 | NO_SWAP_BIAS |
| RandomBaseline | 0.533 | PASS |
| FeatureAwareBaseline | **1.000** | PASS |

### 3.4 Identity on No-Swap-Only Split (featureless)

| Model | No-Swap Identity |
|-------|-----------------|
| RawTrajectoryKNN | 0.942 |
| RawDeltaKNN | 0.971 |
| TranslationNormKNN | 0.981 |
| RandomBaseline | 0.471 |
| FeatureAwareBaseline | 0.471 |

## 4. Why Overall Identity Cannot Be the Primary Metric

### The No-Swap Bias Problem

In a mixed test set with 50% swap rate:
- No-swap episodes are "easy" — the correct answer is always "no swap" ([0,1])
- Swap episodes are "hard" — the model must detect that identities were exchanged
- A model that always predicts "no swap" achieves ~50% overall accuracy (all no-swap correct, all swap wrong)
- A model with good prediction but poor swap detection achieves high overall accuracy because no-swap episodes dominate the signal

### Evidence from v2.3

| Model | Mixed Overall | Swap-Only | Gap |
|-------|--------------|-----------|-----|
| RawTrajectoryKNN | 0.800 | 0.689 | 0.111 |
| TranslationNormKNN | 0.758 | 0.667 | 0.091 |
| RawDeltaKNN | 0.547 | 0.189 | 0.358 |

RawDeltaKNN has the largest gap: 0.547 overall vs 0.189 swap-only. This is the Delta-Output Identity Paradox in stark relief — delta-output prediction is very good at matching no-swap trajectories (0.971 on no-swap-only) but catastrophically bad at detecting swaps.

### No-Swap Episodes Are Not Identity Tests

On the no-swap-only split, all KNN models achieve >0.94 accuracy. But this is not identity tracking — it's trajectory prediction quality. When no swap occurs, the "correct" identity assignment is simply the default [0,1], which is what any reasonable prediction will match.

**No-swap accuracy measures prediction quality, not identity understanding.**

## 5. Featureless vs Feature-Bearing Swap-Only Identity

### Featureless (not identifiable)

- FeatureAwareBaseline swap-only: **0.533** (random)
- RawKNN swap-only: **0.689** (above random, but not genuine identity tracking)
- This is NOT a model failure — the task is genuinely unidentifiable without features
- RawKNN's 0.689 comes from trajectory-matching heuristics, not identity understanding

### Feature-Bearing (identifiable)

- FeatureAwareBaseline swap-only: **1.000** (perfect)
- RawKNN swap-only: **0.689** (same as featureless — KNN ignores features)
- The identity task IS solvable when features are available
- KNN's failure is because it only uses positions, not features
- This does NOT mean the task is unsolvable — it means position-only methods are insufficient

**Critical distinction**: Featureless swap-only near-random ≠ model failure. Feature-bearing swap-only near-random with features available ≠ task unsolvable.

## 6. Swap Detection Analysis

### Swap Detect Recall

On the mixed split (featureless):

| Model | Swap Recall | FPR | Interpretation |
|-------|------------|-----|----------------|
| RawTrajectoryKNN | 0.638 | 0.042 | Detects 64% of swaps, rarely false-alarms |
| RawDeltaKNN | 0.128 | 0.042 | Detects only 13% of swaps |
| TranslationNormKNN | 0.532 | 0.021 | Detects 53% of swaps |
| RandomBaseline | 0.532 | 0.500 | Random guessing |

RawKNN's low FPR (0.042) means it almost never incorrectly labels a no-swap episode as a swap. But its recall (0.638) means it misses 36% of actual swaps. The high overall accuracy comes from correctly identifying no-swap episodes, not from detecting swaps.

## 7. Gated SVT Score v2.3

All gated scores are 0.000 because clean_skill < 0.5 for all KNN models on the enhanced environment. This is expected — k-NN baselines are not strong enough predictors on this physics-rich environment.

The key comparison is between `gated_score_overall_id` and `gated_score_swap_only_id`:
- Both are 0.000 (gated by prediction quality)
- But the identity components differ: overall=0.800 vs swap_only=0.638
- When models eventually pass the prediction gate, the swap-only identity will produce a more honest gated score

**Recommendation**: GatedSVTScore must use `identity_swap_only` as the identity component, not `identity_overall`.

## 8. Pass/Fail Criteria

| Criterion | Result | Status |
|-----------|--------|--------|
| FeatureAwareBaseline feature_bearing swap_only >= 0.95 | 1.000 | PASS |
| RandomIdentityBaseline swap_only ≈ 0.5 | 0.533 | PASS |
| RawKNN overall high but swap_only low → NO_SWAP_BIAS | 0.800 vs 0.689 | FLAGGED |
| GatedSVTScore uses swap_only identity | Implemented | PASS |

## 9. Required Changes for All Future SVT Work

1. **All identity claims must report swap-only identity** — overall identity alone is insufficient and misleading.

2. **Balanced identity** (average of swap and no-swap accuracy) is a better single-number summary than overall identity.

3. **Swap detect recall and FPR** must be reported alongside accuracy to distinguish between "correctly detecting swaps" and "correctly predicting no-swap."

4. **Feature mode must always be reported** — featureless and feature-bearing results must never be conflated.

5. **Featureless swap-only near-random is expected** — it reflects task unidentifiability, not model failure.

6. **Feature-bearing swap-only is the true identity test** — only this setting can distinguish genuine identity tracking from prediction heuristics.

7. **No-swap-only accuracy is not an identity metric** — it measures prediction quality, not identity understanding.

## 10. Recommendation

**Proceed to v3 with swap-only metric as the primary identity measure.**

The swap-only identity stress test has:
- Confirmed the no-swap bias in overall identity (RawKNN: 0.800 → 0.689)
- Validated FeatureAwareBaseline on feature-bearing swap-only (1.000)
- Validated RandomBaseline on swap-only (~0.5)
- Established that swap-only identity is the correct metric for identity evaluation

Before v3 experiments, ensure:
- All models report `identity_swap_only` as the primary identity metric
- `balanced_identity` is reported as a secondary summary
- Swap detect recall and FPR are included in all identity reports
- Feature-bearing swap-only is used as the gold standard for identity tracking
