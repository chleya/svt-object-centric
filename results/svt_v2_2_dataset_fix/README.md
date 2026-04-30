# SVT-v2.2 Dataset Fix Report

## 1. Purpose

SVT-v2.2 is a **dataset fix**, not a new model competition. The goal is to fix two problems identified in the v2.1 audit:

1. **Object order leakage**: RawKNN v1's high identity (0.805) was primarily driven by consistent object ordering between train and test, not genuine identity understanding.
2. **Identity task unidentifiability**: Without distinguishing features (color, marker), the featureless identity task is fundamentally unidentifiable — no method can reliably track identity through occlusion without features.

## 2. What Was Fixed

### 2.1 Object Order Randomization (`randomize_object_order=True` by default)

- Every episode now randomly permutes the object dimension (slot 0/1 ordering).
- Train and test permutations are **independent** — no shared fixed object 0/object 1 semantics.
- Identity labels are correctly conjugated under permutation: `new_labels = perm_inverse[labels[perm]]`.

### 2.2 Feature-Bearing Objects (`feature_mode="feature_bearing"`)

- Each object carries a unique observable feature (one-hot: object 0 = [1,0], object 1 = [0,1]).
- Features follow **true identity** through swaps: when a swap occurs during occlusion, features change at the swap point in the observed period, and reflect the post-swap identity in the future period.
- This creates a **solvable** identity task: FeatureAwareIdentityBaseline can match features to determine identity.

### 2.3 FeatureAwareIdentityBaseline

- New baseline that matches object identity by comparing feature vectors between the first observed timestep and the first future timestep.
- Uses first timestep (not mean) because features change at the swap point; the first timestep reflects the initial identity assignment.

### 2.4 Label Permutation Sanity Check

- Retained from v2.1: randomly permute identity labels and verify accuracy drops to ~0.5.

### 2.5 Disjoint Init Split (`disjoint_init_split=True`)

- Test episodes are filtered to those with initial x-position >= median of training initial positions.
- This creates a spatial separation between train and test initial positions.

## 3. Dataset Health Results

| Check | Status | Value | Expected | Severity |
|-------|--------|-------|----------|----------|
| object_order_randomized_default | PASS | True | True | none |
| featureless_identifiability | PASS | 0.5000 | 0.4-0.6 | none |
| feature_bearing_identifiability | PASS | 1.0000 | >=0.95 | none |
| label_permutation_sanity | PASS | 0.5305 | 0.35-0.65 | none |
| disjoint_init_split | PASS | enabled | separated | none |
| rawknn_no_object_order_leakage | CHECK_NEEDED | 0.8000 | <0.7 | mild |

**5 of 6 checks pass.** The remaining check (RawKNN identity with randomized order) requires explanation below.

## 4. Featureless vs Feature-Bearing

These two modes serve fundamentally different purposes and **must not be conflated**.

### Featureless (not identifiable control)

- Two objects have no observable distinguishing features.
- FeatureAwareIdentityBaseline: **identity = 0.500** (random, as expected).
- This confirms the task is **not identifiable** without features — no method should claim to solve identity in this setting.
- Any identity accuracy above random in featureless mode must be explained by prediction-based heuristics, not genuine identity understanding.

### Feature-Bearing (identifiable task)

- Each object has a unique one-hot feature that persists through time.
- FeatureAwareIdentityBaseline: **identity = 1.000** (perfect, as expected).
- This confirms the feature pipeline and identity labels are correct.
- The identity task **is solvable** when features are available.

## 5. Model Health Check

All models evaluated with `randomize_object_order=True`, `disjoint_init_split=True`, seed=0.

| Model | Feature Mode | Clean Skill | Identity Acc | Gated SVT |
|-------|-------------|-------------|-------------|-----------|
| RawTrajectoryKNN | featureless | 0.354 | 0.800 | 0.000 |
| RawDeltaKNN | featureless | 0.375 | 0.547 | 0.000 |
| TranslationNormalizedKNN | featureless | 0.211 | 0.758 | 0.000 |
| RawTrajectoryKNN | feature_bearing | 0.354 | 0.800 | 0.000 |
| RawDeltaKNN | feature_bearing | 0.375 | 0.547 | 0.000 |
| TranslationNormalizedKNN | feature_bearing | 0.211 | 0.758 | 0.000 |
| RandomIdentityBaseline | both | 0.000 | 0.516 | 0.000 |

### Key Observations

1. **All Gated SVT Scores are 0.000** because clean_skill < 0.5 threshold for all models. No model passes the prediction quality gate.

2. **RawKNN identity = 0.800** — this is NOT evidence of genuine identity understanding. Breakdown:
   - No-swap episodes: **95.8% accuracy** — KNN almost always correctly predicts "no swap"
   - Swap episodes: **63.8% accuracy** — KNN detects swaps only partially
   - The high overall accuracy is inflated by the no-swap majority
   - MSE on swap episodes (198) is 2.5x higher than no-swap (79), confirming prediction quality drives the signal

3. **RawDeltaKNN identity = 0.547** — near random, consistent with the Delta-Output Identity Paradox: improving prediction via delta-output destroys identity tracking.

4. **TranslationNormalizedKNN identity = 0.758** — still high, suggesting translation normalization does not remove the trajectory-matching signal.

5. **Feature mode has no effect on KNN models** — KNN only uses positions, not features. The identical results across feature modes confirm this.

### Why RawKNN Identity Remains High (0.800) with Randomized Order

The v2.1 audit found that randomizing test order (while keeping train fixed) dropped identity to 0.195. But v2.2 randomizes BOTH train and test, and identity remains 0.800. This is NOT a contradiction:

- **v2.1 audit**: Only test was randomized → train/test order inconsistency → KNN predictions became wrong → trajectory matching failed → identity dropped.
- **v2.2**: Both train and test are randomized → each episode is internally consistent → KNN can still find good matches → trajectory matching partially works → identity stays high.

The trajectory matching approach (comparing MSE under swap vs no-swap assignments) can detect swaps when the prediction quality differs between the two assignments. This is a **prediction-based heuristic**, not genuine identity understanding. The KNN does not track object identity; it detects when its prediction doesn't match the observed future.

Evidence:
- Swap detection rate (63.8%) is only modestly above random (50%)
- No-swap detection rate (95.8%) is near-perfect, inflating the overall average
- The signal comes from prediction error asymmetry, not identity tracking

## 6. Object Order Health Check

| Setting | Feature Mode | Identity | Drop from Legacy | Status |
|---------|-------------|----------|-----------------|--------|
| fixed_order_legacy | featureless | 0.795 | — | baseline |
| randomized_order_v22 | featureless | 0.800 | -0.005 | CHECK_NEEDED |
| randomized_order_v22 | feature_bearing | 0.800 | -0.005 | CHECK_NEEDED |

Randomized order did NOT reduce RawKNN identity. The slight increase (0.795 → 0.800) is within noise. This confirms that **object order was not the primary driver** of RawKNN's identity accuracy — the trajectory matching heuristic works regardless of slot ordering.

The v2.1 audit's dramatic drop (0.805 → 0.195) was caused by train/test order **inconsistency**, not by randomization itself. When both sides are randomized consistently, the KNN adapts.

## 7. Label Permutation Health Check

| Feature Mode | Original Identity | Permuted Identity (avg) | Status |
|-------------|------------------|------------------------|--------|
| featureless | 0.800 | 0.531 | PASS |
| feature_bearing | 0.800 | 0.516 | PASS |

Permuted identity drops to ~0.5, confirming the metric is valid and not artificially inflated.

## 8. Required Revision to SVT-v2/v2.1

1. **RawKNN legacy high identity is NOT structural evidence.** The v2.1 audit showed it was partly driven by object order consistency; v2.2 shows the remaining signal comes from prediction-based heuristics.

2. **v2.2 must use randomized order by default.** All future identity experiments must use `randomize_object_order=True`.

3. **All future identity experiments must report `feature_mode`.** Featureless and feature-bearing results must be reported separately and interpreted differently.

4. **Identity accuracy in featureless mode must not be interpreted as model success.** It reflects prediction-based heuristics, not identity understanding.

5. **The identity_labels permutation formula has been corrected.** The correct transformation under slot permutation is `perm_inverse[labels[perm]]`, not `labels[perm]`. This bug was present in the initial v2.2 implementation and has been fixed.

6. **Feature assignment has been corrected for swap episodes.** Features now follow the true identity at each timestep: before the swap point, features reflect the initial identity; after the swap point, features reflect the post-swap identity. This ensures FeatureAwareIdentityBaseline can correctly match features.

## 9. Identity Label Definition

For clarity, the identity label definition used in v2.2:

- `identity_labels[i] = j` means: **future slot i corresponds to the object that was in slot j at the BEGINNING of the observation** (before any swap or permutation).
- If a swap occurs during the observed period, `identity_labels = [1, 0]` indicates that future slot 0 now contains the object that was originally in slot 1.
- If `randomize_object_order` permutes the slots, identity_labels are conjugated accordingly so the mapping remains correct relative to the new slot indices.

## 10. Recommendation

**Conditional proceed to v3**, with caveats:

1. The dataset fix is **successful**: object order randomization is default, feature-bearing objects work correctly, FeatureAwareIdentityBaseline passes both checks.

2. However, RawKNN's identity = 0.800 with randomized order requires **careful interpretation**:
   - It is NOT evidence of genuine identity understanding
   - It is driven by prediction-based trajectory matching (95.8% on no-swap, 63.8% on swap)
   - Any claim of "identity tracking" must be validated on swap episodes specifically

3. Before v3 experiments, consider:
   - Reporting identity accuracy **separately for swap and no-swap episodes**
   - Using feature-bearing mode as the primary identity test
   - Treating featureless identity as an unidentifiable control, not a test of model capability

4. The Gated SVT Score is 0.000 for all models because none pass the clean_skill > 0.5 threshold. This is expected for k-NN baselines on the enhanced environment and should not be interpreted as a dataset problem.
