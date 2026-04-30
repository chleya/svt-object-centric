# SVT-v3: Nonlinear Feature-Bearing OOD Benchmark

## 1. Purpose

v3 tests whether models can maintain **prediction skill, swap-only identity tracking, occlusion/crossing robustness, and OOD dynamics transfer** under nonlinear force fields.

This is NOT a victory report. v3 builds on the identity pipeline fixes from v2.1-v2.4 and runs on a healthy identity gate.

## 2. Health Gate Result

**Status**: PASS

All identity health checks passed.

## 3. Why Swap-Only Identity Is Primary

Per v2.4 policy:
- `identity_swap_only` is the **primary** identity metric
- `identity_overall` is diagnostic only — it is inflated by no-swap episodes
- No-swap bias gap > 0.1 must be flagged
- Using `identity_overall` alone as evidence of identity understanding is prohibited

## 4. Dataset

- **Force fields**: Train=attractor, OOD Test=vortex
- **Feature mode**: feature_bearing (with featureless control)
- **Object order**: randomized by default
- **Train/test split**: disjoint initialization
- **Identity splits**: mixed (50%), swap-only (100%), no-swap-only (0%)
- **OOD test**: different force type
- **Crossing/occlusion test**: forced crossing + occlusion episodes

## 5. Main Results

### Prediction Skill
- LastVelocityBaseline clean_skill: -25.2114
- LastVelocityBaseline below 0.5 — nonlinear dynamics successfully break linear extrapolation

### Identity (swap-only, feature_bearing)
- FeatureAwareBaseline swap-only: 1.0000
- RandomBaseline swap-only: 0.5288

### OOD Transfer
- KNN models show OOD skill drop — force field change is detectable

### No-Swap Bias
- No-swap bias detected in some models

## 6. Interpretation Rules

- featureless identity near-random ≠ model failure (task is unidentifiable)
- feature_bearing swap-only failure = identity tracking insufficient
- LastVelocityBaseline skill drop = velocity shortcut weakened by nonlinearity
- RawKNN ID high but OOD low = retrieval, not structure
- DeltaKNN prediction high but identity_swap_only low = Delta-Output Identity Paradox continues

## 7. Final Recommendation

**proceed_to_v3_learned_models**
