# SVT-v2 Phase 0 Report: k-NN Retrieval Attack v1 & v2

**Date**: 2026-04-27
**Config**: smoke.yaml (1000 train, 200 test per split)
**Status**: Phase 0 Complete (v1 + v2)

---

## 1. Oracle Upper Bound

| Split | MSE | Skill Score | Identity |
|-------|-----|-------------|----------|
| clean_test | 0.000 | 1.000 | 1.000 |
| counterfactual_test | 284.944 | 0.232 | 1.000 |
| compositional_test | 0.000 | 1.000 | 1.000 |
| identity_test | 38.599 | 0.874 | 1.000 |

**Interpretation**: The task is solvable. Oracle achieves perfect prediction on clean/compositional splits. Counterfactual is harder (Oracle doesn't know the intervention), and identity_test has moderate MSE due to position swaps.

---

## 2. k-NN v1 vs v2 Comparison

### v1 (absolute-output) Best Results:

| Model | Best k | Clean Skill | Identity | Gated Score |
|-------|--------|-------------|----------|-------------|
| TranslationNormalizedKNN | 5 | 0.852 | 0.810 | 0.269 |
| RawTrajectoryKNN | 3 | 0.746 | **0.830** | 0.201 |
| VelocityOnlyKNN | 1 | -0.198 | 0.645 | 0.000 |

### v2 (delta-output) Best Results:

| Model | Best k | Clean Skill | Identity | Gated Score |
|-------|--------|-------------|----------|-------------|
| VelocityDeltaKNN | 10 | **0.930** | 0.555 | 0.117 |
| TranslationNormalizedDeltaKNN | 5 | 0.919 | 0.570 | 0.127 |
| RawDeltaKNN | 5 | 0.907 | 0.570 | 0.128 |
| LastVelocityBaseline | — | 0.944 | 0.560 | 0.000 |

### Key comparison:

| Metric | v1 Best | v2 Best | Winner |
|--------|---------|---------|--------|
| Clean Skill | 0.852 | **0.930** | v2 (delta-output) |
| Identity | **0.830** | 0.570 | v1 (absolute-output) |
| Gated Score | **0.269** | 0.128 | v1 (due to identity) |

---

## 3. Critical Finding: Delta-Output Identity Paradox

### The paradox:
- v2 delta-output **dramatically improves prediction** (MSE 46.6 vs 162.5 on identity_test)
- v2 delta-output **dramatically worsens identity** (0.570 vs 0.830)

### Root cause (confirmed by per-swap analysis):

| Model | Identity (no swap) | Identity (swap) |
|-------|--------------------|-----------------|
| v1 Raw | 0.972 | **0.663** |
| v2 Delta | **0.991** | **0.076** |

**v2 delta-output is nearly perfect on no-swap episodes (0.991) but catastrophically fails on swap episodes (0.076).**

### Explanation:
When a swap occurs, the two objects exchange positions. The delta-output model predicts future positions relative to the **last observed position**. After a swap, the last observed position already reflects the swapped identity. So the model's prediction naturally follows the swapped trajectory, making the "no-swap" hypothesis have **higher** MSE than the "swap" hypothesis — which is backwards.

In contrast, v1 absolute-output predicts from the training data's absolute positions, which are noisier but less biased by the swap. The noise actually helps — it prevents the model from being overconfident in the wrong direction.

### Implication:
**The trajectory-matching identity method is fundamentally flawed for delta-output models in swap scenarios.** A better identity method is needed — one that doesn't rely on comparing predicted vs actual future positions, but instead uses the observed trajectory structure itself (e.g., velocity continuity, acceleration patterns).

---

## 4. Old SMSS Bug — CONFIRMED

**VelocityOnlyKNN v1 (k=1)**:
- Clean skill = **-0.198** (worse than mean predictor!)
- Old SMSS = **0.428** (non-zero, suggesting structure exists)
- Gated SVTScore = **0.000** (correctly rejects)

This confirms the document's claim: the old SMSS metric gives non-zero scores to models that can't even predict better than the mean.

---

## 5. LastVelocityBaseline

The simplest possible baseline — just extrapolate the last velocity — achieves:
- Clean skill = **0.944** (better than ALL k-NN models!)
- Identity = 0.560 (near random)
- Gated Score = 0.000 (fails identity gate)

This is remarkable: **a trivial linear extrapolation outperforms all retrieval-based methods on clean prediction.** This means the 2D motion world is so simple that even the most naive baseline works well. The only thing it can't do is track identity through swaps.

---

## 6. Scale Sweep Results

| n_train | TransNormDelta (k=5) | RawDelta (k=5) | LastVelocity |
|---------|---------------------|-----------------|--------------|
| 50 | 0.857 | 0.860 | 0.944 |
| 100 | 0.877 | 0.869 | 0.944 |
| 250 | 0.893 | 0.873 | 0.944 |
| 500 | 0.906 | 0.894 | 0.944 |
| 1000 | 0.919 | 0.907 | 0.944 |

**Key insight**: LastVelocityBaseline is constant regardless of training set size (it doesn't use training data). k-NN improves with more data but never surpasses the trivial baseline on clean prediction. This suggests the environment is too simple — the dynamics are purely linear with wall bouncing.

---

## 7. Implications for SVT-v2

### What this means for the Retrieval Gate:
- k-NN CAN predict future positions (clean_skill up to 0.930)
- LastVelocityBaseline achieves 0.944 — even better than k-NN
- **The 2D motion world is too simple for the Retrieval Gate to be meaningful on prediction alone**
- The real test is identity tracking through swaps

### What this means for the Identity Gate:
- v1 k-NN achieves identity = 0.830 through retrieval
- v2 delta-output fails identity on swap episodes (0.076)
- **The trajectory-matching identity method needs redesign for delta-output models**
- A velocity-continuity-based identity method may be more appropriate

### What this means for the environment:
- The 2D linear-motion world is too simple — even LastVelocityBaseline achieves 0.944
- Need to add: non-linear dynamics, acceleration, gravity, friction, or multi-step interactions
- Alternatively: make the prediction horizon longer (t_pred=20 or 50)

### Next steps:
1. **Fix identity method**: Implement velocity-continuity identity (compare last observed velocity with first future velocity for each object)
2. **Harden the environment**: Add non-linear dynamics or longer prediction horizons
3. **More gates**: Underfit Gate, Blind Encoding Gate
4. **Full-scale run**: Use svt_v2.yaml (5000 train) for final results

---

## 8. Technical Decisions Made

1. **Identity prediction method**: Changed from label voting to trajectory matching (comparing MSE under swap vs no-swap). This is the correct method because training labels don't contain swap information.

2. **50% swap rate in identity_test**: Changed from 100% swap to 50% swap, making the identity test balanced and meaningful.

3. **TranslationNormalizedKNN re-anchoring**: Fixed bug where train center was added instead of test center.

4. **Gated SVTScore negative value fix**: Added max(0.0, ...) clamping to prevent negative scores when cf_skill or comp_skill is negative.
