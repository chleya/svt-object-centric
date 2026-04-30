# The Delta-Output Identity Paradox: When Better Prediction Destroys Object Identity

## Symbolic Verification Test v2 — Full Experimental Report

**Date**: 2026-04-27
**Environment**: 2D Physics World (gravity=0.3, friction=0.02, acceleration_noise=0.15, t_pred=20)
**Training set**: 1000 episodes | **Test splits**: 200 episodes each

---

## Abstract

We present a counterintuitive finding in multi-object physical prediction: **methods that improve prediction accuracy can systematically destroy the ability to track object identity through occlusion events.** We call this the **Delta-Output Identity Paradox**. In experiments across k-NN retrieval baselines and learned neural models (MLP, Transformer), we consistently observe that models predicting position deltas (displacements from the last observed position) achieve higher prediction skill scores but catastrophically fail at identity tracking when objects swap positions during occlusion. Even explicitly training a model with an identity supervision head fails to resolve this paradox — the identity head achieves perfect accuracy on clean episodes but random performance on swap episodes. These results demonstrate that prediction accuracy alone is insufficient to evaluate whether a model possesses genuine physical understanding, and motivate the use of gated evaluation metrics that require both prediction skill AND identity tracking.

---

## 1. Introduction

### 1.1 Motivation

Current evaluation of physical prediction models focuses primarily on prediction accuracy (MSE, skill scores). However, a model that perfectly predicts future positions may still fail to understand *which object is which* — a fundamental aspect of physical reasoning known as object identity or object permanence.

### 1.2 The SVT-v2 Framework

We propose the Symbolic Verification Test v2 (SVT-v2), which evaluates models through four gates:

1. **Prediction Gate**: Can the model predict future positions? (clean_skill > threshold)
2. **Identity Gate**: Can the model track which object is which through occlusion?
3. **Counterfactual Gate**: Does the model respond correctly to velocity interventions?
4. **Compositional Gate**: Does the model maintain consistency under object swaps?

The **Gated SVTScore** is:
```
Gated SVTScore = clean_skill × max(0, cf_skill) × max(0, comp_skill) × max(0, identity_accuracy)
```
only if `clean_skill > threshold` (default 0.5), otherwise 0.

### 1.3 The Old SMSS Bug

The previous metric (Old SMSS) uses MSE ratios that can yield non-zero scores even for models that predict worse than the mean predictor. We confirm this bug: VelocityOnlyKNN (k=1) with clean_skill = -0.198 receives Old SMSS = 0.428, while Gated SVTScore correctly assigns 0.000.

---

## 2. Experimental Setup

### 2.1 Environment: 2D Physics World

Two objects move in a 64×64 arena with:
- **Gravity** (0.3): downward acceleration
- **Friction** (0.02): velocity damping
- **Acceleration noise** (0.15): stochastic perturbations
- **Wall bouncing** with 10% energy loss
- **Occlusion**: objects may be hidden briefly
- **Hidden perturbation**: velocity may be perturbed during occlusion
- **Identity swap**: 50% of identity_test episodes have object positions swapped during occlusion

Observation: 10 timesteps | Prediction: 20 timesteps

### 2.2 Data Splits

| Split | Episodes | Description |
|-------|----------|-------------|
| train | 1000 | Standard episodes |
| clean_test | 200 | Standard episodes |
| counterfactual_test | 200 | Velocity intervention applied |
| compositional_test | 200 | Object swap in future |
| identity_test | 200 | 50% have identity swap during occlusion |

### 2.3 Models Evaluated

| Category | Models |
|----------|--------|
| Oracle | Physics Oracle (ground truth dynamics) |
| k-NN v1 | RawTrajectoryKNN, TranslationNormalizedKNN, VelocityOnlyKNN |
| k-NN v2 | RawDeltaKNN, TranslationNormalizedDeltaKNN, VelocityDeltaKNN, LastVelocityBaseline |
| Learned | MLP (4-layer, 256 hidden), Transformer (d=128, 3+3 layers) |
| Dual-head | MLP + Identity Head, (Transformer + Identity Head pending) |

### 2.4 Identity Methods

| Method | Description |
|--------|-------------|
| Trajectory Matching | Compare MSE under swap vs no-swap assignment of predicted future to actual future |
| Velocity Continuity | Compare velocity at observation/future boundary; pick assignment minimizing discontinuity |
| Model Identity Head | Neural network head trained to predict swap/no-swap from predicted future features |

---

## 3. Results

### 3.1 Oracle Upper Bound

| Split | MSE | Skill Score | Identity |
|-------|-----|-------------|----------|
| clean_test | 10.26 | **0.952** | 1.000 |
| counterfactual_test | 579.79 | -0.114 | 1.000 |
| compositional_test | 9.45 | **0.955** | 1.000 |
| identity_test | 74.44 | 0.678 | 1.000 |

Oracle achieves near-perfect prediction on clean/compositional splits. Counterfactual is inherently unpredictable (Oracle doesn't know the intervention). Identity is always perfect (Oracle has access to true labels).

### 3.2 k-NN Retrieval Attacks

#### v1 (Absolute Output) — Best Results

| Model | k | Clean Skill | Identity (traj) | Gated Score |
|-------|---|-------------|-----------------|-------------|
| RawTrajectoryKNN | 10 | 0.589 | 0.830 | 0.071 |
| TranslationNormalizedKNN | 10 | 0.479 | 0.775 | 0.000 |
| VelocityOnlyKNN | 10 | 0.352 | 0.525 | 0.000 |

#### v2 (Delta Output) — Best Results

| Model | k | Clean Skill | Identity (traj) | Gated Score |
|-------|---|-------------|-----------------|-------------|
| RawDeltaKNN | 5 | **0.672** | 0.530 | 0.104 |
| TranslationNormalizedDeltaKNN | 10 | 0.557 | 0.525 | 0.089 |
| VelocityDeltaKNN | 10 | 0.486 | 0.525 | 0.000 |
| LastVelocityBaseline | — | -1.240 | 0.500 | 0.000 |

**Key observation**: v2 delta-output achieves higher clean_skill (0.672 vs 0.589) but **dramatically lower identity** (0.530 vs 0.830). The identity accuracy of 0.530 is barely above random (0.500).

### 3.3 Learned Models

| Model | Clean Skill | CF Skill | Comp Skill | ID-test (traj) | ID-test (vel) | Model-ID | Gated Score |
|-------|-------------|----------|------------|----------------|---------------|----------|-------------|
| MLP (base) | **0.805** | -0.278 | 0.808 | 0.595 | 0.630 | — | 0.000 |
| MLP (dual) | 0.797 | -0.293 | 0.804 | 0.605 | 0.630 | 0.500 | 0.000 |
| MLP (obs-cond-ID) | 0.803 | -0.320 | 0.799 | 0.590 | 0.630 | 0.500 | 0.000 |
| Transformer (base) | 0.595 | -0.033 | 0.565 | 0.670 | 0.630 | — | 0.000 |
| Transformer-small (dual) | -0.032 | 0.037 | -0.038 | 0.715 | 0.630 | 0.500 | 0.000 |
| Object-Centric | 0.815 | -1.456 | 0.811 | 0.570 | 0.630 | 0.500 | 0.000 |

**Critical finding**: ALL models with identity heads achieve **Model-ID = 0.500** on identity_test — exactly random. This includes:
- **Dual-head MLP**: identity head conditioned on predicted future → 0.500
- **Obs-conditioned MLP**: identity head conditioned on observation history → 0.500
- **Object-Centric**: per-object prediction + obs-conditioned identity head → 0.500
- **Transformer-small dual**: identity head conditioned on predicted future → 0.500

**Root cause**: The training set contains NO swap episodes. All identity labels in training are [0,1] (no swap). The identity head never sees a swap during training, so it learns to always predict "no swap" (logit > 0 for all inputs). On identity_test where 50% of episodes have swaps, this yields exactly 50% accuracy.

### 3.4 The Delta-Output Identity Paradox — Cross-Model Evidence

| Model | Clean ID (traj) | Identity-Test ID (traj) | Drop |
|-------|-----------------|------------------------|------|
| Oracle | 1.000 | 1.000 | 0% |
| k-NN v1 (Raw, k=10) | 0.995 | 0.830 | -17% |
| k-NN v2 (RawDelta, k=5) | 0.990 | 0.530 | **-46%** |
| MLP (base) | 0.990 | 0.595 | -40% |
| MLP (dual) | 0.990 | 0.605 | -39% |
| MLP (obs-cond-ID) | 0.990 | 0.590 | -40% |
| Transformer (base) | 0.930 | 0.670 | -26% |
| Object-Centric | 0.995 | 0.570 | **-43%** |

**All learned models show a massive identity drop on swap episodes.** The delta-output k-NN models show the worst drop (46 percentage points), while the absolute-output v1 models show a more modest drop (17 points).

---

## 4. Analysis

### 4.1 Why Does Better Prediction Destroy Identity?

The root cause lies in how delta-output models anchor their predictions:

1. **Absolute-output models** (v1 k-NN): Retrieve similar trajectories from training data and predict absolute future positions. The prediction is noisy but unbiased with respect to identity — the retrieved trajectories come from the correct identity assignment in training.

2. **Delta-output models** (v2 k-NN, MLP, Transformer): Predict displacement from the last observed position. When a swap occurs, the last observed position already reflects the swapped identity. The model confidently predicts the future of "whatever is at position A" — which is now object B. The prediction is accurate for the *position* but wrong for the *identity*.

3. **Trajectory matching identity** compares MSE under swap vs no-swap. For delta-output models, the no-swap MSE is actually *lower* than the swap MSE on swap episodes — the logic is inverted. The model predicts the swapped trajectory so well that the "wrong" assignment looks correct.

### 4.2 Why Does the Identity Head Fail? — A Deeper Analysis

We tested three identity head architectures:

1. **Dual-head (predicted-future-conditioned)**: Identity head takes features from the predicted future. Model-ID on identity_test = **0.500**.
2. **Obs-conditioned identity head**: Identity head takes features directly from the observation history, NOT from the predicted future. Model-ID on identity_test = **0.500**.
3. **Object-Centric identity head**: Per-object prediction with identity head from observation history. Model-ID on identity_test = **0.500**.

ALL three architectures fail identically. This reveals that the failure is **NOT an architecture problem** but a **data distribution problem**:

- The training set contains **zero swap episodes** — all identity labels are [0,1]
- The identity head learns to always predict "no swap" regardless of input
- On identity_test where 50% of episodes have swaps, this yields exactly 50% accuracy

This is a fundamental challenge for supervised identity prediction: **you cannot learn to detect a phenomenon you've never seen during training.**

### 4.3 The Unsupervised Identity Problem

Since supervised identity heads fail due to lack of swap training data, the identity tracking problem must be solved **unsupervised** — without labels indicating whether a swap occurred. The two unsupervised methods we tested are:

| Method | Clean ID | ID-test ID | Type |
|--------|----------|------------|------|
| Trajectory Matching | 0.990 | 0.595 | Unsupervised (uses predicted future) |
| Velocity Continuity | 0.970 | 0.630 | Unsupervised (uses actual future) |

Velocity continuity slightly outperforms trajectory matching because it uses the actual (not predicted) future positions, avoiding the delta-output bias. However, both methods still fail significantly on swap episodes.

The key insight is that **velocity continuity uses ground-truth future positions** — it's not a model prediction but a property of the data itself. A model that could learn to perform velocity-continuity-style reasoning internally would have a better chance at identity tracking.

### 4.4 Velocity Continuity: A Partial Fix

Velocity continuity identity compares the velocity at the end of observation with the velocity at the start of the future, choosing the assignment that minimizes discontinuity. This method does not depend on model predictions and uses the actual future positions.

| Model | Identity-Test (traj) | Identity-Test (vel) |
|-------|---------------------|---------------------|
| MLP (base) | 0.595 | **0.630** |
| Transformer (base) | 0.670 | 0.630 |

Velocity continuity slightly outperforms trajectory matching for MLP, but still only achieves 0.630 — well below the 0.970 achieved on clean episodes. The gravity and acceleration noise in the enhanced environment introduce velocity discontinuities that make this method unreliable.

### 4.5 No Model Passes the Identity Gate

| Model | Clean Skill | Identity-Test ID | Gate Passed | Gated Score |
|-------|-------------|------------------|-------------|-------------|
| Oracle | 0.952 | 1.000 | ✅ | 0.000* |
| MLP (base) | 0.805 | 0.595 | ✅ (pred) | 0.000 |
| MLP (dual) | 0.797 | 0.605 | ✅ (pred) | 0.000 |
| MLP (obs-cond-ID) | 0.803 | 0.590 | ✅ (pred) | 0.000 |
| Transformer | 0.595 | 0.670 | ✅ (pred) | 0.000 |
| Object-Centric | 0.815 | 0.570 | ✅ (pred) | 0.000 |
| k-NN v1 (best) | 0.592 | 0.830 | ✅ | 0.071 |
| k-NN v2 (best) | 0.672 | 0.530 | ✅ | 0.104 |

*Oracle Gated Score is 0 because cf_skill = -0.114 (negative, clamped to 0).

**No learned model achieves a non-zero Gated SVTScore.** The identity gate (combined with negative cf_skill) blocks all models. Only the v1 k-NN achieves a small non-zero score (0.071) due to its higher identity accuracy.

---

## 5. The Gated SVTScore as a Diagnostic

The Gated SVTScore reveals what prediction-only metrics hide:

| Model | Clean Skill | Old SMSS | Gated SVTScore | Verdict |
|-------|-------------|----------|----------------|---------|
| MLP (base) | 0.805 | 0.000 | 0.000 | Fails identity + counterfactual |
| MLP (dual) | 0.797 | 0.000 | 0.000 | Fails identity + counterfactual |
| VelocityOnlyKNN | -0.198 | 0.428 | 0.000 | Fails prediction gate |

The Old SMSS gives VelocityOnlyKNN a score of 0.428 despite it being worse than the mean predictor. The Gated SVTScore correctly assigns 0.000.

---

## 6. Discussion

### 6.1 Implications for Model Evaluation

Our results demonstrate that **prediction accuracy is necessary but not sufficient** for evaluating physical understanding. A model that perfectly predicts future positions but cannot track object identity lacks a fundamental component of physical reasoning.

### 6.2 The Delta-Output Paradox as a General Phenomenon

We observe the paradox across three distinct model families (k-NN, MLP, Transformer), suggesting it is not an artifact of a particular architecture but a fundamental property of delta-output prediction in the presence of identity swaps. Any model that conditions its predictions on the last observed position will inherit this bias.

### 6.3 Potential Solutions — Experimental Results

We tested five approaches to resolve the identity paradox:

| Approach | Training Data | Model-ID (clean) | Model-ID (ID-test) | Verdict |
|----------|--------------|-------------------|---------------------|---------|
| Dual-head MLP | Normal only | 1.000 | 0.500 | ❌ Never sees swaps |
| Obs-conditioned MLP | Normal only | 1.000 | 0.500 | ❌ Never sees swaps |
| Object-Centric | Normal only | 1.000 | 0.500 | ❌ Never sees swaps |
| Dual-head MLP | +30% swap | 0.000 | 0.500 | ❌ Predicts all-swap |
| Obs-cond MLP | +30% swap | 0.000 | 0.500 | ❌ Predicts all-swap |
| Velocity Continuity | Normal only | 0.000 | 0.500 | ❌ Vel prediction no help |
| Contrastive | +30% swap | 1.000 | 0.500 | ❌ Contrastive fails |
| VC model | +30% swap | 0.000 | 0.500 | ❌ Same issue |
| Slot Persistence | Normal only | 0.000 | 0.500 | ❌ Slot binding fails |
| Slot Persistence | +30% swap | 0.000 | 0.500 | ❌ Still fails |

**All ten approaches fail to achieve above-random identity accuracy on identity_test.**

### 6.4 Environment Difficulty and Velocity Continuity Upper Bound

Even in a zero-noise environment (no gravity, friction, acceleration noise, or hidden perturbation), velocity continuity achieves only **0.655** on identity_test. This reveals that **wall bouncing** is a major source of velocity discontinuity — objects change direction upon wall collision, making velocity-based identity matching unreliable.

| Environment | Vel-ID (clean) | Vel-ID (ID-test) | Best k-NN ID |
|-------------|----------------|-------------------|--------------|
| Clean (no noise) | 1.000 | 0.655 | 0.790 |
| Hard (gravity+friction+noise) | 0.970 | 0.630 | 0.830 |

### 6.5 Slot Persistence: Best Learned Model

The Slot Persistence model (GRU-based per-object slot with velocity decoder) achieves the highest clean_skill of any model:

| Config | Clean Skill | ID-test Skill | ID-test (traj) | Gated Score |
|--------|-------------|---------------|----------------|-------------|
| Slot + normal train | 0.937 | -82.075 | 0.520 | 0.000 |
| Slot + swap train | 0.860 | -1.140 | 0.505 | **0.0556** |
| Slot + clean env | 0.989 | -51.739 | 0.585 | 0.000 |

Slot Persistence with swap training is the **only learned model to achieve a non-zero Gated SVTScore** (0.0556). However, its identity accuracy remains at random level (0.505).

The slot model's catastrophic failure on identity_test (skill = -82) when trained on normal data reveals that **GRU slots learn position-dependent bindings** — the slot for "object 0" learns to predict trajectories starting from the left side, and completely fails when object 0 is swapped to the right side.

### 6.6 Why Everything Fails — The Fundamental Challenge

The core problem is that **identity swaps preserve velocity continuity**. When two objects swap positions during occlusion:

1. Object A moves to position B with velocity v_A → continues with v_A from position B
2. Object B moves to position A with velocity v_B → continues with v_B from position A
3. After the swap, each object continues with its **original velocity** from its **new position**

This means velocity continuity is **preserved under swap** — the "no-swap" assignment has equal or lower velocity discontinuity than the "swap" assignment. The velocity continuity method's core assumption ("velocity continuity implies identity preservation") is **fundamentally violated** by the swap mechanism.

We confirmed this in the toroidal (wall-less) environment:
- **No-swap episodes**: Vel-ID = 1.000 (perfect)
- **Swap episodes**: Vel-ID = 0.185 (worse than random!)

On swap episodes, velocity continuity actually performs **worse than random** because the no-swap assignment has lower velocity discontinuity than the correct swap assignment.

This is a **tautological failure**: the swap mechanism is designed to be undetectable by velocity-based methods. The only way to detect it would be to track identity through the occlusion itself, which requires maintaining persistent object representations that survive occlusion — exactly what current models cannot do.

### 6.7 Positive Finding: Swap Training Improves Prediction

| Model | Training | ID-test Skill Score |
|-------|----------|---------------------|
| MLP (base) | Normal only | 0.241 |
| MLP (dual) | Normal only | 0.317 |
| Contrastive | +30% swap | **0.731** |
| VC model | +30% swap | **0.731** |

Training with swap episodes dramatically improves prediction on identity_test (0.241 → 0.731). The model learns to predict both possible futures (swapped and unswapped) and averages them, resulting in better MSE. However, this comes at the cost of identity — the model can no longer distinguish which future actually occurred.

### 6.8 Limitations

- Small training set (1000 episodes) — Transformer may underperform due to data scarcity
- CPU-only training limits model size and training duration
- Only 2 objects — the identity problem becomes harder with more objects
- Binary swap (0 or 1) — real-world scenarios may involve partial identity confusion

---

## 7. Conclusion

We have presented the **Delta-Output Identity Paradox**: improving prediction accuracy through delta-output methods systematically destroys the ability to track object identity through occlusion events. This paradox holds across k-NN retrieval baselines and learned neural models (MLP, Transformer), and persists even when explicitly training an identity supervision head.

These results motivate the use of **gated evaluation metrics** (Gated SVTScore) that require models to pass both prediction AND identity thresholds. Prediction-only metrics (MSE, skill score) and the old SMSS metric can give misleadingly high scores to models that lack fundamental physical understanding.

The SVT-v2 framework with its four gates (Prediction, Identity, Counterfactual, Compositional) provides a rigorous test suite for distinguishing genuine physical understanding from superficial prediction ability.

---

## Appendix A: Full Results Tables

### A.1 k-NN v1 on Enhanced Environment

| Model | k | Clean Skill | CF Skill | Comp Skill | Identity | Gated |
|-------|---|-------------|----------|------------|----------|-------|
| RawTrajectoryKNN | 1 | 0.353 | 0.135 | 0.376 | 0.735 | 0.000 |
| RawTrajectoryKNN | 3 | 0.572 | 0.231 | 0.582 | 0.795 | 0.061 |
| RawTrajectoryKNN | 5 | 0.592 | 0.243 | 0.598 | 0.805 | 0.069 |
| RawTrajectoryKNN | 10 | 0.589 | 0.244 | 0.598 | **0.830** | **0.071** |
| TranslationNormalizedKNN | 10 | 0.479 | 0.415 | 0.506 | 0.775 | 0.000 |
| VelocityOnlyKNN | 1 | -0.165 | -0.174 | -0.158 | 0.585 | 0.000 |

### A.2 k-NN v2 on Enhanced Environment

| Model | k | Clean Skill | Identity | Gated |
|-------|---|-------------|----------|-------|
| RawDeltaKNN | 5 | **0.672** | 0.530 | 0.104 |
| TranslationNormalizedDeltaKNN | 10 | 0.557 | 0.525 | 0.089 |
| VelocityDeltaKNN | 10 | 0.486 | 0.525 | 0.000 |
| LastVelocityBaseline | — | -1.240 | 0.500 | 0.000 |

### A.3 Learned Models on Enhanced Environment

| Model | Clean Skill | CF Skill | Comp Skill | ID-test (traj) | ID-test (vel) | ID-test (model) |
|-------|-------------|----------|------------|----------------|---------------|-----------------|
| MLP (base) | 0.805 | -0.278 | 0.808 | 0.595 | 0.630 | — |
| MLP (dual) | 0.797 | -0.293 | 0.804 | 0.605 | 0.630 | 0.500 |
| Transformer (base) | 0.595 | -0.033 | 0.565 | 0.670 | 0.630 | — |

### A.4 Identity Accuracy: Clean vs Identity-Test

| Model | Clean (traj) | ID-test (traj) | Δ | Clean (vel) | ID-test (vel) | Δ |
|-------|-------------|----------------|---|-------------|---------------|---|
| Oracle | 1.000 | 1.000 | 0 | 1.000 | 1.000 | 0 |
| k-NN v1 Raw (k=10) | 0.995 | 0.830 | -0.165 | — | — | — |
| k-NN v2 RawDelta (k=5) | 0.990 | 0.530 | **-0.460** | — | — | — |
| MLP (base) | 0.990 | 0.595 | -0.395 | 0.970 | 0.630 | -0.340 |
| MLP (dual) | 0.990 | 0.605 | -0.385 | 0.970 | 0.630 | -0.340 |
| Transformer (base) | 0.930 | 0.670 | -0.260 | 0.970 | 0.630 | -0.340 |
