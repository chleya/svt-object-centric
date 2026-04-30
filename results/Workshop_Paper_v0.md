# From Feature Matching to Object Files: Stress-Testing Identity Binding in Agents

---

## Abstract

Current agent models can achieve near-perfect identity assignment under clean conditions, but does this indicate genuine object-file maintenance? We present a structural diagnostic chain demonstrating that clean feature matching, while sufficient for identity assignment when features are reliable, is not equivalent to maintaining an object-file. Through four stages of stress tests (SVT v3.6–v4.2), we show that: (1) temporal-aligned feature matching achieves perfect identity assignment (swap-only = 1.000) under clean conditions; (2) FeatureOnly is 100% misled by tampered features in feature-trajectory conflict, while MinimalObjectFile (swap-only = 0.096) correctly resolves 93.3% of conflicts; (3) confidence-based adaptive weighting fails to resolve this trade-off (calibration error = 0.000); (4) conflict-first adjudication achieves the first meaningful balance (conflict resolution = 0.648, confidence calibration error = 0.637). A supplementary negative result (v4.3) confirms that gate heuristic improvements have diminishing returns, identifying trajectory-state quality as the fundamental bottleneck. A preliminary 3-object continuous-feature sanity check (v5/v5.1) confirms the diagnostic pattern persists beyond the 2-object one-hot setting. We conclude that clean feature matching can read out identity, but object-file requires structural adjudication across feature, trajectory, occlusion, and conflict signals.

---

## 1. Introduction

Object identity binding — the ability to maintain and track which object is which across time — is a prerequisite for object permanence. Recent learned models achieve high prediction accuracy and near-perfect identity assignment under clean conditions. But does this performance indicate genuine structural understanding of object identity, or is it merely exploiting reliable feature correlations?

This paper does not claim to have solved object permanence. Instead, we establish a structural diagnostic chain: a sequence of stress tests that systematically expose the gap between feature matching and object-file maintenance. Our contribution is not a model, but a discriminative method for determining whether a mechanism's identity binding is structurally sound or superficially patterned.

**Our core claim**: Clean feature matching can read out identity, but object-file requires structural adjudication across feature, trajectory, occlusion, and conflict signals. We demonstrate this through four findings:

1. **Feature matching is a positive control, not a solution**: Temporal-aligned feature keys achieve swap-only = 1.000 under clean conditions, but this assumes features are always reliable.

2. **Feature matching fails catastrophically under conflict**: When features are tampered with (feature says "swapped," trajectory says "not swapped"), FeatureOnly is 100% misled, while a minimal rule-based ObjectFile correctly resolves 93.3% of conflicts.

3. **Confidence-based fusion is insufficient**: Simply weighting feature and trajectory signals by confidence fails because confidence itself is uninformative (correct = incorrect = 1.0).

4. **Conflict-first adjudication is a necessary structural component**: Detecting conflict before adjudicating, rather than weighted fusion, achieves the first meaningful balance between normal performance and conflict resolution.

---

## 2. Related Work

**Object-file theory.** Kahneman and Treisman (1992) proposed that object identity is maintained through file-like structures that bind features, locations, and temporal continuity. Our ObjectFile mechanism is a computational implementation of this theory, testing whether maintaining separate information channels with conflict adjudication is a necessary structure for identity binding.

**Object permanence in developmental psychology.** Piaget (1954) and Baillargeon (1987) established that infants demonstrate object permanence through violation-of-expectation experiments. Our stress tests are analogous: they probe whether models maintain structural prerequisites for object permanence by presenting "impossible" events (feature-trajectory conflict).

**Multi-object tracking.** The MOT literature (Bewley et al., 2016; Wojke et al., 2017) uses appearance and motion cues for identity assignment. Our work differs in focusing on the structural properties of identity binding rather than tracking performance.

**Adversarial evaluation.** Our feature-trajectory conflict test is related to adversarial evaluation (Goodfellow et al., 2015), but we target structural properties rather than robustness to input perturbations.

**Relation internalization.** Our interpretation builds on the view that object identity is not a label but a set of relation mechanisms (Bottou, 2014; Battaglia et al., 2018). We instantiate five relations: feature-identity, trajectory-continuity, occlusion-persistence, conflict-resolution, and confidence.

---

## 3. Method

### 3.1 Problem Formulation

Given observed positions $\mathbf{p}^{obs} \in \mathbb{R}^{T_{obs} \times N \times 2}$ and observed features $\mathbf{f}^{obs} \in \mathbb{R}^{T_{obs} \times N \times D}$ for $N$ objects over $T_{obs}$ timesteps, predict the identity assignment $\mathbf{A} \in \{0,...,N-1\}^N$ mapping future objects to observed objects.

**Swap episodes.** At a random timestep, two objects exchange trajectories. Identity binding is only tested when identities actually change, so we report **identity_swap_only**: accuracy restricted to swap episodes.

### 3.2 Stress Tests

| Stress Test | Manipulation | What It Tests |
|-------------|-------------|---------------|
| Feature Ablation | Shuffle or zero features | Feature dependency |
| Feature Noise | Add Gaussian noise (σ=0.1,0.3,0.5) | Robustness |
| Occlusion Without Feature | Zero features during occlusion | Occlusion persistence |
| Feature-Trajectory Conflict | Flip future features in no-swap episodes | Conflict resolution |

The feature-trajectory conflict is the most diagnostic: in no-swap episodes, future features are flipped between objects 0 and 1, creating a scenario where feature similarity says "swapped" but trajectory continuity says "not swapped."

### 3.3 Mechanisms

| Mechanism | Type | Key Structure |
|-----------|------|---------------|
| FeatureOnly | Learned | Cosine similarity of temporal-aligned feature keys |
| TrajectoryOnly | Learned | Nearest predicted position |
| Hybrid | Learned | Weighted combination of feature and trajectory logits |
| MinimalObjectFile | Rule-based | identity_key + trajectory_state + occlusion_state |
| ImprovedObjectFile | Rule-based + learned traj | + confidence calibration + adaptive weighting |
| ConflictFirstObjectFile | Rule-based + learned traj | + conflict detection → margin comparison → adjudicate/abstain |

### 3.4 Evaluation Metrics

- **identity_swap_only**: Accuracy on swap episodes only
- **feature_dependency_score**: normal identity − shuffled identity
- **trajectory_dependency_score**: Zero-feature identity
- **Conflict resolution rate**: Accuracy under feature-trajectory conflict
- **Confidence calibration error**: |avg_conf_correct − avg_conf_incorrect|

---

## 4. Experiments

### 4.1 Setup

2D toy world (64×64 arena), 2 objects, 2D one-hot features. Train on attractor dynamics, test on vortex dynamics (OOD). 1000 training episodes, 200 test episodes. 30 training epochs.

### 4.2 Stage 1: Clean Feature Matching (Positive Control)

Temporal-aligned feature keys (first-timestep pooling) achieve perfect identity assignment:

| Model | Swap-Only | Shuffled | Zero | Feature Dep |
|-------|-----------|----------|------|-------------|
| FeatureOnly | **1.000** | 0.510 | 0.000 | **0.490** |
| Hybrid β=1.0 | 0.990 | 0.538 | 0.433 | 0.452 |

Feature dependency = 0.490 confirms that feature similarity is an effective mechanism for identity assignment. But this is a positive control — it assumes features are always reliable.

### 4.3 Stage 2: Feature Matching Fails Under Conflict

| Mechanism | Swap-Only | Conflict Identity | Trajectory Correct |
|-----------|-----------|------------------|--------------------|
| **FeatureOnly** | **1.000** | **0.000** | 0.000 |
| TrajectoryOnly | 0.135 | 0.943 | **0.943** |
| Hybrid β=1.0 | **1.000** | **0.000** | 0.048 |
| **ObjectFile** | 0.096 | **0.933** | **0.933** |

FeatureOnly achieves perfect clean accuracy but is 100% misled by tampered features. Hybrid suffers the same fate (95% misled). ObjectFile, despite weak normal performance, correctly resolves 93.3% of conflicts.

**Key insight**: Feature reader ≠ Object-file keeper. The value of ObjectFile is not its absolute performance, but its structural bias: it maintains separate feature and trajectory channels and adjudicates when they conflict.

### 4.4 Stage 3: Learned Trajectory Helps But Calibration Fails

| Mechanism | Swap-Only | Conflict Res | Conf Correct | Conf Incorrect | Cal Error |
|-----------|-----------|-------------|-------------|---------------|-----------|
| ObjectFile_v4 | 0.096 | **0.933** | 1.000 | 1.000 | 0.000 |
| ImprovedObjectFile_v4.1 | **0.558** | 0.610 | 1.000 | 1.000 | 0.000 |

Learned trajectory improves swap-only identity (0.096 → 0.558), but conflict resolution drops (0.933 → 0.610). Confidence calibration fails entirely: correct and incorrect predictions both have confidence ≈ 1.0. The conflict gate always chooses feature — this is surrender, not adjudication.

### 4.5 Stage 4: Conflict-First Adjudication

| Strategy | Swap-Only | Conflict Res | Conf Corr | Conf Inc | Cal Error |
|----------|-----------|-------------|-----------|----------|-----------|
| prefer_trajectory | 0.452 | **0.762** | 0.900 | 0.318 | 0.582 |
| prefer_feature | **0.740** | 0.371 | 0.766 | 0.665 | 0.101 |
| **margin_gated** | **0.519** | **0.648** | **0.874** | **0.237** | **0.637** |
| abstain | 0.452 | 0.762 | 0.900 | 0.318 | 0.582 |

Conflict-first gate replaces weighted fusion with: detect conflict → compare margins → adjudicate or abstain. The margin_gated strategy achieves the best balance:
- Conflict resolution = 0.648 (> v4.1's 0.610) ✅
- Swap-only = 0.519 (maintained) ✅
- Confidence calibration: correct = 0.874 vs incorrect = 0.237 ✅
- Uncertain/abstain appears on high-conflict samples ✅

This is the first mechanism in the series to achieve meaningful confidence calibration.

---

## 5. Analysis

### 5.1 Core Findings

**Finding 1: Prediction ≠ Identity Binding.** TrajectoryOnly achieves low swap-only identity (0.135) despite being trained on trajectory prediction. Identity binding requires not just predicting future positions, but determining which object occupies which position.

**Finding 2: Clean Feature Matching ≠ Object-File.** FeatureOnly achieves swap-only = 1.000 under clean conditions but 0.000 under conflict. Object-file's core challenge is not "can features match when reliable?" but "can identity be maintained when features are unreliable?"

**Finding 3: Hybrid Weighted Fusion Is Not Enough.** Hybrid β=1.0 is 95% misled in conflict. Learned feature logits dominate trajectory logits in magnitude, making weighted combination ineffective.

**Finding 4: ObjectFile's Key Metric Is Conflict Resolution, Not Raw Accuracy.** ObjectFile_v4's swap-only = 0.096 is unimpressive, but its conflict resolution = 0.933 demonstrates correct structural bias. The diagnostic value is in the structural property, not the absolute number.

**Finding 5: Conflict-First Gate Enables Structural Internalization.** v4.2's conflict-first gate is the first mechanism to achieve both meaningful confidence calibration and conflict resolution above v4.1. It represents a shift from "weighted fusion" to "detect conflict, then adjudicate."

### 5.2 Relation-Internalization Interpretation

ObjectFile can be interpreted as the minimal case of relation internalization — maintaining and adjudicating a set of relations rather than a single representation:

| Relation | ObjectFile Component | Failure Mode |
|----------|---------------------|-------------|
| Feature-identity | identity_key | Feature hijacking |
| Trajectory-continuity | trajectory_state | OOD prediction failure |
| Occlusion-persistence | occlusion_state + decay | Identity drift |
| Conflict-resolution | conflict-first gate | Weighted fusion surrender |
| Confidence | calibration mechanism | Uniform high confidence |

**Core thesis**: Structure is not a representation, but a set of updatable, adjudicable, transferable relation mechanisms.

### 5.3 The Normal-Performance vs. Conflict-Resolution Trade-off

All mechanisms exhibit a systematic trade-off:

```
prefer_trajectory → conflict high, swap low
prefer_feature    → swap high, conflict low
margin_gated      → balanced compromise
```

This is not a parameter problem but a structural problem: when the trajectory signal is weak (TrajectoryOnly swap-only = 0.135), any gate that relies on trajectory will sacrifice normal performance. The remaining bottleneck is trajectory-state quality, not gate design.

---

## 6. Limitations

1. **2D toy world** with limited dynamics (attractor, vortex)
2. **Only 2 objects** in main experiments
3. **Simplified one-hot features** (2-dimensional)
4. **Rule-based conflict gate** with manually set thresholds
5. **Weak trajectory predictor** (TrajectoryOnly swap-only = 0.135 on OOD)
6. **Not a demonstration of full object permanence** — only structural prerequisites

---

## 7. Conclusion

We have established a structural diagnostic chain showing that clean feature matching can read out identity, but object-file requires structural adjudication across feature, trajectory, occlusion, and conflict signals. The key contributions are:

1. A stress-testing framework that systematically evaluates identity binding under feature noise, occlusion, and feature-trajectory conflict
2. Empirical evidence that FeatureOnly (100% clean accuracy) is 100% misled by tampered features, while MinimalObjectFile (9.6% clean accuracy) correctly resolves 93.3% of conflicts
3. The ConflictFirst gate, the first mechanism to achieve meaningful confidence calibration alongside conflict resolution
4. A negative result showing that gate heuristic improvements have diminishing returns, identifying trajectory-state quality as the fundamental bottleneck

The contribution is a diagnostic chain, not a solved problem. Future work should focus on stronger trajectory predictors, principled uncertainty models, and relation-specific object files.

---

## References

Baillargeon, R. (1987). Object permanence in 3½- and 4½-month-old infants. *Developmental Psychology*, 23(5), 655.

Battaglia, P. W., et al. (2018). Relational inductive biases, deep learning, and graph networks. *arXiv preprint arXiv:1806.01261*.

Bewley, A., et al. (2016). Simple online and realtime tracking. *IEEE ICIP*.

Bottou, L. (2014). From machine learning to machine reasoning. *Machine Learning*, 94(2), 133-149.

Goodfellow, I. J., et al. (2015). Explaining and harnessing adversarial examples. *ICLR*.

Kahneman, D., & Treisman, A. (1992). The reviewing of object files: Object-specific integration of information. *Cognitive Psychology*, 24(2), 175-219.

Piaget, J. (1954). *The construction of reality in the child*. Basic Books.

Wojke, N., et al. (2017). Simple online and realtime tracking with a deep association metric. *IEEE ICIP*.

---

## Appendix A: Gate Heuristic Diminishing Returns (v4.3)

We additionally tested whether observed-period approach detection and simple trajectory augmentation could improve the normal-performance vs. conflict-resolution trade-off. Approach detection improved swap-only identity (0.529 → 0.625) but reduced conflict resolution (0.714 → 0.619). Trajectory augmentation did not improve OOD robustness (TrajectoryOnly swap-only: 0.173 standard vs. 0.125 augmented). This confirms that the remaining bottleneck lies in trajectory-state quality rather than gate heuristics.

## Appendix B: Preliminary 3-Object Continuous-Feature Sanity Check (v5/v5.1)

As a preliminary scaling sanity check, we repeated the diagnostic pattern under a 3-object setting with continuous 16-dimensional features.

### B.1 Results

| Mechanism | Swap-Only | Conflict Identity |
|-----------|-----------|------------------|
| FeatureOnly (3obj, cont16) | **1.000** | **0.000** |
| TrajectoryOnly (3obj) | 0.067 | 0.815 |
| ConflictFirst_margin (3obj, cont16) | 0.650 | 0.343 |

FeatureOnly achieves perfect clean accuracy but fails completely under conflict. ConflictFirstObjectFile achieves higher swap-only identity (0.650 vs. 0.519 at 2 objects) but lower conflict resolution (0.343 vs. 0.648 at 2 objects), reflecting the increased assignment complexity with 3 objects.

### B.2 Sanity Audit

We audited four aspects of the 3-object continuous-feature setting:

1. **N=3 permutation metric**: Perfect prediction = 1.0, random ≈ 0.03, fixed identity drops for non-identity permutations. ✅
2. **Continuous feature oracle**: Normal = 1.000, shuffled = 0.170, wrong = 0.000. ✅
3. **Conflict construction**: Conflict rate = 0.889, feature matches true = 0.000, trajectory correct when disagreeing = 0.615. ✅
4. **FeatureOnly conflict = 0 is not an artifact**: Restoring correct features recovers accuracy to 1.000. ✅

All audits passed, confirming that the diagnostic pattern persists under 3 objects + continuous features and that FeatureOnly's failure under conflict is genuine.

### B.3 Caveat

We do not claim that SVT has generalized to complex environments. The 3-object continuous-feature result is preliminary supplementary evidence that the diagnostic pattern is not limited to two one-hot objects. Broader scaling (4+ objects, higher-dimensional features, more dynamics types) remains future work.

Note: The existing `identity_breakdown` metric uses `true_identity[:, 0] != 0` for swap detection, which works for single-swap permutations but is not fully general for N-object permutations. For the supplementary analysis, we additionally audited permutation-level assignment accuracy directly.
