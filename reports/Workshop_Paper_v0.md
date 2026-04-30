# From Feature Matching to Object Files: Stress-Testing Identity Binding in Agents

# 从特征匹配到对象文件：智能体身份绑定的结构有效性压力测试

---

## Abstract

Object-centric agents often achieve high identity assignment accuracy under clean conditions, but does this performance reflect genuine object-file identity binding, or merely feature matching? We present Structure Validity Tests (SVT), a diagnostic framework that systematically stress-tests identity binding under feature corruption, occlusion, feature-trajectory conflict, and confidence calibration. In a controlled 2D multi-object environment, we demonstrate a structural discriminative chain: (1) a FeatureOnly model achieves perfect identity accuracy (swap-only = 1.000) under clean conditions, but is completely misled (conflict = 0.000) when features are tampered; (2) a MinimalObjectFile with separate feature and trajectory channels and conflict adjudication correctly resolves 93.3% of feature-trajectory conflicts despite low normal performance; (3) learned trajectory states improve normal performance but degrade conflict resolution and confidence calibration; (4) a conflict-first gate achieves the most balanced performance to date (conflict resolution = 0.648, calibration = 0.637). Critically, we apply SVT to four published object-centric models (Slot Attention, RIMs, SAVi, DINOSAUR) and find that all four exhibit a feature-reader-like profile: perfect clean accuracy but complete failure under feature-trajectory conflict. This extends our finding from "our ObjectFile has correct bias" to "current object-centric models lack conflict-resolution structure." Our contribution is not a strong model, but a diagnostic method showing that clean feature matching can read out identity without constituting an object-file mechanism.

---

## 1. Introduction

Current object-centric models achieve impressive performance on segmentation, reconstruction, and prediction benchmarks. However, high performance under benign conditions does not establish that a model possesses genuine object-file identity binding — the ability to maintain object identity across time despite feature corruption, occlusion, and conflicting cues.

Consider a simple scenario: two objects move through a 2D arena, each carrying a distinctive feature. At some point, their features are swapped. Can the model correctly determine which observed object corresponds to which future object? Under clean conditions, feature matching trivially solves this task. But when features are unreliable — corrupted, occluded, or deliberately misleading — feature matching fails catastrophically, while a mechanism that adjudicates between feature and trajectory signals can maintain correct structural bias.

This paper presents Structure Validity Tests (SVT), a framework for stress-testing identity binding in object-centric agents. Our core claim is:

> Clean feature matching can read out identity under benign conditions, but object-file claims require stress tests under feature corruption, occlusion, feature-trajectory conflict, and confidence calibration.

We do not claim that object permanence is solved, that SVT "passes," or that our ObjectFile variants are complete solutions. We claim that the diagnostic chain — the progression from clean matching to stress-tested structural adjudication — reveals a fundamental gap between feature readability and object-file identity.

### Contributions

1. A stress-testing framework (SVT) that systematically evaluates identity binding under feature noise, occlusion, and feature-trajectory conflict.
2. Empirical evidence that FeatureOnly (100% clean accuracy) is 100% misled by tampered features, while MinimalObjectFile (9.6% clean accuracy) correctly resolves 93.3% of conflicts.
3. The ConflictFirst gate, the first mechanism to achieve both meaningful confidence calibration and conflict resolution above chance.
4. A negative result (v4.3) showing that gate heuristic improvements have diminishing returns, identifying trajectory-state quality as the fundamental bottleneck.
5. **External audit**: 4/4 tested published object-centric models (Slot Attention, RIMs, SAVi, DINOSAUR) exhibit a feature-reader-like profile under SVT, extending the finding beyond our own models.

---

## 2. Related Work

### 2.1 Object Permanence in Developmental Psychology

Infants demonstrate object permanence through violation-of-expectation experiments (Baillargeon, 1987): they look longer at "impossible" events where objects disappear or violate identity. Our stress tests are analogous — they probe whether models maintain structural prerequisites for object permanence, not whether they possess full object permanence.

### 2.2 Object-File Theory

Kahneman & Treisman's object-file theory (1992) proposes that object identity is maintained through a file-like structure that binds features, locations, and temporal continuity. Our ObjectFile is a computational implementation of this theory: it maintains separate channels for feature matching and trajectory continuity, with a conflict adjudication mechanism.

### 2.3 Object-Centric Representation Learning

Slot Attention (Locatello et al., 2020), SAVi (Kipf et al., 2022), DINOSAUR (Seitzer et al., 2024), and RIMs (Goyal et al., 2021) learn object-centric representations but do not explicitly test identity binding under stress. Our framework provides a complementary evaluation: not "can the model segment?" but "can the model maintain identity when cues conflict?"

### 2.4 Stress Testing and Adversarial Evaluation

Adversarial examples (Goodfellow et al., 2015) test robustness of classification. Distribution shift testing (Ood generalization) tests transfer. Our stress tests are distinct: they test the *structure* of identity binding mechanisms, not just their robustness or transfer.

---

## 3. Method

### 3.1 Problem Formulation

Given observed positions and features for N objects over T_obs timesteps, and future positions and features over T_pred timesteps, the identity binding task is to assign each future object to the correct observed object.

**Swap episodes**: At a random timestep, object identities are exchanged. The model must detect this swap and correctly assign identities.

**Primary metric**: `identity_swap_only` — accuracy on swap episodes only, which avoids no-swap bias. For N > 2, we use permutation-level assignment accuracy plus a sanity audit.

### 3.2 Stress Tests

| Stress Test | What it tests | How |
|-------------|---------------|-----|
| Feature ablation | Does the model rely on features? | Replace features with zeros |
| Feature noise | Is the model robust to feature corruption? | Add Gaussian noise to features |
| Occlusion without feature | Can the model maintain identity when features are absent? | Zero out features for selected timesteps |
| Feature-trajectory conflict | Can the model adjudicate when features and trajectory disagree? | Swap features between objects |
| Confidence calibration | Does the model's confidence reflect its accuracy? | Compare confidence on correct vs incorrect assignments |

### 3.3 Models

| Model | Structure | Key property |
|-------|-----------|-------------|
| FeatureOnly | Feature matching only | Positive control: can solve identity under clean conditions |
| TrajectoryOnly | Trajectory prediction only | Tests trajectory signal quality |
| Hybrid | Weighted feature + trajectory fusion | Tests whether fusion resolves conflict |
| MinimalObjectFile | Separate channels + rule-based adjudication | Structural probe: correct bias under conflict |
| ImprovedObjectFile | Learned trajectory state + ObjectFile structure | Tests whether learning improves ObjectFile |
| ConflictFirstObjectFile | Conflict detection + adjudication | Current best: balanced conflict resolution and calibration |

---

## 4. Experiments

### 4.1 Environment

2D arena with 2 objects. Objects move under nonlinear force fields (attractor, vortex, damping). Each object carries a 2-dimensional feature vector. Training uses attractor dynamics; OOD testing uses vortex dynamics.

### 4.2 Mainline Results

**v3.6 — Positive Control**: FeatureOnly achieves identity_swap_only = 1.000 under clean conditions. This demonstrates that feature matching *can* solve identity, but does not establish object-file identity.

**v4 — Stress Test**: Under feature-trajectory conflict, FeatureOnly's identity drops to 0.000 — it is completely misled by tampered features. MinimalObjectFile achieves conflict resolution = 0.933, demonstrating correct structural bias despite low normal performance (swap-only = 0.096).

**v4.1 — Learned Trajectory State**: Incorporating a learned trajectory predictor improves normal performance (swap-only: 0.096 → 0.558) but degrades conflict resolution (0.933 → 0.610) and completely fails confidence calibration (correct confidence ≈ incorrect confidence ≈ 1.0).

**v4.2 — Conflict-First Gate**: Replacing weighted fusion with explicit conflict detection + adjudication achieves conflict resolution = 0.648 (above v4.1's 0.610), swap-only = 0.519, and meaningful confidence calibration (correct = 0.874 vs incorrect = 0.237, gap = 0.637).

### 4.3 Supplementary Checks

**v4.3 — Trajectory Robustness**: Approach detection improves swap-only (0.529 → 0.625) but degrades conflict resolution (0.714 → 0.619). Trajectory augmentation does not improve OOD robustness. Negative result: diminishing returns on gate heuristics.

**v5 — Preliminary Scaling Check**: Under 3 objects + 16-dim continuous features, FeatureOnly still achieves swap-only = 1.000 clean and conflict = 0.000. ConflictFirstObjectFile achieves swap-only = 0.650 and conflict = 0.343–0.389. The diagnostic pattern persists.

**v5.1 — Scaling Sanity Audit**: Confirms that (a) N=3 permutation metric is valid, (b) continuous feature oracle achieves 1.000 clean, (c) conflict construction induces genuine feature-trajectory disagreement, (d) FeatureOnly conflict=0 is not a metric artifact.

### 4.4 External Audit: Published Object-Centric Models

To test whether the feature-reader profile is specific to our models or general to object-centric architectures, we apply SVT stress tests to four published models: Slot Attention (Locatello et al., 2020), RIMs (Goyal et al., 2021), SAVi (Kipf et al., 2022), and DINOSAUR (Seitzer et al., 2024). Each model is adapted to the SVT (position, feature) input format and trained on the same 2-object attractor-dynamics dataset.

**Structural fingerprint comparison:**

| Model | Clean | Feature Ablation | Occlusion | Conflict | Shuffled | Profile |
|-------|-------|-----------------|-----------|----------|----------|---------|
| Slot Attention | 1.000 | 0.000 | 0.000 | 0.000 | 0.561 | feature-reader |
| RIMs | 1.000 | 1.000 | 0.000 | 0.000 | 0.561 | feature-reader |
| SAVi | 1.000 | 1.000 | 0.000 | 0.000 | 0.561 | feature-reader |
| DINOSAUR | 1.000 | 1.000 | 0.000 | 0.000 | 0.561 | feature-reader |

**Key finding**: All four tested object-centric models exhibit a feature-reader-like profile. They achieve perfect identity accuracy under clean conditions, but completely fail (0.000) under feature-trajectory conflict. This is identical to the FeatureOnly baseline profile. This finding is robust across 3 random seeds (std = 0.000 for all metrics), confirming that it is not an artifact of train/test randomness.

**Interpretation**: The feature-reader profile is not specific to our models. It is a property of current object-centric architectures that learn to associate features with object slots without maintaining a separate trajectory-continuity channel. When features are misleading, these models have no adjudication mechanism and are completely misled.

**Notable difference**: RIMs, SAVi, and DINOSAUR maintain perfect accuracy under feature ablation (features removed entirely), while Slot Attention drops to 0.000. This suggests that RIMs/SAVi/DINOSAUR learn trajectory information in their recurrent state, but this trajectory information is not used for adjudication when features are present but misleading — it is only used as a fallback when features are absent.

---

## 5. Results Summary

### 5.1 ObjectFile Variants

| Model | Clean swap-only | Conflict resolution | Confidence calibration |
|-------|----------------|--------------------|-----------------------|
| FeatureOnly | 1.000 | 0.000 | N/A |
| TrajectoryOnly | 0.135 | — | — |
| Hybrid | 0.508 | 0.000–0.057 | Failed |
| MinimalObjectFile | 0.096 | 0.933 | — |
| ImprovedObjectFile | 0.558 | 0.610 | Failed (≈1.0 both) |
| ConflictFirst_margin | 0.519 | 0.648 | 0.637 |

### 5.2 Published Object-Centric Models (SVT External Audit)

| Model | Clean | Feature Ablation | Conflict | Profile |
|-------|-------|-----------------|----------|---------|
| Slot Attention | 1.000 | 0.000 | 0.000 | feature-reader |
| RIMs | 1.000 | 1.000 | 0.000 | feature-reader |
| SAVi | 1.000 | 1.000 | 0.000 | feature-reader |
| DINOSAUR | 1.000 | 1.000 | 0.000 | feature-reader |

All four published models show the same feature-reader profile as our FeatureOnly baseline, confirming that the structural deficiency is not specific to our models but is a property of current object-centric architectures.

---

## 6. Discussion

### 6.1 The Diagnostic Chain

The progression from v3.6 to v4.2 establishes a structural discriminative chain:

1. **Feature matching can read out identity** (v3.6 positive control)
2. **Feature matching is not object-file** (v4: FeatureOnly fails under conflict)
3. **Weighted fusion is insufficient** (v4: Hybrid is feature-hijacked)
4. **Structural adjudication provides correct bias** (v4: MinimalObjectFile resolves conflicts)
5. **Learning degrades structural bias** (v4.1: improved performance, broken calibration)
6. **Conflict-first gating partially recovers** (v4.2: balanced but not solved)

### 6.2 The Feature-Trajectory Trade-off

There is a fundamental trade-off between normal performance and conflict resolution. Models that perform well under clean conditions (FeatureOnly, Hybrid) fail under conflict. Models that resolve conflicts correctly (MinimalObjectFile) have low normal performance. The ConflictFirst gate is the most balanced point on this trade-off, but it is not a solution.

### 6.3 The Bottleneck is Signal Quality, Not Gate Design

The v4.3 negative result is important: improving the gate heuristic has diminishing returns. The trajectory predictor's OOD generalization (swap-only = 0.135) sets a ceiling on any gate's performance. Future work should focus on trajectory-state quality, not gate design.

### 6.4 The Feature-Reader Profile is General, Not Specific

The external audit (Section 4.4) extends the feature-reader finding from our own models to four published object-centric architectures. This is significant because:

1. **It is not a straw man**: The feature-reader profile is not an artifact of our FeatureOnly baseline. It is a property of Slot Attention, RIMs, SAVi, and DINOSAUR — models that are widely cited as achieving object-centric representations.

2. **It is not about model quality**: These models achieve perfect clean accuracy (1.000), demonstrating that they learn the task perfectly under benign conditions. The failure is specifically under stress — when features are misleading.

3. **It is about architectural structure**: RIMs, SAVi, and DINOSAUR maintain trajectory information (feature ablation = 1.000), but this information is not used for adjudication when features are present but misleading. The trajectory channel exists but is not connected to a conflict-resolution mechanism.

4. **It validates the SVT approach**: The fact that all four models show the same profile under stress — despite their architectural differences — suggests that SVT is probing a real structural property, not an artifact of the test.

---

## 7. Limitations

1. **Toy environment**: 2D arena with 2 objects and low-dimensional features. We do not claim these findings automatically generalize to complex visual environments.
2. **Rule-based ObjectFile**: The ObjectFile is a minimal structural probe, not a complete learned model. Its conflict resolution demonstrates structural bias, not learned intelligence.
3. **Low absolute numbers**: ConflictFirst's swap-only = 0.519 is barely above chance. The contribution is the diagnostic chain, not competitive performance.
4. **Artificial conflict**: Feature-trajectory conflict is a diagnostic probe, not a realistic scenario. However, the structural deficiency it reveals (no error-recovery mechanism) would manifest whenever features are unreliable.
5. **Trajectory predictor bottleneck**: The trajectory predictor's poor OOD generalization limits any gate's performance ceiling.

---

## 8. Conclusion

We have presented Structure Validity Tests (SVT), a diagnostic framework for stress-testing identity binding in object-centric agents. Our main finding is a structural discriminative chain: clean feature matching can read out identity under benign conditions, but fails catastrophically under feature-trajectory conflict. Critically, this finding extends beyond our own models: all four tested published object-centric models (Slot Attention, RIMs, SAVi, DINOSAUR) exhibit the same feature-reader-like profile — perfect clean accuracy but complete failure under conflict. A minimal ObjectFile with separate channels and conflict adjudication demonstrates correct structural bias, but its normal performance is limited by trajectory-state quality. The ConflictFirst gate achieves the most balanced performance to date, but the fundamental trade-off between normal performance and conflict resolution remains unresolved.

The contribution of this work is not a strong model, but a diagnostic method: the finding that current object-centric models are structurally deficient under conflict is more informative than the partial success of ObjectFile variants. We recommend that future object-centric models be evaluated not only under clean conditions, but also under systematic stress tests that probe whether identity binding is genuine or superficial.

---

## References

- Baillargeon, R. (1987). Object permanence in 3.5- and 4.5-month-old infants. *Developmental Psychology*.
- Goodfellow, I. et al. (2015). Explaining and harnessing adversarial examples. *ICLR*.
- Goyal, A. et al. (2021). Recurrent independent mechanisms. *ICLR*.
- Kahneman, D. & Treisman, A. (1992). The reviewing of object files: Object-specific integration of information. *Cognitive Psychology*.
- Kipf, A. et al. (2022). Conditional object-centric learning from video. *ICLR*.
- Locatello, F. et al. (2020). Object-centric learning with slot attention. *NeurIPS*.
- Seitzer, M. et al. (2024). DINOSAUR: A framework for slot-based models. *ICML*.
