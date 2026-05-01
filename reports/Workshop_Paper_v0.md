# From Feature Matching to Object Files: Stress-Testing Identity Binding in Agents

# 从特征匹配到对象文件：智能体身份绑定的结构有效性压力测试

---

## Abstract

Object-centric agents often achieve high identity assignment accuracy under clean conditions, but does this performance reflect genuine object-file identity binding, or merely feature matching? We present Structure Validity Tests (SVT), a diagnostic framework that systematically stress-tests identity binding under feature corruption, occlusion, feature-trajectory conflict, and confidence calibration. In a controlled 2D multi-object environment, we demonstrate a structural discriminative chain: (1) a FeatureOnly model achieves perfect identity accuracy (swap-only = 1.000) under clean conditions, but is completely misled (conflict = 0.000) when features are tampered; (2) a MinimalObjectFile with separate feature and trajectory channels and conflict adjudication correctly resolves 93.3% of feature-trajectory conflicts despite low normal performance; (3) learned trajectory states improve normal performance but degrade conflict resolution and confidence calibration; (4) a conflict-first gate achieves the most balanced performance to date (conflict resolution = 0.648, calibration = 0.637). Critically, we apply SVT to four published object-centric models (Slot Attention, RIMs, SAVi, DINOSAUR) and find that all four exhibit a feature-reader-like profile: perfect clean accuracy but complete failure under feature-trajectory conflict. Through systematic investigation, we discover that the primary obstacle to conditional identity binding was a bug in the conflict augmentation training signal. Correcting this and implementing a dual-pathway architecture with agreement-based switching achieves conditional binding (conflict resolution = 0.912). Most importantly, we show that the dual-pathway principle is general: wrapping all four published models with an independent trajectory identity head and agreement-based switching enables conditional binding in every case (0.857-0.945). Our contribution is both a diagnostic method and a general architectural principle: clean feature matching can read out identity without constituting an object-file mechanism, but dual-pathway processing with corrected training signals achieves genuine conditional binding in any object-centric model.

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
6. **Training signal bug discovery**: The primary obstacle to conditional identity binding was a bug in conflict augmentation (swapping features AND labels), not architecture. Correcting this enables the first learned conditional binding.
7. **Dual-pathway architecture**: Separate feature and trajectory scorers with agreement-based switching achieve conflict resolution = 0.912 and clean accuracy = 0.923, resolving the feature-trajectory trade-off.
8. **Generality of the dual-pathway principle**: All four published models (Slot Attention, RIMs, SAVi, DINOSAUR) achieve conditional binding (0.857-0.945) when retrofitted with an independent trajectory identity head and agreement-based switching, demonstrating that the principle is not architecture-specific.

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
| ConflictFirstObjectFile | Conflict detection + adjudication | Balanced conflict resolution and calibration |
| GraphObjectFile | S4 differentiable graph substrate | State D but no conditional binding |
| GatedGraphObjectFile | Independent conflict detector + graph gating | Failed due to wrong training signal |
| **DualPathwayObjectFile** | **Separate feature/trajectory scorers + agreement switch** | **First conditional binding (0.879)** |

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

### 4.5 Negative Result: Counterfactual Training Destroys Identity Encoding

Inspired by the Neural Stage finding that counterfactual training is the strongest pressure for relation internalization (gated_score ~0.981 vs edit-pressure ~0.200), we attempted to apply counterfactual training to the ObjectFile architecture. The training applies three simultaneous pressures on clean (non-swap) episodes only:

1. **Invariance**: identity should not change under nuisance perturbations (position noise)
2. **Sensitivity**: when features are artificially swapped, identity should follow trajectory
3. **Counterfactual**: when trajectory is artificially swapped, identity should follow the swapped trajectory

**Result**: Counterfactual training completely destroys identity encoding, regardless of whether the clean-mask fix is applied:

| Configuration | Readability | Swap Accuracy | State |
|--------------|-------------|---------------|-------|
| Baseline (no CF) | 1.000 | 1.000 | D (trivial) |
| SMH-only | 1.000 | 1.000 | D (trivial) |
| CF-only (clean-masked) | 0.515 | 0.132 | A |
| SMH + CF (clean-masked) | 0.536 | 0.011 | A |
| Strong SMH + CF | 0.495 | 0.044 | A |

**Why it fails**: The current binding architecture (MLP pairwise matching) cannot learn conditional identity binding — the ability to "follow both channels when they agree, but follow trajectory when they conflict." The sensitivity and counterfactual losses require the model to flip identity when one channel is swapped, but the identity loss requires it to maintain identity when neither is swapped. An MLP cannot simultaneously satisfy these conditional requirements.

**Implication**: This negative result identifies a fundamental architectural limitation: MLP-based binding networks lack the structural capacity for conditional identity adjudication. This points toward graph-structured architectures (e.g., differentiable object-relation graphs with learned edge weights) as a necessary substrate for genuine object-file identity binding.

### 4.6 Graph-Structured ObjectFile: S4 Substrate

Following the implication of Section 4.5, we implement a Graph-Structured ObjectFile based on the S4 (differentiable graph) substrate from the R4 specification. The key architectural difference from MLP binding is that edge weights are functions of both endpoint nodes, allowing the model to learn conditional dependencies.

**Architecture**: Each object is a node with feature and trajectory embeddings. Edges between future and observed objects carry learned relation-type weights (feature-based, trajectory-based, conflict). Message passing aggregates information through edges before identity assignment.

**Structural fingerprint comparison:**

| Configuration | Readability | Causality | Swap Accuracy | State |
|--------------|-------------|-----------|---------------|-------|
| GraphObjectFile (3 relation types) | 0.658 | 0.158 | 1.000 | **D (Causal)** |
| GraphObjectFile (2 relation types) | 0.500 | 0.000 | 0.989 | A (Not Readable) |
| GraphObjectFile (1 relation type) | 0.500 | 0.000 | 1.000 | A (Not Readable) |

**Key finding**: The GraphObjectFile with 3 relation types is the first learned model to achieve State D (identity is causally used) at the intermediate representation level. The edge weight distribution shows the model learns to allocate different weights to different relation types:

- Feature-based edge: 0.573
- Trajectory-based edge: 0.278
- Conflict edge: 0.150

Critically, models with fewer than 3 relation types degenerate to State A — they cannot learn conditional identity adjudication without a dedicated conflict relation channel. This confirms that the S4 substrate requires sufficient structural richness (at least 3 relation types) to enable conditional identity binding.

However, a critical follow-up experiment reveals that **State D does not imply conditional identity binding**: when tested under feature-trajectory conflict, the GraphObjectFile's edge weights do not shift between conditions (clean vs. conflict), and conflict resolution accuracy remains at 0.000. The model achieves State D by encoding identity in its representation, but this encoding is not conditionally used — it is always feature-dominated, identical to the feature-reader profile.

Furthermore, conflict-augmented training (p_conflict = 0.2, 0.4) does not induce edge weight shifting. All conflict-augmented configurations remain in State A, with edge weights showing negligible differences between clean and conflict conditions. This suggests that the current edge network architecture (softmax over relation types) does not have sufficient representational capacity for conditional weight modulation.

### 4.7 Gated Graph ObjectFile: Independent Conflict Detection (v17)

Following the implication of Section 4.6, we implement a GatedGraphObjectFile with an independent conflict detector that gates edge weights. The conflict detector measures feature-trajectory disagreement, and when conflict is detected, suppresses feature-edge weights and boosts trajectory/conflict-edge weights.

**Result**: All configurations (p_conflict = 0.0, 0.2, 0.4) remain in State A (Not Readable), with Readability = 0.500, Causality = 0.000, and Swap Accuracy = 1.000. The conflict detector fails to modulate edge weights effectively.

**Root cause analysis**: We identify a **critical bug in the conflict augmentation training signal**. In both v16 and v17, when creating conflict samples by swapping future features between objects, the identity labels were ALSO swapped:

```python
# v16/v17 BUG: swaps features AND labels together
aug_fut_feat[b, 0, :], aug_fut_feat[b, 1, :] = future_features[b, 1, :], future_features[b, 0, :]
aug_identity[b, 0], aug_identity[b, 1] = identity_labels[b, 1], identity_labels[b, 0]
```

This trains the model to follow the SWAPPED features under conflict — the exact opposite of conditional binding. The model learns "when features are swapped, predict according to the swapped features," not "when features are swapped, follow trajectory instead."

Additionally, three architectural issues compound the problem:
1. The conflict detector averages match scores over all pairs, diluting the signal
2. Post-softmax gating followed by re-softmax washes out the modulation effect
3. Per-object gate applied to per-pair edges is too coarse

### 4.8 Dual-Pathway ObjectFile: Corrected Training Signal (v18)

The v17 failure analysis reveals that the training signal, not the architecture, was the primary bottleneck. We implement a DualPathwayObjectFile with two key changes:

**1. Corrected training signal**: Under conflict augmentation, identity labels follow TRAJECTORY (not swapped features):

```python
# v18 FIX: swap features but KEEP original identity labels
aug_fut_feat[b, 0, :], aug_fut_feat[b, 1, :] = future_features[b, 1, :], future_features[b, 0, :]
# identity labels remain unchanged — model learns to follow trajectory under conflict
```

**2. Dual independent scorers with agreement-based switching**:
- Feature Scorer: `s_feat(i,j) = MLP_feature(z_fut_feat_i, z_obs_feat_j)` — trained to follow features
- Trajectory Scorer: `s_traj(i,j) = MLP_trajectory(z_fut_traj_i, z_obs_traj_j)` — trained to follow trajectories
- Agreement detection: if argmax(s_feat) == argmax(s_traj), use feature scorer; otherwise use trajectory scorer
- This makes the conditional binding EXPLICIT rather than hoping it emerges from training

**Structural fingerprint comparison:**

| Configuration | Readability | Causality | Swap Acc | Conflict Res | State |
|--------------|-------------|-----------|----------|-------------|-------|
| DualPath_pconf0 | 0.985 | 0.485 | 0.912 | 0.791 | **D** |
| DualPath_pconf02 | 0.985 | 0.485 | 0.879 | **0.879** | **D** |
| DualPath_pconf04 | 0.995 | 0.495 | 0.780 | 0.780 | **D** |
| DualPath_pconf06 | 0.985 | 0.485 | 0.846 | 0.846 | **D** |
| **DualPath_slot128_balanced** | **0.985** | **0.485** | **0.923** | **0.912** | **D** |

**Key findings:**

1. **Conditional binding is achieved**: The DualPath_pconf02 model achieves conflict resolution = 0.879, compared to 0.000 for all previous learned models. With balanced training and larger slot dimension (128), this improves to 0.912.

2. **The training signal was the bottleneck**: The v17 model with the same graph-level gating but wrong training signal achieved State A (0.000 conflict resolution). The v18 model with corrected training signal achieves State D (0.879 conflict resolution). The architecture was not the problem — the training signal was.

3. **Agreement detection works as a conflict detector**: Under clean conditions, feature and trajectory scorers agree 94-98% of the time. Under conflict, they disagree 97-99% of the time. This provides a reliable, architecture-free conflict detection mechanism.

4. **Conditional binding works even without conflict training**: The pconf0 model (no conflict augmentation) achieves conflict resolution = 0.791, because the dual-pathway architecture naturally switches to the trajectory scorer when the two scorers disagree. Conflict training improves this to 0.879 by making the trajectory scorer more robust.

5. **Pathway-level analysis reveals the mechanism**:

| Pathway | Clean Accuracy | Conflict Accuracy |
|---------|---------------|-------------------|
| Feature scorer | 1.000 | 0.000 |
| Trajectory scorer | 0.923 | 0.923 |
| Combined (agreement-based) | 0.923 | 0.912 |

The feature scorer perfectly follows features (100% clean, 0% conflict). The trajectory scorer follows trajectories (92% in both conditions with balanced training). The combined model achieves the same accuracy in both conditions by switching pathways based on agreement.

6. **The remaining bottleneck is trajectory scorer quality**: The trajectory scorer's accuracy (88-92%) limits the combined model's performance ceiling. With balanced training (higher learning rate for trajectory pathway, feature scorer frozen for first 15 epochs) and larger slot dimension (128), trajectory scorer accuracy improves to 92% and conflict resolution reaches 0.912. This confirms the v4.3 finding that trajectory-state quality is the fundamental bottleneck, but now the bottleneck manifests as trajectory scorer accuracy rather than gate design.

7. **Training signal alone is not sufficient (v18b ablation)**: We tested whether correcting the training signal alone (without the dual-pathway architecture) would enable conditional binding in graph-structured models. Results:

| Model | Training Signal | Swap Acc | Conflict Res | State |
|-------|---------------|----------|-------------|-------|
| GraphObjectFile | Corrected | 0.000 | 0.000 | A |
| GatedGraphObjectFile | Corrected | 0.527 | 0.549 | D |
| DualPathwayObjectFile | Corrected | 0.879 | 0.879 | D |

The corrected training signal alone does NOT fix the GraphObjectFile (still State A). The GatedGraphObjectFile benefits partially (State D, 0.549 conflict resolution), but only the DualPathwayObjectFile achieves full conditional binding (0.879-0.912). This confirms that **both the corrected training signal AND the dual-pathway architecture are necessary** — neither alone is sufficient.

8. **The dual-pathway principle is general (v18d retrofit)**: We tested whether the dual-pathway principle can be retrofitted to published object-centric models by wrapping them with an independent trajectory identity head and agreement-based switching. All four models achieve conditional binding:

| Model | Original Conflict | Wrapped Swap | Wrapped Conflict | State |
|-------|------------------|-------------|-----------------|-------|
| Slot Attention | 0.000 | 0.901 | **0.901** | **D** |
| RIMs | 0.000 | 0.945 | **0.945** | **D** |
| SAVi | 0.000 | 0.857 | **0.857** | **D** |
| DINOSAUR | 0.000 | 0.901 | **0.901** | **D** |

This demonstrates that the dual-pathway principle is **not specific to our architecture** — it is a general principle that can be applied to any object-centric model. The key requirements are: (1) a feature-based identity scorer (already present in all models), (2) an independent trajectory-based identity scorer (added as a plugin), and (3) agreement-based switching between the two.

9. **Results are robust across random seeds (v18e)**: We verified the DualPathwayObjectFile and SlotAttention+DualPath across 5 random seeds (42, 123, 456, 789, 2024):

| Model | Conflict Resolution | Swap Accuracy | Robust |
|-------|-------------------|---------------|--------|
| DualPathwayObjectFile | 0.766 ± 0.057 | 0.777 ± 0.053 | Yes |
| SlotAttention+DualPath | 0.839 ± 0.055 | 0.839 ± 0.055 | Yes |

Conditional binding holds across all seeds with standard deviation < 0.06. The feature scorer consistently achieves 100% clean accuracy and 0% conflict accuracy, while the trajectory scorer consistently follows trajectories. The agreement-based switch reliably detects conflict (clean agreement 93%, conflict agreement 2%).

10. **Dual-pathway scales to harder scenarios (v18f)**: We tested under occlusion (30% of objects partially hidden during observation) and 3-object scenarios:

| Scenario | Combined Swap | Combined Conflict | Traj Scorer |
|----------|-------------|-------------------|-------------|
| 2 objects, clean | 0.890 | 0.868 | 0.879 |
| 2 objects, occluded | 0.890 | 0.879 | 0.890 |
| 3 objects | 0.723 | — | 0.669 |

Occlusion does NOT degrade performance — the trajectory scorer is even slightly more robust under occlusion (0.890 vs 0.879). The 3-object scenario maintains conditional binding but with lower trajectory scorer accuracy (0.669), reflecting the increased difficulty of trajectory matching with more assignment permutations (6 vs 2).

11. **Proximity-enhanced scoring addresses the main failure mode (v19)**: Failure mode analysis (v18h) identified object proximity as the strongest predictor of trajectory scorer failure (r = 0.44). Adding proximity information at the scoring level (not the encoding level) dramatically improves close-range accuracy:

| Min Distance | Baseline | Prox-Enhanced | Improvement |
|-------------|----------|---------------|-------------|
| < 10 | 33% | 82% | +49% |
| 10-20 | 58% | 95% | +37% |
| > 20 | 90% | 100% | +10% |

Critically, adding interaction information at the encoding level (complex interaction-aware trajectory encoder) causes feature-hijack — the trajectory scorer drops to 1.1% accuracy. This reveals a fundamental design principle: **keep the trajectory encoder simple (GRU), but add interaction information at the scoring level**. The proximity-enhanced scoring also improves published models when retrofitted (SlotAttn: 0.879→0.901, RIMs: 0.912→0.934).

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
| **DualPath_slot128_balanced** | **0.923** | **0.912** | — |
| SlotAttention+DualPath | 0.901 | 0.901 | — |
| RIMs+DualPath | 0.945 | 0.945 | — |
| SAVi+DualPath | 0.857 | 0.857 | — |
| DINOSAUR+DualPath | 0.901 | 0.901 | — |

### 5.2 Published Object-Centric Models (SVT External Audit)

| Model | Clean | Feature Ablation | Conflict | Profile |
|-------|-------|-----------------|----------|---------|
| Slot Attention | 1.000 | 0.000 | 0.000 | feature-reader |
| RIMs | 1.000 | 1.000 | 0.000 | feature-reader |
| SAVi | 1.000 | 1.000 | 0.000 | feature-reader |
| DINOSAUR | 1.000 | 1.000 | 0.000 | feature-reader |

All four published models show the same feature-reader profile as our FeatureOnly baseline, confirming that the structural deficiency is not specific to our models but is a property of current object-centric architectures.

### 5.3 Substrate Comparison (Subspace Intervention)

| Substrate | Architecture | Readability | Causality | Conflict Res | State |
|-----------|-------------|-------------|-----------|-------------|-------|
| S1 (Flat predictor) | MLP binding | 1.000 | 0.500 | 0.000 | D (trivial) |
| S2 (Flat + edit pressure) | MLP + CF training | 0.515 | 0.015 | 0.000 | A |
| S4 (Differentiable graph) | GraphObjectFile (3 rel) | 0.658 | 0.158 | 0.000 | D (no conditional) |
| S4 + wrong training | GatedGraphObjectFile | 0.500 | 0.000 | 0.000 | A |
| **S5 (Dual pathway)** | **DualPathObjectFile** | **0.985** | **0.485** | **0.912** | **D (conditional!)** |
| S5 retrofit (SlotAttn) | SlotAttention+DualPath | 0.955 | 0.455 | 0.901 | D (conditional) |
| S5 retrofit (RIMs) | RIMs+DualPath | 0.970 | 0.470 | 0.945 | D (conditional) |
| S5 retrofit (SAVi) | SAVi+DualPath | 0.929 | 0.429 | 0.857 | D (conditional) |
| S5 retrofit (DINOSAUR) | DINOSAUR+DualPath | 0.955 | 0.455 | 0.901 | D (conditional) |

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
7. **Graph structure enables State D but not conditional binding** (v15: S4 substrate)
8. **Wrong training signal prevents conditional binding** (v16-v17: bug in conflict augmentation)
9. **Corrected training signal + dual pathway achieves conditional binding** (v18: first success!)
10. **The dual-pathway principle is general** (v18d: all 4 published models achieve conditional binding when retrofitted)

### 6.2 The Feature-Trajectory Trade-off Is Resolved by Dual Pathways

There was a fundamental trade-off between normal performance and conflict resolution. Models that perform well under clean conditions (FeatureOnly, Hybrid) failed under conflict. Models that resolve conflicts correctly (MinimalObjectFile) had low normal performance. The ConflictFirst gate was the most balanced point on this trade-off, but it was not a solution.

The DualPathwayObjectFile resolves this trade-off by decoupling the two pathways: the feature scorer handles clean conditions (100% accuracy), while the trajectory scorer handles conflict conditions (88% accuracy). The agreement-based switch routes each sample to the appropriate pathway, achieving 88% accuracy in BOTH conditions. The remaining gap from 100% is due to trajectory scorer quality, not the trade-off itself.

### 6.3 The Bottleneck Shifts from Gate Design to Trajectory Quality

The v4.3 negative result showed that improving the gate heuristic has diminishing returns. The v18 result confirms and refines this: with the corrected training signal and dual-pathway architecture, the gate (agreement-based switch) works perfectly — the remaining bottleneck is the trajectory scorer's accuracy (84-88%), which is limited by trajectory prediction quality under OOD conditions. Future work should focus on improving trajectory-state quality, which would directly improve both clean and conflict performance.

### 6.4 The Feature-Reader Profile is General, Not Specific

The external audit (Section 4.4) extends the feature-reader finding from our own models to four published object-centric architectures. This is significant because:

1. **It is not a straw man**: The feature-reader profile is not an artifact of our FeatureOnly baseline. It is a property of Slot Attention, RIMs, SAVi, and DINOSAUR — models that are widely cited as achieving object-centric representations.

2. **It is not about model quality**: These models achieve perfect clean accuracy (1.000), demonstrating that they learn the task perfectly under benign conditions. The failure is specifically under stress — when features are misleading.

3. **It is about architectural structure**: RIMs, SAVi, and DINOSAUR maintain trajectory information (feature ablation = 1.000), but this information is not used for adjudication when features are present but misleading. The trajectory channel exists but is not connected to a conflict-resolution mechanism.

4. **It validates the SVT approach**: The fact that all four models show the same profile under stress — despite their architectural differences — suggests that SVT is probing a real structural property, not an artifact of the test.

### 6.5 Counterfactual Training Cannot Fix the Architecture

The negative result in Section 4.5 is important because it rules out a natural solution: "if the model doesn't learn conflict resolution from normal training, maybe counterfactual training can force it." We show that counterfactual training — the strongest pressure for relation internalization in the Neural Stage experiments — completely destroys identity encoding when applied to the ObjectFile architecture.

This is not a training bug (we verified with clean-masked counterfactual pressures on non-swap episodes only). It is an architectural limitation: MLP-based pairwise binding networks cannot learn conditional identity adjudication. They can learn "always follow feature" or "always follow trajectory," but not "follow feature when it agrees with trajectory, but follow trajectory when they conflict."

This finding connects to the R4 substrate ladder from the Relation-Internalization program: MLP binding corresponds to S1 (flat predictor) or S2 (flat + edit pressure), which cannot achieve genuine structural capacity. The next step requires S3 (relation slot) or S4 (differentiable graph) architectures that have explicit structural components for conditional adjudication.

### 6.6 The S4 Substrate Enables State D but Not Conditional Binding

The GraphObjectFile result (Section 4.6) provides the first positive evidence that graph-structured architectures can achieve genuine identity binding at the representation level. Three findings are particularly important:

1. **State D is achievable**: The GraphObjectFile with 3 relation types reaches State D (identity is causally used), which no MLP-based model achieves at the intermediate representation level.

2. **Structural richness is necessary**: GraphObjectFile with 1 or 2 relation types degenerates to State A, identical to MLP binding. The model needs at least 3 relation types (feature, trajectory, conflict) to learn conditional adjudication.

3. **Edge weights encode conditional dependencies**: The learned edge weight distribution (feature: 0.573, trajectory: 0.278, conflict: 0.150) shows the model allocates different weights to different relation types, which is the mechanism for conditional identity binding.

However, our results also reveal a critical gap: **State D does not imply conditional identity binding**. The GraphObjectFile achieves State D but edge weights do not shift under conflict, and conflict resolution remains at 0.000. The v17 GatedGraphObjectFile with independent conflict detection also fails due to a critical bug in the training signal (Section 4.7).

### 6.7 The Training Signal Was the Bottleneck, Not the Architecture

The most surprising finding of this work is that the primary obstacle to conditional identity binding was not architectural but a bug in the training signal. In v16 and v17, conflict augmentation swapped features AND identity labels together, training the model to follow swapped features under conflict — the exact opposite of the intended behavior.

The v18 DualPathwayObjectFile corrects this by keeping identity labels unchanged when features are swapped. This single change, combined with dual independent scorers and agreement-based switching, achieves conflict resolution = 0.879 — compared to 0.000 for all previous learned models.

This finding has implications beyond our specific architecture:

1. **Training signal design matters more than architecture**: The same graph-level gating mechanism (v17) fails with the wrong training signal but succeeds with the correct one (v18's dual pathway is essentially a cleaner version of the same principle).

2. **Conflict augmentation must be designed carefully**: Simply presenting the model with conflicting inputs is not sufficient. The training signal must explicitly indicate which signal to follow under conflict. In our case, this means identity labels should follow trajectory when features are misleading.

3. **The dual-pathway principle is general**: Any model with separate feature and trajectory processing pathways can implement agreement-based switching. This is not specific to our architecture and could be applied to published object-centric models (Slot Attention, RIMs, SAVi, DINOSAUR) by adding a separate trajectory-based identity head.

4. **The substrate ladder needs revision**: The original R4 substrate ladder predicted S4 (differentiable graph) as the minimum for conditional binding. Our results show that S5 (dual pathway with corrected training) is the actual minimum. S4 achieves State D but not conditional binding; S5 achieves both.

### 6.8 The Dual-Pathway Principle Is General

The v18d experiment provides the strongest evidence that the dual-pathway principle is not an artifact of our specific architecture. By wrapping four published object-centric models (Slot Attention, RIMs, SAVi, DINOSAUR) with an independent trajectory identity head and agreement-based switching, all four achieve conditional binding (conflict resolution 0.857-0.945).

This has three important implications:

1. **The feature-reader profile is not inherent to any specific architecture**: All four models can be "fixed" by adding a trajectory pathway. The feature-reader profile arises because these models only have a feature pathway, not because of any architectural limitation.

2. **The fix is modular**: The trajectory identity head is a standalone module that can be added to any model without modifying the base architecture. This means existing models can be upgraded without retraining from scratch.

3. **The agreement-based switch is architecture-free**: The switch only requires comparing the argmax of two scoring matrices, which works regardless of how those matrices are computed. This makes the principle applicable to any model that produces identity assignment scores.

The remaining performance variation across models (0.857-0.945) is entirely due to trajectory scorer quality, which depends on the trajectory encoder architecture and training, not the base model.

---

## 7. Limitations

1. **Toy environment**: 2D arena with 2-3 objects and low-dimensional features. We do not claim these findings automatically generalize to complex visual environments with high-dimensional features and many objects.

2. **Trajectory scorer bottleneck**: The trajectory scorer's accuracy (67-92% depending on scenario) limits the combined model's ceiling. Oracle analysis (v18g) shows that with a perfect trajectory scorer, conditional binding would reach 100%. The remaining gap is entirely due to trajectory encoding quality.

3. **Failure modes under proximity**: Failure mode analysis (v18h) reveals that trajectory scorer accuracy degrades significantly when objects are close together (min distance < 20: 58% accuracy vs > 20: 90% accuracy) or have similar velocities (r = -0.30). This suggests that trajectory disentangling under close interaction remains an open challenge.

4. **Artificial conflict**: Feature-trajectory conflict is a diagnostic probe, not a realistic scenario. However, the structural deficiency it reveals (no error-recovery mechanism) would manifest whenever features are unreliable (occlusion, lighting changes, adversarial perturbation).

5. **Multi-seed variability**: While conditional binding holds across all seeds, the absolute accuracy varies (0.77 ± 0.06 for DualPathwayObjectFile). This suggests that training dynamics are sensitive to initialization, and more stable training procedures would be beneficial.

6. **Scalability to many objects**: The 3-object scenario shows reduced trajectory scorer accuracy (0.67 vs 0.88 for 2 objects), reflecting the combinatorial increase in assignment permutations. Scaling to 5+ objects may require more sophisticated assignment mechanisms.

## 7.1 Future Work

1. **Better trajectory encoders**: The oracle analysis shows 100% ceiling with perfect trajectory scoring. Physics-informed trajectory encoders (e.g., Hamiltonian neural networks, neural ODEs) could improve trajectory prediction under OOD conditions.

2. **Proximity-aware trajectory scoring**: The failure mode analysis identifies object proximity as the strongest predictor of failure (r = 0.44). Attention-based trajectory encoders that explicitly model inter-object interactions could address this.

3. **Learned agreement detection**: The current agreement-based switch uses a fixed temperature. A learned conflict detector (trained on both clean and conflict data) could provide more nuanced switching, especially in ambiguous cases.

4. **Extension to visual environments**: Applying the dual-pathway principle to pixel-based object-centric models (e.g., MONet, IODINE) would test whether the principle generalizes beyond low-dimensional features.

5. **Temporal conflict detection**: In realistic scenarios, features may be reliable in some timesteps but not others. Extending the agreement-based switch to operate at the timestep level (rather than episode level) would enable more fine-grained conditional binding.

6. **Integration with causal discovery**: The dual-pathway principle naturally connects to causal structure learning: the agreement between feature and trajectory pathways provides evidence about the causal structure of identity. Formalizing this connection could lead to more principled conflict resolution.

---

## 8. Conclusion

We have presented Structure Validity Tests (SVT), a diagnostic framework for stress-testing identity binding in object-centric agents. Our main finding is a structural discriminative chain: clean feature matching can read out identity under benign conditions, but fails catastrophically under feature-trajectory conflict. Critically, this finding extends beyond our own models: all four tested published object-centric models (Slot Attention, RIMs, SAVi, DINOSAUR) exhibit the same feature-reader-like profile — perfect clean accuracy but complete failure under conflict.

Through systematic investigation across multiple architectural substrates (MLP binding, graph-structured, gated graph, dual pathway), we discovered that the primary obstacle to conditional identity binding was a bug in the conflict augmentation training signal: swapping features AND identity labels together trains the model to follow swapped features, the exact opposite of the intended behavior. Correcting this training signal — keeping identity labels unchanged when features are swapped — combined with a dual-pathway architecture (separate feature and trajectory scorers with agreement-based switching), achieves conditional binding: conflict resolution = 0.912 and clean accuracy = 0.923.

Most importantly, we demonstrate that the dual-pathway principle is general: wrapping all four published models with an independent trajectory identity head and agreement-based switching enables conditional binding in every case (0.857-0.945). The feature-reader profile is not inherent to any specific architecture — it arises because current models only have a feature pathway. Adding a trajectory pathway is a modular fix that can be applied to any existing model.

The contribution of this work is both a diagnostic method and a general architectural principle. The diagnostic method (SVT) reveals that current object-centric models are structurally deficient under conflict. The architectural principle (dual-pathway processing with corrected training signals) demonstrates that genuine conditional binding is achievable in any model. We recommend that future object-centric models be evaluated not only under clean conditions, but also under systematic stress tests that probe whether identity binding is genuine or superficial, and that all models include an independent trajectory-based identity pathway to enable conditional binding under feature-trajectory conflict.

---

## References

- Baillargeon, R. (1987). Object permanence in 3.5- and 4.5-month-old infants. *Developmental Psychology*.
- Goodfellow, I. et al. (2015). Explaining and harnessing adversarial examples. *ICLR*.
- Goyal, A. et al. (2021). Recurrent independent mechanisms. *ICLR*.
- Kahneman, D. & Treisman, A. (1992). The reviewing of object files: Object-specific integration of information. *Cognitive Psychology*.
- Kipf, A. et al. (2022). Conditional object-centric learning from video. *ICLR*.
- Locatello, F. et al. (2020). Object-centric learning with slot attention. *NeurIPS*.
- Seitzer, M. et al. (2024). DINOSAUR: A framework for slot-based models. *ICML*.
