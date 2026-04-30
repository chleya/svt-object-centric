# Paper Outline: From Feature Matching to Object Files

## Title

From Feature Matching to Object Files: Stress-Testing Identity Binding in Agents

## 主贡献声明

本文的主贡献不是某个模型或某个机制，而是一条结构诊断链：

> Clean feature matching can read out identity, but object-file requires structural adjudication across feature, trajectory, occlusion, and conflict signals.

这条诊断链通过四阶段递进实验建立，每一阶段回答一个研究问题，每一阶段的负结果推动下一阶段的设计。

---

## 论文结构

### 1. Introduction (1.5 pages)

**Opening hook**: Current agent models achieve high prediction accuracy, but does prediction equal identity binding? Can a model that perfectly matches features under clean conditions still maintain object identity when features are unreliable?

**Core claim**: We demonstrate a structural discriminative chain showing that:
1. Clean feature matching ≠ object-file
2. Weighted hybrid fusion is insufficient for conflict resolution
3. Conflict-first gating is a necessary (but not sufficient) component of object-file mechanisms
4. The remaining bottleneck is trajectory-state quality, not gate design

**Contributions**:
- A stress-testing framework (SVT) that systematically evaluates identity binding under feature noise, occlusion, and feature-trajectory conflict
- Empirical evidence that FeatureOnly (100% clean accuracy) is 100% misled by tampered features, while MinimalObjectFile (9.6% clean accuracy) correctly resolves 93.3% of conflicts
- The ConflictFirst gate, the first mechanism to achieve both meaningful confidence calibration and conflict resolution above chance
- A negative result (v4.3) showing that gate heuristic improvements have diminishing returns, identifying trajectory-state quality as the fundamental bottleneck

### 2. Related Work (1 page)

- **Object permanence in developmental psychology**: Piaget, Baillargeon, violation-of-expectation paradigm
- **Object-file theory in cognitive science**: Kahneman & Treisman, object-file as mid-level representation
- **Identity binding in computer vision**: MOT, re-identification, appearance vs motion cues
- **Stress testing and adversarial evaluation**: adversarial examples, distribution shift, OOD generalization
- **Relation internalization**: structural vs statistical learning, relational reasoning

### 3. Method (2.5 pages)

#### 3.1 Problem Formulation
- Identity binding as assignment problem: given observed features/positions and future features/positions, assign future objects to observed objects
- Swap episodes: object identities are exchanged at a random timestep
- identity_swap_only metric: accuracy on swap episodes only (avoids no-swap bias)

#### 3.2 Stress Tests
- **Feature Ablation**: shuffle/zero features → measure feature_dependency_score
- **Feature Noise**: Gaussian noise (σ=0.1, 0.3, 0.5) → test robustness
- **Occlusion Without Feature**: zero features during occlusion → test occlusion persistence
- **Feature-Trajectory Conflict**: flip future features in no-swap episodes → test conflict resolution

#### 3.3 Mechanisms
- **FeatureOnly**: identity via cosine similarity of temporal-aligned feature keys
- **TrajectoryOnly**: identity via nearest predicted position
- **Hybrid**: weighted combination of feature and trajectory logits
- **MinimalObjectFile**: rule-based mechanism maintaining identity_key + trajectory_state + occlusion_state
- **ImprovedObjectFile**: + learned trajectory predictor + confidence calibration + adaptive weighting
- **ConflictFirstObjectFile**: + conflict detection → margin comparison → adjudicate/abstain

#### 3.4 Evaluation Metrics
- identity_swap_only, feature_dependency_score, trajectory_dependency_score
- Conflict resolution rate, trajectory correct rate, feature wrong rate
- Confidence calibration: correct vs incorrect confidence, calibration error
- Abstention rate, accuracy when not abstaining

### 4. Experiments (4 pages)

#### 4.1 Setup
- 2D toy world: 64×64 arena, attractor/vortex dynamics
- Train on attractor, test on vortex (OOD)
- 2 objects, 2D one-hot features
- 1000 training episodes, 200 test episodes

#### 4.2 Stage 1: Clean Feature Matching (Positive Control)
- Temporal-aligned feature key achieves swap-only=1.000
- Feature dependency=0.490
- This is the ceiling under clean conditions

#### 4.3 Stage 2: Feature Matching Fails Under Conflict
- FeatureOnly: conflict identity=0.000 (100% misled)
- Hybrid: conflict identity=0.000-0.057 (95% misled)
- MinimalObjectFile: conflict identity=0.933 (93.3% correct)
- Feature reader ≠ Object-file keeper

#### 4.4 Stage 3: Learned Trajectory Helps But Calibration Fails
- ImprovedObjectFile: swap-only 0.096→0.558
- But conflict resolution drops: 0.933→0.610
- Confidence correct=incorrect=1.0 (calibration fails)
- Conflict gate always chooses feature (surrender, not adjudication)

#### 4.5 Stage 4: Conflict-First Gate
- Four strategies: prefer_trajectory, prefer_feature, abstain, margin_gated
- margin_gated: conflict=0.648, swap=0.519, conf_correct=0.874, conf_incorrect=0.237
- First meaningful confidence calibration in the series
- Four pass criteria all met

#### 4.6 Supplementary: Gate Heuristic Diminishing Returns (v4.3)
- Approach detection: swap 0.529→0.625, conflict 0.714→0.619
- Trajectory augmentation: no improvement
- Confirms bottleneck is trajectory-state quality, not gate design

### 5. Analysis (2 pages)

#### 5.1 Core Findings
1. Prediction ≠ Identity Binding
2. Clean Feature Matching ≠ Object-File
3. Hybrid Weighted Fusion Is Not Enough
4. ObjectFile's key metric is conflict resolution, not raw accuracy
5. Conflict-first gate enables structural internalization of identity relations

#### 5.2 Relation-Internalization Interpretation
- ObjectFile as minimal case of relation internalization
- Five relations: feature-identity, trajectory-continuity, occlusion-persistence, conflict-resolution, confidence
- Structure is not a representation, but a set of updatable, adjudicable, transferable relation mechanisms

#### 5.3 The Normal-Performance vs Conflict-Resolution Trade-off
- Systematic trade-off across all mechanisms
- Not a parameter problem, but a structural problem
- Trajectory signal weakness is the root cause

### 6. Limitations (0.5 page)

1. 2D toy world with limited dynamics
2. Only 2 objects
3. Simplified one-hot features
4. Rule-based conflict gate
5. Weak trajectory predictor
6. Not a demonstration of full object permanence

### 7. Conclusion (0.5 page)

- The contribution is a diagnostic chain, not a solved problem
- Clean feature matching can read out identity, but object-file requires structural adjudication
- Conflict-first gating is a necessary component
- The remaining bottleneck is trajectory-state quality
- Future: stronger trajectory predictors, principled uncertainty models, relation-specific object files

---

## Appendix

### A. Full Numerical Results
### B. Feature Noise and Occlusion Curves
### C. Approach Signal Analysis (v4.3)
### D. Confidence Calibration Details
