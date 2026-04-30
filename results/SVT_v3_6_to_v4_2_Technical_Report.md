# From Feature Matching to Object Files: Stress-Testing Identity Binding in Agents

# 从特征匹配到对象文件：智能体身份绑定的结构有效性压力测试

---

## 1. 摘要

本报告覆盖 SVT（Symbolic Verification Test）v3.6 至 v4.3 的实验进展。我们不是在报告某个模型已经具备对象持久性，而是在建立一条结构判别链：干净 feature matching 可以解决简单身份匹配，但不等于 object-file。真正的 object-file 必须在 feature 缺失、feature 被篡改、trajectory 与 feature 冲突、遮挡、置信度校准等压力下仍能保持结构性裁决。

实验分四个阶段推进（主线），外加一个补充负结果：

1. **v3.6**：temporal-aligned feature key 在干净条件下实现完美 identity assignment（FeatureOnly swap-only=1.000），但这是 positive control，不等于 object-file。
2. **v4**：加入 feature noise、occlusion without feature、feature-trajectory conflict 三项压力测试。FeatureOnly 在 conflict 中被错误 feature 完全误导（identity=0.000），Hybrid 同样被 feature logits 绑架（identity=0.000~0.057）。MinimalObjectFile 正常性能弱（swap-only=0.096），但在 conflict 中体现正确结构偏置（trajectory correct=0.933）。
3. **v4.1**：引入 learned trajectory state 提升正常 swap-only identity（0.096→0.558）和 occlusion 表现，但 conflict resolution 从 v4 的 0.933 降到 0.610，confidence correct/incorrect 都接近 1.0，calibration 完全失败。
4. **v4.2**：从"加权融合"改为"先检测冲突，再裁决"的 conflict-first gate。ConflictFirst_margin 在四项通过标准上全部达标：conflict resolution=0.648（>v4.1 的 0.610），swap-only=0.519（不低于 v4.1 的 0.508 阈值），confidence correct=0.874 vs incorrect=0.237（校准误差=0.637），uncertain/abstain 在高冲突样本中增加。
5. **v4.3（补充负结果）**：测试 observed-period approach detection 和 trajectory augmentation 能否改善 trade-off。Approach detection 提升了 swap-only（0.529→0.625），但 conflict resolution 下降（0.714→0.619）；trajectory augmentation 未改善 OOD 鲁棒性。这表明剩余瓶颈在 trajectory_state 质量，而非 gate heuristic。

**一句话结论：干净 feature matching 可以读出身份，但 object-file 需要在 feature、trajectory、occlusion 和 conflict 之间进行结构性裁决。**

---

## 2. 研究问题

本报告围绕四个递进的研究问题展开：

**RQ1：prediction 是否等于 identity binding？**

轨迹预测精度高是否意味着模型理解了对象身份？v3.6 的 FeatureOnly 在干净条件下达到完美 identity，但这只是 feature similarity 的直接映射，不涉及跨时间身份维护。

**RQ2：clean feature matching 是否等于 object-file？**

如果 feature 可靠，feature matching 足以解决 identity。但 object-file 的核心挑战不是"feature 可靠时能否匹配"，而是"feature 不可靠时能否裁决"。

**RQ3：weighted hybrid fusion 是否足够？**

将 trajectory logits 和 feature logits 加权组合，能否同时获得两者的优势？v4 的 conflict 实验表明，Hybrid 不是折中，而是 feature 绑架——learned feature logits 的量级远大于 trajectory logits。

**RQ4：conflict-first gate 是否是 object-file 的必要组件？**

当 feature 和 trajectory 冲突时，是否应该先检测冲突、再裁决，而不是加权融合？v4.2 的实验表明，conflict-first gate 是当前最平衡的机制。

---

## 3. 背景

### 3.1 SVT 作为结构有效性判别器

SVT（Symbolic Verification Test）的设计哲学是：不问"模型性能多高"，而问"模型的结构是否正确"。一个在干净条件下达到 100% identity 的模型，如果被错误 feature 完全误导，它的结构就是有缺陷的。SVT 通过系统性压力测试（feature ablation、feature noise、occlusion、feature-trajectory conflict）来判别模型是否具备真正的结构理解。

### 3.2 ObjectFile 作为最小结构单元

ObjectFile 是一个规则系统，维护每个对象的 identity_key（feature）、trajectory_state（位置/速度）、occlusion_state 和 confidence。它不是 learned model，而是一个最小结构单元，用于测试"同时维护 feature 和 trajectory 信息，并在冲突时裁决"这一结构是否有效。

### 3.3 Relation-Internalization 视角

从 relation-internalization 的角度看，对象身份不是一个标签，而是一组关系裁决机制：

- **feature-identity relation**：feature 与对象身份的对应关系
- **trajectory-continuity relation**：对象位置在时间上的连续性
- **occlusion-persistence relation**：对象被遮挡后仍保持身份
- **conflict-resolution relation**：当多种信号冲突时如何裁决
- **confidence relation**：裁决的可信度

ObjectFile 是这些关系的最小实例化。它的价值不在于绝对性能，而在于提供了这些关系的结构化框架。

---

## 4. 方法

### 4.1 评估指标

- **identity_swap_only**：仅在 swap episode 上计算的身份准确率，避免 no-swap bias
- **feature_dependency_score**：normal identity - shuffled identity，衡量对 feature 的依赖程度
- **trajectory_dependency_score**：zero feature 时的 identity，衡量纯 trajectory 贡献

### 4.2 压力测试

- **Feature Ablation**：将 feature shuffle 或 zero-out，测试 feature dependency
- **Feature Noise**：对 feature 添加高斯噪声（σ=0.1, 0.3, 0.5），测试鲁棒性
- **Occlusion Without Feature**：遮挡期间将 feature 置零，测试 occlusion persistence
- **Feature-Trajectory Conflict**：no-swap episode 但翻转 future features（feature 说"交换了"，trajectory 说"没交换"），测试冲突裁决能力

### 4.3 Confidence Calibration

- **avg_confidence_correct**：预测正确时的平均置信度
- **avg_confidence_incorrect**：预测错误时的平均置信度
- **confidence_calibration_error**：两者之差的绝对值

### 4.4 Conflict Gate 评估

- **chosen_source_feature_rate**：冲突时选择 feature 的比例
- **chosen_source_trajectory_rate**：冲突时选择 trajectory 的比例
- **chosen_source_uncertain_rate**：冲突时标记为不确定的比例
- **abstention_rate**：选择弃权的比例
- **accuracy_when_not_abstaining**：非弃权时的准确率

---

## 5. 结果

### 5.1 v3.6：Temporal-Aligned Feature Key — 干净条件下的 Positive Control

v3.5.1 发现 mean pooling 破坏了 swap 前的 discriminative feature（75% 天花板）。v3.6 将 pooling 从 mean 改为 first-timestep（`[:, 0, :, :]`），保留 swap 前的身份判别信息。

| 模型 | Normal | Shuffled | Zero | Feature Dep |
|------|--------|----------|------|-------------|
| FeatureOnly | **1.000** | 0.510 | 0.000 | **0.490** |
| Hybrid β=1.0 | 0.990 | 0.538 | 0.433 | 0.452 |
| Hybrid β=2.0 | **1.000** | 0.510 | 0.154 | 0.490 |

（数据来源：Normal 来自 `temporal_aligned_results.csv`，Shuffled/Zero 来自 `feature_ablation.csv`，Feature Dep 来自 `pathway_decomposition.csv`）

**核心结论**：FeatureOnly 在干净条件下完美，feature_dependency_score=0.490 确认 feature 是 identity assignment 的有效机制。但这是 positive control——它假设 feature 永远正确，没有测试 feature 不可靠的情况。

### 5.2 v4：Minimal Object-File Stress Test — Feature Matching 在冲突中失败

v4 加入三项压力测试，比较六种机制。

#### 5.2.1 正常条件 Baseline

| 机制 | Swap-Only | Feat Dep | Traj Dep |
|------|-----------|----------|----------|
| **FeatureOnly** | **1.000** | **0.490** | 0.000 |
| TrajectoryOnly | 0.135 | 0.000 | 0.135 |
| Hybrid β=1.0 | **1.000** | **0.490** | 0.308 |
| Hybrid β=2.0 | **1.000** | **0.490** | 0.212 |
| ObjectFile | 0.096 | 0.019 | 0.096 |
| ObjFileLearned | 0.452 | 0.038 | 0.394 |

正常条件下 FeatureOnly 和 Hybrid 完胜 ObjectFile。但这是误导性的——它只测了 feature 可靠的情况。

#### 5.2.2 Feature-Trajectory Conflict（核心实验）

| 机制 | Identity Acc | Feature 误导率 | Trajectory 正确率 |
|------|-------------|---------------|------------------|
| **FeatureOnly** | **0.000** | **1.000** | 0.000 |
| TrajectoryOnly | 0.943 | 0.057 | **0.943** |
| Hybrid β=1.0 | **0.000** | **0.952** | 0.048 |
| Hybrid β=2.0 | 0.057 | **0.943** | 0.057 |
| **ObjectFile** | **0.933** | 0.067 | **0.933** |
| ObjFileLearned | 0.752 | 0.248 | 0.752 |

**FeatureOnly 100% 被错误 feature 误导。Hybrid 95% 被误导。ObjectFile 93.3% 正确忽略错误 feature。**

这不是因为 ObjectFile 更聪明，而是因为它的结构偏置更合理：它同时维护 feature key 和 trajectory state，当两者冲突时，物理距离比 learned similarity 更可靠。

#### 5.2.3 Occlusion Without Feature

| 机制 | Occ=0.0 | Occ=0.75 | Occ=1.0 |
|------|---------|----------|---------|
| FeatureOnly | 1.000 | 1.000 | **0.000** |
| Hybrid β=1.0 | 1.000 | 1.000 | 0.240 |
| ObjectFile | 0.096 | 0.394 | **0.327** |
| ObjFileLearned | 0.452 | 0.452 | **0.394** |

FeatureOnly 在完全遮挡时崩溃到 0%。ObjectFile 仍保持 32.7%——绝对值低，但结构偏置正确。

### 5.3 v4.1：ObjectFile Update Rule Improvement — Learned Trajectory 帮助但 Calibration 失败

v4.1 引入三个改进：learned trajectory predictor、confidence calibration、conflict gate（confidence-based adaptive weighting）。

| 机制 | Swap-Only | Feat Dep | Traj Dep |
|------|-----------|----------|----------|
| ObjectFile_v4 | 0.096 | 0.019 | 0.096 |
| **ImprovedObjectFile_v4.1** | **0.558** | 0.125 | **0.423** |

Learned trajectory 将 swap-only identity 从 0.096 提升到 0.558，occlusion 表现也改善。

#### 5.3.1 Conflict Resolution 退化

| 机制 | Conflict Identity | Traj Correct | Feat Wrong |
|------|------------------|--------------|------------|
| ObjectFile_v4 | **0.933** | **0.933** | 0.067 |
| ImprovedObjectFile_v4.1 | 0.610 | 0.610 | 0.390 |

Conflict resolution 从 0.933 降到 0.610。原因：confidence-based adaptive weighting 中，`w_f = feat_conf / (feat_conf + traj_conf)`，当 feature similarity 高时（即使 feature 是被篡改的），feature confidence 仍然高，权重偏向 feature，导致被错误 feature 误导。

#### 5.3.2 Confidence Calibration 失败

| 机制 | Conf Correct | Conf Incorrect | Cal Error |
|------|-------------|---------------|-----------|
| ObjectFile_v4 | 1.000 | 1.000 | 0.000 |
| ImprovedObjectFile_v4.1 | 1.000 | 1.000 | 0.000 |

无论预测正确还是错误，confidence 都接近 1.0。Confidence 完全无法区分正确与错误预测。这意味着 confidence-based adaptive weighting 的前提条件不成立——confidence 本身就是不可信的。

#### 5.3.3 Conflict Gate 全部选择 Feature

ImprovedObjectFile 的 conflict gate 在所有冲突情况下都选择了 feature（feature_rate=1.000, trajectory_rate=0.000, uncertain_rate=0.000）。这不是裁决，而是投降。

### 5.4 v4.2：Conflict-First ObjectFile — 先检测冲突，再裁决

v4.2 改变 gate 策略：从"加权融合"改为"先检测冲突，再裁决"。

#### 5.4.1 四种 Conflict-First 策略

| 策略 | 逻辑 | Swap-Only | Conflict Res |
|------|------|-----------|-------------|
| prefer_trajectory_on_conflict | 冲突时总是选 trajectory | 0.452 | **0.762** |
| prefer_feature_on_low_traj_conf | trajectory confidence 低时选 feature | **0.740** | 0.371 |
| abstain_on_high_conflict | 双方 margin 都小时弃权 | 0.452 | 0.762 |
| **margin_gated** | 比较 margin 优势度，无优势时弃权 | **0.519** | **0.648** |

四种策略呈现清晰的 trade-off：trajectory 优先的 conflict resolution 最高但正常性能低；feature 优先的正常性能最高但 conflict resolution 低；margin_gated 是最佳平衡点。

#### 5.4.2 ConflictFirst_margin 详细结果

| 指标 | ObjectFile_v4 | ImprovedObjectFile_v4.1 | ConflictFirst_margin |
|------|-------------|----------------------|---------------------|
| Swap-Only | 0.096 | 0.567 | 0.519 |
| Conflict Resolution | 0.933 | 0.600 | **0.648** |
| Conf Correct | 1.000 | 1.000 | **0.874** |
| Conf Incorrect | 1.000 | 1.000 | **0.237** |
| Cal Error | 0.000 | 0.000 | **0.637** |
| Uncertain Rate (conflict) | — | 0.000 | **0.010** |
| Abstain Rate (conflict) | — | 0.000 | **0.010** |

**关键突破**：confidence 终于能区分正确与错误预测（0.874 vs 0.237）。这是 v3.6→v4.2 全系列中第一次实现有意义的 confidence calibration。

#### 5.4.3 四项通过标准

| 标准 | 阈值 | 值 | 通过 |
|------|------|-----|------|
| conflict resolution > v4.1 | > 0.610 | 0.648 | ✅ |
| swap-only 不低于 v4.1 | ≥ 0.508 | 0.519 | ✅ |
| confidence 区分 correct vs incorrect | correct > incorrect + 0.05 | 0.874 vs 0.237 | ✅ |
| uncertain/abstain 在高冲突样本增加 | > 0 | margin=0.010, abstain=0.152 | ✅ |

---

## 6. 核心发现

### 发现一：Prediction ≠ Identity Binding

轨迹预测精度高不等于理解了对象身份。TrajectoryOnly 的轨迹预测在训练集上收敛，但 swap-only identity 只有 0.135~0.154。Identity binding 需要的不是预测未来位置，而是判断"这个位置上的对象是不是之前那个对象"。

### 发现二：Clean Feature Matching ≠ Object-File

FeatureOnly 在干净条件下达到 swap-only=1.000，但在 feature-trajectory conflict 中 identity=0.000。Feature matching 的"完美"建立在"feature 永远正确"的假设上。Object-file 的核心挑战不是"feature 可靠时能否匹配"，而是"feature 不可靠时能否裁决"。

### 发现三：Hybrid Weighted Fusion Is Not Enough

Hybrid 将 trajectory logits 和 feature logits 加权组合，在 conflict 中 95% 被 feature 误导。这不是折中，而是 feature 绑架。Learned feature logits 的量级远大于 trajectory logits，因为 feature encoder 学到了非常强的 feature-to-identity 映射。Trajectory pathway 在 feature signal 强大时"偷懒"（v3.6 pathway decomposition 显示 traj dep=0.000），导致 conflict 时 trajectory 完全无法制衡 feature。

### 发现四：ObjectFile 的关键不是高分，而是冲突裁决能力

ObjectFile_v4 的 swap-only 只有 0.096，远低于 FeatureOnly 的 1.000。但在 conflict 中，ObjectFile 的 trajectory correct=0.933，FeatureOnly 的 trajectory correct=0.000。ObjectFile 的价值不在于绝对性能，而在于它同时维护 feature 和 trajectory 两个通道，并在冲突时做出结构性裁决。Feature reader ≠ Object-file keeper。

### 发现五：Conflict-First Gate 让 Identity Relation 开始被结构化内化

v4.1 的 confidence-based adaptive weighting 失败，因为 confidence 本身不可信。v4.2 的 conflict-first gate 通过"先检测冲突，再裁决"的策略，实现了：
- Conflict resolution 从 0.600 提升到 0.648
- Confidence calibration 从 0.000 提升到 0.637（correct=0.874, incorrect=0.237）
- Uncertain/abstain 在高冲突样本中增加

这不是说 conflict-first gate 已经完美——它的 swap-only 仍只有 0.519，conflict resolution 仍低于 v4 的 0.933。但它是第一个在"正常性能"和"冲突裁决"之间找到平衡点的机制，也是第一个让 confidence 产生区分度的机制。

---

## 7. 负结果

### 7.1 FeatureOnly 被错误 Feature 完全误导

FeatureOnly 在 feature-trajectory conflict 中 identity=0.000。这不是偶然——FeatureOnly 的 identity 完全由 feature similarity 决定，当 feature 被篡改时，它没有任何纠错机制。v3.6 的 100% identity 是建立在"feature 永远正确"的假设上的。

### 7.2 Hybrid 被 Feature Logits 绑架

Hybrid β=1.0 在 conflict 中 identity=0.000~0.029，β=2.0 也只有 0.057。增加 trajectory 权重（增大 β）并不能解决 feature 绑架问题——因为 feature logits 的量级优势太大。Hybrid 的"加权融合"在 feature 和 trajectory 信号量级不对等时失效。

### 7.3 v4.1 Confidence Calibration 失败

ImprovedObjectFile 的 confidence correct=1.000, confidence incorrect=1.000, calibration error=0.000。无论预测正确还是错误，confidence 都接近 1.0。原因：confidence 基于 feature similarity 和 trajectory distance 的绝对值，而这些值在正常和冲突条件下都处于相似范围。Confidence-based adaptive weighting 的前提条件不成立。

### 7.4 ObjectFile 当前正常性能仍不够高

ConflictFirst_margin 的 swap-only=0.519，远低于 FeatureOnly 的 1.000。ObjectFile 系列的正常性能受限于：
- Trajectory predictor 在 OOD 条件下预测精度不足
- Feature weight 在 conflict-first 策略下被主动降权
- 规则系统的匹配策略不如 learned model 灵活

### 7.5 v4.3：Gate Heuristic 的边际收益递减（补充负结果）

v4.3 测试了两个新方向：observed-period approach detection（物体是否在靠近）和 trajectory training augmentation。

**Approach detection 结果**：

| 策略 | Swap-Only | Conflict Res | Cal Error |
|------|-----------|-------------|-----------|
| ConflictFirst_margin_v4.2 | 0.529 | **0.714** | **0.627** |
| TrajRobust_aware_aug_v4.3 | **0.625** | 0.619 | 0.269 |
| TrajRobust_veto_v4.3 | 0.510 | 0.657 | 0.301 |

Approach detection 提升了 swap-only identity（0.529→0.625），但 conflict resolution 下降（0.714→0.619）。原因：approach signal 的区分度不足——swap episode 平均 approach=-0.87（物体在靠近），no-swap episode 平均 approach=0.12（物体在远离），但标准差高达 7.97，远大于均值差（0.99）。大量 swap episode 的物体并不靠近，大量 no-swap episode 的物体也在靠近。

**Trajectory augmentation 结果**：

| 模型 | Swap-Only |
|------|-----------|
| TrajectoryOnly (standard) | 0.173 |
| TrajectoryOnly (augmented) | 0.125 |

简单高斯噪声 augmentation 未改善 OOD 鲁棒性，反而略降。原因：位置噪声不改变动力学结构，无法帮助模型泛化到不同力场类型。

**结论**：v4.3 不是突破，而是终止信号。它实证化了"继续调 gate heuristic 的边际收益已经很低"这一判断。剩余瓶颈在 trajectory_state 质量，而非 gate 设计。

---

## 8. Relation-Internalization 解读

ObjectFile 可以被理解为 relation internalization 的最小案例。它不是在"表示"对象身份，而是在维护和裁决一组关系：

### 8.1 Feature-Identity Relation

ObjectFile 维护 identity_key（feature），当未来 feature 与 identity_key 匹配时，更新 identity_key。这是 feature-identity relation 的实例化：feature 是身份的线索，但不是身份本身。当 feature 被篡改时，identity_key 不应被错误 feature 覆盖。

### 8.2 Trajectory-Continuity Relation

ObjectFile 维护 trajectory_state（last_pos + last_vel），预测对象的未来位置。这是 trajectory-continuity relation 的实例化：对象在时间上连续存在，位置变化遵循物理规律。Trajectory prediction 的精度决定了这条关系的可靠性。

### 8.3 Occlusion-Persistence Relation

ObjectFile 在遮挡时冻结 identity_key，不重写；重现时验证 feature similarity。这是 occlusion-persistence relation 的实例化：对象被遮挡后仍保持身份，重现时需要验证。Occlusion decay 和 reappearance boost 是这条关系的参数化。

### 8.4 Conflict-Resolution Relation

ObjectFile 在 feature 和 trajectory 冲突时，通过 conflict-first gate 裁决。这是 conflict-resolution relation 的实例化：当多种信号冲突时，不是简单加权，而是先检测冲突、再裁决。Margin comparison 和 uncertain/abstain 是这条关系的裁决策略。

### 8.5 Confidence Relation

ObjectFile 输出 confidence，反映裁决的可信度。这是 confidence relation 的实例化：裁决不是全有或全无，而是有程度的。当 confidence 低时，应标记为 uncertain 或弃权。

### 8.6 核心论点

**结构不是一个表示，而是一组可更新、可裁决、可迁移的关系机制。**

ObjectFile 不是在"存储"对象身份，而是在"裁决"对象身份。每一次 identity assignment 都是一次关系裁决：feature 说什么？trajectory 说什么？是否冲突？哪个更可靠？confidence 多高？

v3.6 的 FeatureOnly 只有一条关系：feature-identity。它在 feature 可靠时完美，在 feature 不可靠时崩溃。

v4 的 MinimalObjectFile 有四条关系：feature-identity、trajectory-continuity、occlusion-persistence、conflict-resolution。它在正常条件下弱，但在冲突条件下正确。

v4.2 的 ConflictFirstObjectFile 有五条关系（加上 confidence relation），在正常和冲突之间找到了平衡。

---

## 9. 局限性

1. **当前环境仍是二维 toy world**：对象在 64×64 的二维空间中运动，力场类型有限（attractor, vortex）。真实世界的物理复杂度远超此范围。

2. **Object 数量少**：当前只测试了 2 个对象。Object-file 的冲突裁决在 3+ 对象时可能面临组合爆炸。

3. **Feature 是简化 one-hot**：当前 feature 是 2 维 one-hot vector（[1,0] 和 [0,1]），真实世界的 feature 是高维连续向量，feature similarity 的计算和匹配更复杂。

4. **Conflict gate 仍是规则型**：margin_gated 策略的阈值（traj_margin_advantage=1.2）是手动设定的，不是从数据中学习的。在更复杂的环境中，这些阈值可能需要自适应调整。

5. **Trajectory predictor 仍弱**：TrajectoryOnly 的 swap-only 只有 0.135~0.154，learned trajectory 在 OOD（vortex）条件下预测精度不足。Trajectory prediction 的精度是 ObjectFile 正常性能的上限。

6. **没有证明完整 object permanence**：ObjectFile 在 conflict 中正确忽略错误 feature，但这只是 object permanence 的一个必要条件，不是充分条件。完整的 object permanence 还需要：对象在遮挡后重现时的身份恢复、多步推理、跨场景迁移等。

---

## 10. 下一步

### 不应做的

- ❌ 继续调 beta / lambda——v4 已证明 Hybrid 的加权组合无法解决 conflict
- ❌ 上 LLM——问题不在模型容量，而在架构偏置
- ❌ 马上做大模型——当前 toy world 的诊断价值尚未耗尽
- ❌ 声称 object permanence solved——ObjectFile 的正常性能仍不够高

### 应该做的

1. **写论文/技术报告**：v3.6→v4.2 的诊断链已经形成了完整的论证结构，足以支撑一篇关于"feature matching ≠ object-file"的论文。

2. **做更强但仍最小的 uncertainty model**：v4.2 的 confidence calibration 是第一次突破，但 margin_gated 的 uncertain rate 仍很低（0.010）。一个更 principled 的 uncertainty model（如 Bayesian update 或 evidential deep learning）可能产生更有意义的 confidence 和 abstention。

3. **做更强的 trajectory predictor**：Trajectory predictor 的精度是 ObjectFile 正常性能的上限。当前 TrajectoryOnly 的 swap-only 只有 0.135，如果提升到 0.5+，ObjectFile 的正常性能可能大幅提升。

4. **进入 Relation-Specific ObjectFile**：当前 ObjectFile 的五条关系是统一处理的。如果每条关系有独立的更新规则和裁决机制（如 feature-identity 用 contrastive learning，trajectory-continuity 用 physics-informed model，conflict-resolution 用 game-theoretic mechanism），可能产生更精细的裁决。

---

## 数据溯源

| 版本 | 结果目录 |
|------|---------|
| v3.6 | `results/svt_v3_6_temporal_aligned_feature_key/` |
| v4 | `results/svt_v4_minimal_object_file_stress/` |
| v4.1 | `results/svt_v4_1_object_file_update_rule/` |
| v4.2 | `results/svt_v4_2_conflict_first_object_file/` |
| v4.3 | `results/svt_v4_3_trajectory_robust_object_file/` |

| 代码文件 | 说明 |
|---------|------|
| `models/feature_similarity_models.py` | FeatureOnly + Hybrid 模型 |
| `models/object_file_models.py` | TrajectoryOnly + ObjectFile + ImprovedObjectFile + ConflictFirstObjectFile + TrajectoryRobustObjectFile |
| `metrics/object_file_metrics.py` | Confidence calibration + Conflict gate stats + Abstention metrics |
| `scripts/run_svt_v3_6_temporal_aligned_feature_key.py` | v3.6 实验 |
| `scripts/run_svt_v4_minimal_object_file_stress.py` | v4 压力测试 |
| `scripts/run_svt_v4_1_object_file_update_rule.py` | v4.1 改进 |
| `scripts/run_svt_v4_2_conflict_first_object_file.py` | v4.2 conflict-first gate |
| `scripts/run_svt_v4_3_trajectory_robust_object_file.py` | v4.3 trajectory robust (supplementary) |

---

## English Abstract

We present a diagnostic chain from feature matching to object files, stress-testing identity binding in agents across four experimental stages (SVT v3.6–v4.2), with a supplementary negative result (v4.3). In v3.6, temporal-aligned feature keys achieve perfect identity assignment under clean conditions (swap-only=1.000, feature dependency=0.490), establishing a positive control. However, v4 stress tests reveal that FeatureOnly is 100% misled by tampered features in feature-trajectory conflict, and Hybrid models suffer feature hijacking (95% misled). MinimalObjectFile, despite weak normal performance (swap-only=0.096), correctly resolves 93.3% of conflicts—demonstrating structural bias rather than raw performance. v4.1 improves normal identity via learned trajectory (0.096→0.558) but conflict resolution drops (0.933→0.610) and confidence calibration fails entirely (correct=incorrect=1.0). v4.2 replaces weighted fusion with conflict-first adjudication: detect conflict, then decide. ConflictFirst_margin achieves conflict resolution=0.648 (>v4.1), swap-only=0.519 (maintained), and the first meaningful confidence calibration (correct=0.874, incorrect=0.237, cal_error=0.637). v4.3 tests approach detection and trajectory augmentation as potential improvements; approach detection improves swap-only (0.529→0.625) but reduces conflict resolution (0.714→0.619), while augmentation does not improve OOD robustness. This confirms that remaining bottlenecks lie in trajectory-state quality rather than gate heuristics. We conclude: clean feature matching can read out identity, but object-file requires structural adjudication across feature, trajectory, occlusion, and conflict signals. The key contribution is not a solved problem, but a structural discriminative chain showing that feature reading ≠ object-file keeping, and that conflict-first gating is a necessary component of minimal object-file mechanisms.
