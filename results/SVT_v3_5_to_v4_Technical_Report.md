# SVT 阶段性技术报告：v3.5 → v4

## 从 Feature Matching 到 Object-File：干净匹配不等于对象持久性

---

## 摘要

本报告覆盖 SVT v3.5 至 v4 的实验进展。核心发现分为三个层次：

1. **Feature similarity head + temporal-aligned pooling 可以在干净条件下实现完美 identity assignment**（v3.6: FeatureOnly normal=1.000, zero=0.000, dep_score=0.490）
2. **但 FeatureOnly 和 Hybrid 在 feature-trajectory conflict 中完全失败**（v4: FeatureOnly 100% 被错误 feature 误导，Hybrid 95% 被误导）
3. **MinimalObjectFile 虽然绝对性能弱，但在 conflict 中体现了正确的结构偏置**（v4: 93.3% 正确忽略错误 feature）

结论：**干净的 feature matching ≠ object-file**。Feature matching 是必要条件，不是充分条件。Object-file 需要额外的冲突解决机制。

---

## 1. 实验脉络

```
v3.5  Feature-Similarity Assignment Head
      → FeatureOnly dep_score=0.385, 但只有 75% 天花板
      
v3.5.1 Feature Pathway Audit
      → 发现 75% 天花板是 temporal pooling 问题
      → Oracle: obs=first 达到 100%, obs=last 为 0%
      
v3.6  Temporal-Aligned Feature Key Fix
      → first-timestep pooling: FeatureOnly 0.75→1.00
      → Feature dependency 确认: dep_score=0.490
      
v4    Minimal Object-File Stress Test
      → Feature noise / Occlusion / Feature-Trajectory Conflict
      → FeatureOnly 在 conflict 中 100% 失败
      → ObjectFile 在 conflict 中 93.3% 正确
```

---

## 2. v3.5: Feature Similarity Head 的突破与天花板

### 2.1 核心结果

| 模型 | Beta | Normal | Shuffled | Zero | Dep Score |
|------|------|--------|----------|------|-----------|
| FeatureOnly | N/A | 0.750 | 0.365 | 0.000 | **0.385** |
| Hybrid | 1.0 | 0.952 | 0.683 | 0.731 | 0.269 |
| Hybrid | 2.0 | 0.981 | 0.683 | 0.721 | 0.298 |

### 2.2 关键发现

- **FeatureOnly 的 zero=0.000 是整个 SVT 系列最干净的 feature dependency 证据**：没有 feature，identity 完全崩溃
- **75% 天花板**：FeatureOnly 的 normal identity 只有 75%，远低于 Hybrid 的 95-98%
- **Hybrid 的 dep_score 有"水分"**：zero feature 下仍有 72-73% identity，说明 trajectory shortcut 仍在贡献

### 2.3 未解决的问题

75% 天花板的根因是什么？是模型能力不足，还是 feature 本身的信息被破坏？

---

## 3. v3.5.1: Feature Pathway Audit — 诊断 75% 天花板

### 3.1 Oracle Check（不训练 encoder，直接用原始 one-hot feature）

| Obs Pooling | Fut Pooling | Swap-Only Identity |
|-------------|-------------|-------------------|
| **first** | any | **1.000** |
| mean | any | 0.750 |
| **last** | any | **0.000** |

### 3.2 诊断结论

**75% 天花板是 temporal pooling 问题，不是模型问题。**

在 swap episode 中，feature 的时序结构如下：
- swap 前：feature 是原始 one-hot（obj0=[1,0], obj1=[0,1]）→ 可区分
- swap 后：feature 翻转（obj0=[0,1], obj1=[1,0]）→ 与 identity 矛盾
- mean pooling：把 swap 前后混合 → 信息被冲淡 → 只有 75%

`obs=first`（取 observed 的第一个时间步）保留了 swap 前的 discriminative feature，达到 100%。
`obs=last` 取了 swap 后的翻转 feature，完全崩溃到 0%。

### 3.3 Hybrid Pathway Decomposition

| Beta | Traj Dep | Feat Dep | Hybrid Dep |
|------|----------|----------|------------|
| 0.0 | 0.000 | **0.385** | 0.000 |
| 1.0 | 0.010 | **0.385** | 0.279 |
| 2.0 | 0.000 | **0.385** | 0.240 |
| 5.0 | 0.000 | **0.385** | 0.202 |

**Trajectory pathway 的 dep_score 始终为 0**。Feature pathway 贡献恒定 0.385。Hybrid 的 dep_score 随 beta 增大反而下降——因为 traj shortcut 越来越强，稀释了 feature 的贡献。

---

## 4. v3.6: Temporal-Aligned Feature Key Fix

### 4.1 修复方法

将 `_pool_time` 从 `mean(dim=1)` 改为 `[:, 0, :, :]`（first-timestep pooling）。

### 4.2 修复效果

| 指标 | v3.5 (mean pool) | v3.6 (first pool) | 变化 |
|------|-----------------|-------------------|------|
| FeatureOnly normal | 0.750 | **1.000** | +0.250 |
| FeatureOnly shuffled | 0.365 | 0.510 | +0.145 |
| FeatureOnly zero | 0.000 | 0.000 | 不变 |
| FeatureOnly dep_score | 0.385 | **0.490** | +0.105 |

### 4.3 Feature Ablation（v3.6）

| 模型 | Normal | Shuffled | Zero | Wrong |
|------|--------|----------|------|-------|
| FeatureOnly | **1.000** | 0.510 | 0.000 | 0.125 |
| Hybrid β=1.0 | 0.990 | 0.538 | 0.433 | 0.337 |
| Hybrid β=2.0 | **1.000** | 0.510 | 0.154 | 0.125 |

### 4.4 Pathway Decomposition（v3.6）

| 模型 | Traj Dep | Feat Dep | Hybrid Dep |
|------|----------|----------|------------|
| Hybrid β=1.0 | 0.000 | **0.490** | 0.452 |
| Hybrid β=2.0 | 0.000 | **0.490** | 0.490 |

**Trajectory pathway 的 dep_score 仍为 0**。FeatureOnly 达到 100% identity，feature dependency 完全确认。

### 4.5 v3.6 的局限

v3.6 是一个 **positive control**：在干净 feature 条件下，temporal-aligned feature matching 完美解决 identity。但它没有回答：**当 feature 不可靠时会发生什么？**

---

## 5. v4: Minimal Object-File Stress Test

### 5.1 四种机制

| 机制 | Identity 来源 | 特点 |
|------|-------------|------|
| FeatureOnly | feature cosine similarity | 纯 feature，不用 trajectory |
| TrajectoryOnly | nearest predicted position | 纯 trajectory，不用 feature |
| Hybrid β=1.0/2.0 | traj_logits + β×feat_logits | 加权组合 |
| MinimalObjectFile | feature_key + trajectory_state + occlusion handling | 规则系统，有冲突解决 |

### 5.2 Baseline 比较（正常条件）

| 机制 | Swap-Only | Feat Dep | Traj Dep |
|------|-----------|----------|----------|
| **FeatureOnly** | **1.000** | **0.490** | 0.000 |
| TrajectoryOnly | 0.135 | 0.000 | 0.135 |
| Hybrid β=1.0 | **1.000** | **0.490** | 0.308 |
| Hybrid β=2.0 | **1.000** | **0.490** | 0.212 |
| ObjectFile | 0.096 | 0.019 | 0.096 |
| ObjFileLearned | 0.452 | 0.038 | 0.394 |

**正常条件下，FeatureOnly 和 Hybrid 完胜 ObjectFile。**

### 5.3 Stress Test A: Feature Noise

| 机制 | Noise=0.0 | Noise=0.1 | Noise=0.3 | Noise=0.5 |
|------|-----------|-----------|-----------|-----------|
| FeatureOnly | 1.000 | 1.000 | 0.971 | 0.779 |
| TrajectoryOnly | 0.135 | 0.135 | 0.135 | 0.135 |
| Hybrid β=1.0 | 1.000 | 1.000 | 0.971 | 0.789 |
| ObjectFile | 0.096 | 0.096 | 0.106 | 0.106 |
| ObjFileLearned | 0.452 | 0.442 | 0.433 | 0.442 |

FeatureOnly 对 moderate noise (≤0.3) 鲁棒，对 high noise (0.5) 下降 22%。ObjectFile 不受 feature noise 影响（因为它本来就不怎么依赖 feature）。

### 5.4 Stress Test B: Occlusion Without Feature

| 机制 | Occ=0.0 | Occ=0.5 | Occ=0.75 | Occ=1.0 |
|------|---------|---------|----------|---------|
| FeatureOnly | 1.000 | 1.000 | 1.000 | **0.000** |
| TrajectoryOnly | 0.135 | 0.135 | 0.135 | 0.135 |
| Hybrid β=1.0 | 1.000 | 1.000 | 1.000 | 0.240 |
| ObjectFile | 0.096 | 0.240 | 0.394 | **0.327** |
| ObjFileLearned | 0.452 | 0.452 | 0.452 | **0.394** |

**FeatureOnly 在 occ=1.0 时崩溃到 0%。ObjectFile 仍保持 32.7%。**

ObjectFile 在 occlusion 下确实更稳——但绝对值太低。ObjFileLearned（使用训练好的 trajectory predictor）提升到 39.4%。

### 5.5 Stress Test C: Feature-Trajectory Conflict（核心实验）

**实验设计**：no-swap episodes，但翻转 future features（feature 说"对象交换了"，trajectory 说"没交换"）。

| 机制 | Identity Acc | Feature 误导率 | Trajectory 正确率 |
|------|-------------|---------------|------------------|
| **FeatureOnly** | 0.000 | **1.000** | 0.000 |
| TrajectoryOnly | 0.943 | 0.057 | **0.943** |
| Hybrid β=1.0 | 0.000 | **0.952** | 0.048 |
| Hybrid β=2.0 | 0.057 | **0.943** | 0.057 |
| **ObjectFile** | **0.933** | 0.067 | **0.933** |
| ObjFileLearned | 0.752 | 0.248 | 0.752 |

### 5.6 Conflict 结果的深度解读

**FeatureOnly：100% 被错误 feature 误导。**

这是 v3.6 "完美" 结果的反面。FeatureOnly 的 identity 完全由 feature similarity 决定，当 feature 被篡改时，它没有任何纠错机制。v3.6 的 100% identity 是建立在 "feature 永远正确" 的假设上的。

**Hybrid：95% 被错误 feature 误导。**

Hybrid 不是折中——是 feature 绑架。Learned feature logits 的量级远大于 trajectory logits，因为 feature encoder 学到了非常强的 feature-to-identity 映射。Trajectory pathway 在 feature signal 强大时"偷懒"（v3.6 pathway decomposition 显示 traj head 只有 8.65%），导致 conflict 时 trajectory 完全无法制衡 feature。

**ObjectFile：93.3% 正确忽略错误 feature。**

ObjectFile 同时计算 feature similarity score 和 trajectory distance score。当 feature 说"交换了"但 trajectory 说"没交换"时，trajectory 的距离分数更强（因为实际位置确实没变），所以 ObjectFile 正确选择了 trajectory。

**这不是因为 ObjectFile 更聪明，而是因为它的结构偏置更合理**：
1. 它同时维护 feature key 和 trajectory state
2. 它用物理距离（而非 learned logits）做 trajectory matching
3. 当两者冲突时，物理距离比 learned similarity 更可靠

---

## 6. 核心论点：干净 Feature Matching ≠ Object-File

### 6.1 Feature Matching 的成功条件

v3.6 证明 feature matching 在以下条件下完美：
- Feature 干净（无噪声、无篡改）
- Feature 与 identity 一致（无 temporal misalignment）
- 不需要 occlusion 处理

这些条件在真实场景中几乎不可能同时满足。

### 6.2 Feature Matching 的失败模式

| 失败模式 | FeatureOnly | Hybrid | ObjectFile |
|----------|-------------|--------|------------|
| Feature 噪声 (σ=0.5) | 0.779 (-22%) | 0.789 (-21%) | 0.106 (不变) |
| Feature 完全缺失 | 0.000 | 0.240 | 0.327 |
| Feature 被篡改 | **0.000** | **0.000** | **0.933** |

FeatureOnly 和 Hybrid 在 feature 被篡改时完全失败。ObjectFile 在篡改下反而最稳。

### 6.3 Object-File 的结构性优势

ObjectFile 的优势不是来自更强的性能，而是来自更合理的架构偏置：

1. **双通道维护**：同时维护 identity_key（feature）和 trajectory_state（位置/速度）
2. **Occlusion 韧性**：遮挡时保持 identity_key 不重写，重现时才更新
3. **冲突解决**：feature 和 trajectory 冲突时，物理距离比 learned similarity 更可靠
4. **不依赖训练**：规则系统，不会被训练数据的分布偏置误导

### 6.4 Object-File 的当前弱点

1. **Trajectory prediction 太弱**：简单线性预测在 vortex 场下只有 9.6% identity
2. **Feature weight 太低**：feat_dep=0.019，几乎没有利用 feature 信息
3. **没有 learned components**：无法从数据中学习更好的匹配策略

ObjFileLearned（使用训练好的 trajectory predictor）将 identity 提升到 45.2%，但仍远低于 FeatureOnly 的 100%。

---

## 7. 从 v3.6 到 v4 的认知跃迁

```
v3.6 的认知：Feature matching 可以完美解决 identity（在干净条件下）
v4 的认知：Feature matching 的"完美"是脆弱的——它假设 feature 永远正确

v3.6 的问题：只测了 feature 可靠的情况
v4 的问题：暴露了 feature 不可靠时的灾难性失败

v3.6 的 FeatureOnly = 优秀的 feature reader
v4 的 ObjectFile = 初步的 object-file keeper

Feature reader ≠ Object-file keeper
```

### 7.1 类比

| | Feature Reader | Object-File Keeper |
|---|---|---|
| 核心能力 | 读取当前 feature 并匹配 | 维护跨时间的对象身份 |
| 成功条件 | Feature 可靠 | 任何条件下 |
| 失败模式 | Feature 被篡改时 100% 失败 | Trajectory 弱时性能低但不崩溃 |
| 类比 | 人脸识别 | 人脸识别 + 跟踪 + 记忆 |

---

## 8. 下一步方向

### 8.1 不应做的

- ❌ 继续调 beta / lambda——v4 已证明 Hybrid 的加权组合无法解决 conflict
- ❌ 声称 object permanence solved——ObjectFile 绝对性能太低
- ❌ 增加大模型——问题不在模型容量，而在架构偏置

### 8.2 应该做的

1. **改进 ObjectFile 的 trajectory prediction**：ObjFileLearned 从 9.6% → 45.2%，说明 learned trajectory 是有效路径
2. **自适应 feature-trajectory 权重**：当 feature 和 trajectory 一致时信任 feature，冲突时降权 feature
3. **置信度校准**：在 conflict 时输出低置信度，而不是盲目选择
4. **Occlusion-aware feature update**：遮挡时冻结 identity_key，重现时验证

### 8.3 推荐路线

**`improve_object_file_update_rule`**

ObjectFile 在 conflict resolution 上的 93.3% 正确率证明了架构偏置的正确性。下一步应聚焦于：
- 将 FeatureOnly 的 100% feature matching 能力嵌入 ObjectFile 的 feature channel
- 将 learned trajectory predictor 嵌入 ObjectFile 的 trajectory channel
- 设计自适应权重机制，让 ObjectFile 在 feature 可靠时信任 feature，不可靠时回退到 trajectory

---

## 9. 数据溯源

| 版本 | 结果目录 |
|------|---------|
| v3.5 | `results/svt_v3_5_feature_similarity_head/` |
| v3.5.1 | `results/svt_v3_5_1_feature_pathway_audit/` |
| v3.6 | `results/svt_v3_6_temporal_aligned_feature_key/` |
| v4 | `results/svt_v4_minimal_object_file_stress/` |

| 代码文件 | 说明 |
|---------|------|
| `models/feature_similarity_models.py` | FeatureOnly + Hybrid 模型 |
| `models/object_file_models.py` | TrajectoryOnly + ObjectFile + ObjFileLearned |
| `scripts/run_svt_v3_5_feature_similarity_head.py` | v3.5 实验 |
| `scripts/run_svt_v3_5_1_feature_pathway_audit.py` | v3.5.1 诊断 |
| `scripts/run_svt_v3_6_temporal_aligned_feature_key.py` | v3.6 修复 |
| `scripts/run_svt_v4_minimal_object_file_stress.py` | v4 压力测试 |
