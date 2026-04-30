# SVT-v3.5: Feature-Similarity Assignment Head Test

## 1. Purpose

v3.4 showed contrastive binding loss fails to create feature dependency. v3.5 changes the architecture: feature similarity directly participates in the assignment decision, rather than being an auxiliary loss.

## 2. Models

### FeatureOnlyAssignmentHead
- assignment_logits = cosine_similarity(z_fut, z_obs) / temperature
- No trajectory-based assignment logits
- Identity decision is purely feature-driven

### HybridTrajectoryFeatureAssignmentHead
- trajectory_logits from learned assignment head
- feature_logits from feature cosine similarity
- assignment_logits = trajectory_logits + beta * feature_logits
- Beta sweep: [0.0, 0.5, 1.0, 2.0, 5.0]

## 3. Results

| Model | Beta | Clean Skill | Swap-Only ID | Assignment Acc |
|-------|------|------------|-------------|---------------|
| FeatureOnlyAssignmentHead | N/A | 0.6772 | 0.7500 | 0.7500 |
| HybridTrajectoryFeatureAssignmentHead | 0.0 | 0.6902 | 0.6346 | 0.6346 |
| HybridTrajectoryFeatureAssignmentHead | 0.5 | 0.6706 | 0.5962 | 0.5962 |
| HybridTrajectoryFeatureAssignmentHead | 1.0 | 0.6666 | 0.9519 | 0.9519 |
| HybridTrajectoryFeatureAssignmentHead | 2.0 | 0.6828 | 0.9808 | 0.9808 |
| HybridTrajectoryFeatureAssignmentHead | 5.0 | 0.6854 | 1.0000 | 1.0000 |

## 4. Feature Dependency

| Model | Beta | Normal | Shuffled | Zero | Wrong | Dep Score | Feature Dep? |
|-------|------|--------|----------|------|-------|-----------|-------------|
| FeatureOnlyAssignmentHead | N/A | 0.7500 | 0.3654 | 0.0000 | 0.2500 | 0.3846 | True |
| HybridTrajectoryFeatureAssignmentHead | 0.0 | 0.6346 | 0.6442 | 0.6154 | 0.6250 | -0.0096 | False |
| HybridTrajectoryFeatureAssignmentHead | 0.5 | 0.5962 | 0.4808 | 0.4423 | 0.3654 | 0.1154 | False |
| HybridTrajectoryFeatureAssignmentHead | 1.0 | 0.9519 | 0.6827 | 0.7308 | 0.6442 | 0.2692 | True |
| HybridTrajectoryFeatureAssignmentHead | 2.0 | 0.9808 | 0.6827 | 0.7212 | 0.5192 | 0.2981 | True |
| HybridTrajectoryFeatureAssignmentHead | 5.0 | 1.0000 | 0.7981 | 1.0000 | 0.7981 | 0.2019 | True |

## 5. Key Findings

- FeatureOnlyAssignmentHead dependency score: **0.3846**
- Best overall dependency score: **0.3846**
- Hybrid beta increases feature dependency: **True**

Feature similarity head creates genuine feature dependency!

normal > shuffled by >0.2, confirming the model uses features for identity.

## 6. Final Recommendation

**feature_similarity_head_required**
