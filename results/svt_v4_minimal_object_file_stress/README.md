# SVT-v4: Minimal Object-File Stress Test

## 1. Purpose

v3.6 proved temporal-aligned feature key achieves perfect identity on clean data. v4 adds stress: feature noise, occlusion, and feature-trajectory conflict.

## 2. Mechanisms

1. **FeatureOnly**: assignment = feature cosine similarity (v3.6 temporal-aligned)
2. **TrajectoryOnly**: assignment = nearest predicted position
3. **Hybrid(beta=1.0/2.0)**: assignment = traj_logits + beta * feature_logits
4. **MinimalObjectFile**: rule-based with identity_key, trajectory_state, occlusion handling

## 3. Mechanism Comparison (normal features)

| Mechanism | Swap-Only | Feat Dep | Traj Dep | No-Swap Gap |
|-----------|-----------|----------|----------|-------------|
| FeatureOnly | 1.0000 | 0.4904 | 0.0000 | nan |
| TrajectoryOnly | 0.1346 | 0.0000 | 0.1346 | nan |
| Hybrid_b1.0 | 1.0000 | 0.4904 | 0.3077 | nan |
| Hybrid_b2.0 | 1.0000 | 0.4904 | 0.2115 | nan |
| ObjectFile | 0.0962 | 0.0192 | 0.0962 | nan |
| ObjFileLearned | 0.4519 | 0.0385 | 0.3942 | nan |

## 4. Feature Noise

| Mechanism | Noise=0.0 | Noise=0.1 | Noise=0.3 | Noise=0.5 |
|-----------|-----------|-----------|-----------|-----------|
| FeatureOnly | 1.0000 | 1.0000 | 0.9712 | 0.7788 |
| TrajectoryOnly | 0.1346 | 0.1346 | 0.1346 | 0.1346 |
| Hybrid_b1.0 | 1.0000 | 1.0000 | 0.9712 | 0.7885 |
| Hybrid_b2.0 | 1.0000 | 1.0000 | 0.9519 | 0.7885 |
| ObjectFile | 0.0962 | 0.0962 | 0.1058 | 0.1058 |

## 5. Occlusion Without Feature

| Mechanism | Occ=0.0 | Occ=0.25 | Occ=0.5 | Occ=0.75 | Occ=1.0 |
|-----------|---------|----------|---------|----------|---------|
| FeatureOnly | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| TrajectoryOnly | 0.1346 | 0.1346 | 0.1346 | 0.1346 | 0.1346 |
| Hybrid_b1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2404 |
| Hybrid_b2.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0096 |
| ObjectFile | 0.0962 | 0.1442 | 0.2404 | 0.3942 | 0.3269 |

## 6. Feature-Trajectory Conflict

| Mechanism | Conflict Type | Identity | Feat Correct | Traj Correct |
|-----------|--------------|----------|-------------|-------------|
| FeatureOnly | feature_wrong_trajectory_correct | 0.0000 | 1.0000 | 0.0000 |
| TrajectoryOnly | feature_wrong_trajectory_correct | 0.9429 | 0.0571 | 0.9429 |
| Hybrid_b1.0 | feature_wrong_trajectory_correct | 0.0000 | 0.9524 | 0.0476 |
| Hybrid_b2.0 | feature_wrong_trajectory_correct | 0.0571 | 0.9429 | 0.0571 |
| ObjectFile | feature_wrong_trajectory_correct | 0.9333 | 0.0667 | 0.9333 |
| ObjFileLearned | feature_wrong_trajectory_correct | 0.7524 | 0.2476 | 0.7524 |

## 7. Answers

### Q1: Is FeatureOnly fragile under feature noise?

FeatureOnly drop at noise=0.3: **0.0288**

NO - FeatureOnly is robust to moderate feature noise.

### Q2: Is TrajectoryOnly misled by feature-trajectory conflict?

NO - TrajectoryOnly ignores features and is not misled.

### Q3: Is Hybrid just a weighted compromise?

NO - Hybrid is not simply a weighted compromise.

### Q4: Is MinimalObjectFile more stable under occlusion/no-feature/conflict?

Occlusion(0.75): ObjectFile vs FeatureOnly: No clear advantage

NO - MinimalObjectFile does not show clear advantages over simpler mechanisms.

### Q5: Final Recommendation

**benchmark_too_easy**
