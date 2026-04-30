# SVT-v4.1: ObjectFile Update Rule Improvement

## 1. Purpose

v4 showed MinimalObjectFile has correct structural bias (93.3% conflict resolution) but weak absolute performance (9.6% swap-only). v4.1 improves: learned trajectory, confidence calibration, conflict gate.

## 2. Mechanism Comparison

| Mechanism | Swap-Only | Feat Dep | Traj Dep |
|-----------|-----------|----------|----------|
| FeatureOnly | 1.0000 | 0.4904 | 0.0000 |
| TrajectoryOnly | 0.1442 | 0.0000 | 0.1442 |
| Hybrid_b1.0 | 1.0000 | 0.4904 | 0.1058 |
| ObjectFile_v4 | 0.0962 | 0.0192 | 0.0962 |
| ImprovedObjectFile | 0.5577 | 0.1250 | 0.4231 |

## 3. Confidence Calibration

| Mechanism | Conf Correct | Conf Incorrect | Cal Error | Feat Rate | Traj Rate | Uncertain Rate |
|-----------|-------------|---------------|-----------|-----------|-----------|---------------|
| ObjectFile_v4 | 1.0000 | 1.0000 | 0.0000 | nan | nan | nan |
| ImprovedObjectFile | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |

## 4. Conflict Gate Results

| Mechanism | Conflict Type | Identity | Feat Correct | Traj Correct | Feat Rate | Traj Rate | Uncertain |
|-----------|--------------|----------|-------------|-------------|-----------|-----------|-----------|
| FeatureOnly | feature_wrong_trajectory_correct | 0.0000 | 1.0000 | 0.0000 | nan | nan | nan |
| TrajectoryOnly | feature_wrong_trajectory_correct | 0.9429 | 0.0571 | 0.9429 | nan | nan | nan |
| Hybrid_b1.0 | feature_wrong_trajectory_correct | 0.0286 | 0.9524 | 0.0476 | nan | nan | nan |
| ObjectFile_v4 | feature_wrong_trajectory_correct | 0.9333 | 0.0667 | 0.9333 | nan | nan | nan |
| ImprovedObjectFile | feature_wrong_trajectory_correct | 0.6095 | 0.3905 | 0.6095 | 1.0000 | 0.0000 | 0.0000 |

## 5. Occlusion Memory

| Mechanism | Occ=0.0 | Occ=0.5 | Occ=0.75 | Occ=1.0 |
|-----------|---------|---------|----------|---------|
| FeatureOnly | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| TrajectoryOnly | 0.1442 | 0.1442 | 0.1442 | 0.1442 |
| Hybrid_b1.0 | 1.0000 | 1.0000 | 1.0000 | 0.0577 |
| ObjectFile_v4 | 0.0962 | 0.2404 | 0.3942 | 0.3269 |
| ImprovedObjectFile | 0.5577 | 0.5577 | 0.5577 | 0.4231 |

## 6. Answers

### Q1: Does ImprovedObjectFile improve normal swap-only identity?

ObjectFile_v4: 0.0962, ImprovedObjectFile: 0.5577

YES - improved by 46.1 percentage points.

### Q2: Does it maintain v4's conflict advantage?

NO - conflict resolution degraded.

### Q3: Is it more stable under occlusion without feature?

YES - improved occlusion resilience.

### Q4: Can confidence distinguish correct vs incorrect?

NO - confidence is not well calibrated.

### Q5: Final Recommendation

**tune_conflict_gate**
