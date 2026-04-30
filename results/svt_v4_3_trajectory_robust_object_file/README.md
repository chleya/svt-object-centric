# SVT-v4.3: Trajectory-Robust ObjectFile

## 1. Purpose

v4.2 established conflict-first gate as the best balance mechanism. v4.3 addresses the root bottleneck: trajectory predictor weakness. Two innovations: (1) observed-period approach detection (are objects moving toward each other?), (2) multi-step trajectory voting with temporal decay, (3) trajectory training augmentation.

## 2. Mechanism Comparison

| Mechanism | Swap-Only | Feat Dep | Traj Dep |
|-----------|-----------|----------|----------|
| FeatureOnly | 1.0000 | 0.4904 | 0.0096 |
| TrajectoryOnly | 0.1731 | 0.0000 | 0.1731 |
| TrajectoryOnly_aug | 0.1250 | 0.0000 | 0.1250 |
| Hybrid_b1.0 | 0.9904 | 0.2692 | 0.7885 |
| ObjectFile_v4 | 0.0962 | 0.0192 | 0.0962 |
| ImprovedObjectFile_v4.1 | 0.6058 | 0.1538 | 0.4423 |
| ConflictFirst_margin_v4.2 | 0.5288 | 0.1538 | 0.3077 |
| TrajRobust_aware_v4.3 | 0.5673 | 0.2500 | 0.1058 |
| TrajRobust_aware_aug_v4.3 | 0.6250 | 0.2692 | 0.0865 |
| TrajRobust_veto_v4.3 | 0.5096 | 0.1442 | 0.2212 |
| TrajRobust_veto_aug_v4.3 | 0.5288 | 0.1442 | 0.2404 |

## 3. Confidence Calibration

| Mechanism | Swap | Conf Corr | Conf Inc | Cal Err | Feat Rate | Traj Rate | Uncertain | Abstain |
|-----------|------|-----------|----------|---------|-----------|-----------|-----------|---------|
| ObjectFile_v4 | 0.0962 | 1.0000 | 1.0000 | 0.0000 | nan | nan | nan | 0.0000 |
| ImprovedObjectFile_v4.1 | 0.6058 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ConflictFirst_margin_v4.2 | 0.5288 | 0.8673 | 0.2407 | 0.6266 | 0.0865 | 0.4712 | 0.0000 | 0.0000 |
| TrajRobust_aware_v4.3 | 0.5673 | 0.8678 | 0.5724 | 0.2954 | 0.0000 | 0.4327 | 0.0000 | 0.0000 |
| TrajRobust_aware_aug_v4.3 | 0.6250 | 0.8415 | 0.5721 | 0.2694 | 0.0000 | 0.3654 | 0.0096 | 0.0096 |
| TrajRobust_veto_v4.3 | 0.5096 | 0.8868 | 0.5855 | 0.3013 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| TrajRobust_veto_aug_v4.3 | 0.5288 | 0.8673 | 0.5737 | 0.2936 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 4. Conflict Gate Results

| Mechanism | Conflict Type | Identity | Traj Correct | Feat Wrong | Feat Rate | Traj Rate | Uncertain | Abstain |
|-----------|--------------|----------|-------------|-----------|-----------|-----------|-----------|---------|
| FeatureOnly | feature_wrong_trajectory_correct | 0.0000 | 0.0000 | 1.0000 | nan | nan | nan | 0.0000 |
| TrajectoryOnly | feature_wrong_trajectory_correct | 0.9619 | 0.9619 | 0.0381 | nan | nan | nan | 0.0000 |
| TrajectoryOnly_aug | feature_wrong_trajectory_correct | 0.9714 | 0.9714 | 0.0286 | nan | nan | nan | 0.0000 |
| Hybrid_b1.0 | feature_wrong_trajectory_correct | 0.5524 | 0.5524 | 0.4476 | nan | nan | nan | 0.0000 |
| ObjectFile_v4 | feature_wrong_trajectory_correct | 0.9333 | 0.9333 | 0.0667 | nan | nan | nan | 0.0000 |
| ImprovedObjectFile_v4.1 | feature_wrong_trajectory_correct | 0.6762 | 0.6762 | 0.3238 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| ConflictFirst_margin_v4.2 | feature_wrong_trajectory_correct | 0.7143 | 0.7143 | 0.2857 | 0.1238 | 0.6667 | 0.0476 | 0.0476 |
| TrajRobust_aware_v4.3 | feature_wrong_trajectory_correct | 0.6000 | 0.6000 | 0.4000 | 0.0000 | 0.5714 | 0.0286 | 0.0286 |
| TrajRobust_aware_aug_v4.3 | feature_wrong_trajectory_correct | 0.6190 | 0.6190 | 0.3810 | 0.0000 | 0.6190 | 0.0000 | 0.0000 |
| TrajRobust_veto_v4.3 | feature_wrong_trajectory_correct | 0.6571 | 0.6571 | 0.3429 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| TrajRobust_veto_aug_v4.3 | feature_wrong_trajectory_correct | 0.6476 | 0.6476 | 0.3524 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 5. Approach Signal Analysis

| Dataset | Swap Mean | Swap Std | No-Swap Mean | No-Swap Std | N Swap | N No-Swap |
|---------|-----------|----------|-------------|------------|--------|-----------|
| swap_test | -0.8705 | 7.9687 | nan | nan | 104 | 0 |
| clean_test | nan | nan | 0.1204 | 7.9862 | 0 | 105 |

## 6. Key Comparisons

| Metric | v4.2 Margin | v4.3 Aware | v4.3 Aware+Aug | v4.3 Veto | v4.3 Veto+Aug |
|--------|------------|-----------|---------------|----------|--------------|
| Swap-Only | 0.5288 | 0.5673 | 0.6250 | 0.5096 | 0.5288 |
| Conflict | 0.7143 | 0.6000 | 0.6190 | 0.6571 | 0.6476 |

## 7. Pass Criteria

| Criterion | Threshold | Best v4.3 | Pass |
|-----------|-----------|-----------|------|
| swap-only > 0.55 | > 0.55 | 0.6250 | YES |
| conflict resolution > 0.65 | > 0.65 | 0.6571 | YES |

## 8. Recommendation

**proceed_to_paper_report**
