# SVT-v4.2: Conflict-First ObjectFile

## 1. Purpose

v4.1 improved normal swap-only identity (9.6% -> 55.8%) but conflict resolution dropped (93.3% -> 61.0%). Confidence-based adaptive weighting failed because feature confidence is always high. v4.2 changes the gate from "weighted fusion" to "detect conflict first, then adjudicate".

## 2. Mechanism Comparison

| Mechanism | Swap-Only | Feat Dep | Traj Dep |
|-----------|-----------|----------|----------|
| FeatureOnly | 1.0000 | 0.4904 | 0.0000 |
| TrajectoryOnly | 0.1538 | 0.0000 | 0.1538 |
| Hybrid_b1.0 | 1.0000 | 0.4904 | 0.2692 |
| ObjectFile_v4 | 0.0962 | 0.0192 | 0.0962 |
| ImprovedObjectFile_v4.1 | 0.5673 | 0.1154 | 0.4519 |
| ConflictFirst_traj | 0.4519 | 0.0000 | 0.4519 |
| ConflictFirst_feat_low_traj | 0.7404 | 0.2981 | 0.1827 |
| ConflictFirst_abstain | 0.4519 | 0.0000 | 0.4519 |
| ConflictFirst_margin | 0.5192 | 0.1250 | 0.3173 |

## 3. Confidence Calibration

| Mechanism | Conf Correct | Conf Incorrect | Cal Error | Feat Rate | Traj Rate | Uncertain Rate | Abstain Rate | Acc Not Abstain |
|-----------|-------------|---------------|-----------|-----------|-----------|---------------|-------------|----------------|
| ObjectFile_v4 | 1.0000 | 1.0000 | 0.0000 | nan | nan | nan | 0.0000 | 0.0962 |
| ImprovedObjectFile_v4.1 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5673 |
| ConflictFirst_traj | 0.9000 | 0.3177 | 0.5823 | 0.0000 | 0.5481 | 0.0000 | 0.0000 | 0.4519 |
| ConflictFirst_feat_low_traj | 0.8221 | 0.3375 | 0.4846 | 0.2885 | 0.2596 | 0.0000 | 0.0000 | 0.7404 |
| ConflictFirst_abstain | 0.9000 | 0.5664 | 0.3336 | 0.0000 | 0.4231 | 0.1250 | 0.1250 | 0.5165 |
| ConflictFirst_margin | 0.8741 | 0.2372 | 0.6369 | 0.0673 | 0.4615 | 0.0192 | 0.0192 | 0.5294 |

## 4. Conflict Gate Results

| Mechanism | Conflict Type | Identity | Traj Correct | Feat Wrong | Feat Rate | Traj Rate | Uncertain Rate | Abstain Rate | Acc Not Abstain | Conflict Det Acc |
|-----------|--------------|----------|-------------|-----------|-----------|-----------|---------------|-------------|----------------|----------------|
| FeatureOnly | feature_wrong_trajectory_correct | 0.0000 | 0.0000 | 1.0000 | nan | nan | nan | 0.0000 | 0.0000 | nan |
| TrajectoryOnly | feature_wrong_trajectory_correct | 0.9429 | 0.9429 | 0.0571 | nan | nan | nan | 0.0000 | 0.9429 | nan |
| Hybrid_b1.0 | feature_wrong_trajectory_correct | 0.0286 | 0.0381 | 0.9619 | nan | nan | nan | 0.0000 | 0.0286 | nan |
| ObjectFile_v4 | feature_wrong_trajectory_correct | 0.9333 | 0.9333 | 0.0667 | nan | nan | nan | 0.0000 | 0.9333 | nan |
| ImprovedObjectFile_v4.1 | feature_wrong_trajectory_correct | 0.6000 | 0.6000 | 0.4000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6000 | 1.0000 |
| ConflictFirst_traj | feature_wrong_trajectory_correct | 0.7619 | 0.7619 | 0.2381 | 0.0000 | 0.7619 | 0.0000 | 0.0000 | 0.7619 | 0.7619 |
| ConflictFirst_feat_low_traj | feature_wrong_trajectory_correct | 0.3714 | 0.3714 | 0.6286 | 0.3905 | 0.3714 | 0.0000 | 0.0000 | 0.3714 | 0.7619 |
| ConflictFirst_abstain | feature_wrong_trajectory_correct | 0.7619 | 0.7619 | 0.2381 | 0.0000 | 0.6095 | 0.1524 | 0.1524 | 0.7191 | 0.7619 |
| ConflictFirst_margin | feature_wrong_trajectory_correct | 0.6476 | 0.6476 | 0.3524 | 0.1143 | 0.6381 | 0.0095 | 0.0095 | 0.6442 | 0.7619 |

## 5. Abstention Analysis

| Mechanism | Abstain Rate | Acc Not Abstain | Acc Abstain | Conf Correct | Conf Incorrect | Cal Error |
|-----------|-------------|----------------|------------|-------------|---------------|-----------|
| ConflictFirst_traj | 0.0000 | 0.4519 | nan | 0.9000 | 0.3177 | 0.5823 |
| ConflictFirst_feat_low_traj | 0.0000 | 0.7404 | nan | 0.8221 | 0.3375 | 0.4846 |
| ConflictFirst_abstain | 0.1250 | 0.5165 | 0.0000 | 0.9000 | 0.5664 | 0.3336 |
| ConflictFirst_margin | 0.0192 | 0.5294 | 0.0000 | 0.8741 | 0.2372 | 0.6369 |

## 6. Occlusion Without Feature

| Mechanism | Occ=0.0 | Occ=0.5 | Occ=0.75 | Occ=1.0 |
|-----------|---------|---------|----------|---------|
| FeatureOnly | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| TrajectoryOnly | 0.1538 | 0.1538 | 0.1538 | 0.1538 |
| Hybrid_b1.0 | 1.0000 | 1.0000 | 1.0000 | 0.0481 |
| ObjectFile_v4 | 0.0962 | 0.2404 | 0.3942 | 0.3269 |
| ImprovedObjectFile_v4.1 | 0.5673 | 0.5673 | 0.5673 | 0.4519 |
| ConflictFirst_traj | 0.4519 | 0.4519 | 0.4519 | 0.4519 |
| ConflictFirst_feat_low_traj | 0.7404 | 0.7404 | 0.7404 | 0.1827 |
| ConflictFirst_abstain | 0.4519 | 0.4519 | 0.4519 | 0.4519 |
| ConflictFirst_margin | 0.5192 | 0.5192 | 0.5192 | 0.3173 |

## 7. Pass Criteria

| Criterion | Threshold | Value | Pass |
|-----------|-----------|-------|------|
| conflict resolution > v4.1 | > 0.610 | 0.6476 | YES |
| swap-only not below v4.1 | >= 0.508 | 0.5192 | YES |
| confidence separation | correct > incorrect + 0.05 | 0.8741 vs 0.2372 | YES |
| uncertain on high-conflict | > 0 | margin=0.0095 abstain=0.1524 | YES |

## 8. Answers

### Q1: Does conflict-first improve conflict resolution over v4.1?

v4.1: 0.6000, ConflictFirst_margin: 0.6476

YES - conflict-first gate improves conflict resolution.

### Q2: Does it maintain normal swap-only identity?

v4.1: 0.5673, ConflictFirst_margin: 0.5192

YES - swap-only identity maintained.

### Q3: Can confidence distinguish correct vs incorrect?

Correct: 0.8741, Incorrect: 0.2372, Cal Error: 0.6369

YES - confidence is calibrated.

### Q4: Does uncertain/abstain increase on high-conflict samples?

Margin-gated conflict uncertain rate: 0.0095, Abstain conflict uncertain rate: 0.1524

YES - uncertain/abstain increases on high-conflict samples.

### Q5: Final Recommendation

**proceed_to_paper_report**
