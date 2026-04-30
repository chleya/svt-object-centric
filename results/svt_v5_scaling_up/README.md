# SVT-v5: Scaling Up — 3 Objects + Continuous Features

## 1. Purpose

Test whether the diagnostic chain (clean feature matching ≠ object-file) holds when scaling from 2 objects + one-hot features to 3 objects + 16-dim continuous features. Also test whether a multi-task trajectory predictor (pos+vel+accel) improves ObjectFile performance.

## 2. Mechanism Comparison

| Experiment | Mechanism | Swap-Only | Feat Dep | Traj Dep |
|-----------|-----------|-----------|----------|----------|
| 2obj_onehot | FO_2obj_onehot | 1.0000 | 0.4894 | 0.0000 |
| 2obj_onehot | Traj_2obj_onehot | 0.1596 | 0.0000 | 0.1596 |
| 2obj_onehot | MTTraj_2obj_onehot | 0.1809 | 0.0000 | 0.1809 |
| 2obj_onehot | CF_margin_2obj_onehot | 0.5532 | 0.1277 | 0.2872 |
| 2obj_onehot | CF_margin_MT_2obj_onehot | 0.5745 | 0.1277 | 0.2979 |
| 3obj_cont16 | FO_3obj_cont16 | 1.0000 | 0.8333 | 0.0000 |
| 3obj_cont16 | Traj_3obj | 0.0667 | 0.0000 | 0.0667 |
| 3obj_cont16 | MTTraj_3obj | 0.0500 | 0.0000 | 0.0500 |
| 3obj_cont16 | CF_margin_traj_3obj_cont16 | 0.6500 | 0.4500 | 0.1500 |
| 3obj_cont16 | CF_margin_MT_3obj_cont16 | 0.6333 | 0.4500 | 0.1000 |

## 3. Conflict Test

| Experiment | Mechanism | Conflict Identity | Traj Correct |
|-----------|-----------|------------------|-------------|
| 2obj_onehot | FO_2obj_onehot | 0.0000 | 0.0000 |
| 2obj_onehot | Traj_2obj_onehot | 0.9524 | 0.9524 |
| 2obj_onehot | CF_margin_2obj_onehot | 0.7619 | 0.7619 |
| 3obj_cont16 | FO_3obj_cont16 | 0.0000 | 0.0000 |
| 3obj_cont16 | Traj_3obj | 0.8148 | 0.8148 |
| 3obj_cont16 | MTTraj_3obj | 0.8241 | 0.8241 |
| 3obj_cont16 | CF_margin_traj_3obj_cont16 | 0.3426 | 0.3426 |
| 3obj_cont16 | CF_margin_MT_3obj_cont16 | 0.3889 | 0.3889 |

## 4. Key Questions

1. Does FeatureOnly still fail under conflict with 3 objects + continuous features?
2. Does ConflictFirstObjectFile maintain correct structural bias with 3 objects?
3. Does multi-task trajectory predictor improve ObjectFile performance?
4. Does the diagnostic chain generalize?

## 5. Expected Outcomes

- FeatureOnly should still fail under conflict (diagnostic chain holds)
- ConflictFirstObjectFile should maintain structural bias (possibly with lower absolute numbers)
- Multi-task trajectory predictor should improve swap-only identity
- The trade-off between normal performance and conflict resolution should persist
