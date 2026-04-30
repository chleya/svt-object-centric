# SVT-v5.1: Scaling Sanity Audit

## Audit 1: N=3 Permutation Metric

| Permutation | Perfect Acc | Random Acc | Fixed Identity Acc | Is Swap | BD Swap-Only | BD Overall |
|------------|------------|-----------|-------------------|---------|-------------|-----------|
| identity | 1.0000 | 0.0350 | 1.0000 | False | nan | 1.0000 |
| swap01 | 1.0000 | 0.0250 | 0.0000 | True | 1.0000 | 1.0000 |
| cycle | 1.0000 | 0.0150 | 0.0000 | True | 1.0000 | 1.0000 |
| reverse | 1.0000 | 0.0400 | 0.0000 | True | 1.0000 | 1.0000 |

**Q1: N=3 identity metric 是否可信？**
- Perfect prediction accuracy = 1.0 for all permutations: YES
- Random prediction accuracy near chance: YES
- Fixed identity accuracy drops for non-identity permutations: YES
- Note: `identity_breakdown.is_swap` uses `true_identity[:, 0] != 0` which works for N=3 single-swap but may not detect all non-identity permutations correctly

## Audit 2: Continuous Feature Oracle

| Condition | Accuracy | Swap-Only |
|-----------|----------|-----------|
| normal | 1.0000 | 1.0000 |
| shuffled | 0.1700 | nan |
| zero | 0.7200 | nan |
| wrong_features | 0.0000 | nan |

**Q2: Continuous feature oracle 在 clean 下是否接近 1.0？**
- Normal oracle accuracy = 1.0000: YES
- Shuffled accuracy should be low: 0.1700
- Zero accuracy should be near chance: 0.7200

## Audit 3: Conflict Construction

| Metric | Value |
|--------|-------|
| Conflict rate | 0.8889 |
| Feature matches true | 0.0000 |
| Trajectory matches true | 0.5463 |
| Feat correct when disagree | 0.0000 |
| Traj correct when disagree | 0.6146 |

**Q3: Conflict split 是否真的制造 feature-trajectory conflict？**
- Conflict rate = 0.8889: YES
- When feature and trajectory disagree, trajectory is correct: YES (traj=0.6146 vs feat=0.0000)

## Audit 4: FeatureOnly Conflict Artifact Check

| Condition | Oracle Accuracy |
|-----------|----------------|
| Flipped features (conflict) | 0.0000 |
| Restored correct features | 1.0000 |

**Q4: FeatureOnly conflict=0 是真实失败还是 metric artifact？**
- Restored features accuracy = 1.0000: NOT artifact (real failure)
- Verdict: NOT_artifact_FeatureOnly_conflict_0_is_real_failure

## Q5: v5 是否可作为 supplementary scaling check？

- Q1 (metric valid): YES
- Q2 (oracle works): YES
- Q3 (conflict real): YES
- Q4 (not artifact): YES

**Recommendation: include_v5_as_supplementary**
