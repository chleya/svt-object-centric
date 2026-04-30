# SVT-v6: Diagnosing Published Object-Centric Models

## 1. Purpose

Test whether published object-centric models (Slot Attention, RIMs, SAVi) fail
under the same SVT stress tests that exposed our ObjectFile's weaknesses.

**Key question**: Is conflict-resolution failure a structural deficiency across
the entire field, or just our ObjectFile's problem?

## 2. Models Tested

| Model | Type | Source |
|-------|------|--------|
| Slot Attention | Published (Locatello et al., 2020) | Set-based adaptation |
| RIMs | Published (Goyal et al., 2020) | Set-based adaptation |
| SAVi | Published (Kipf et al., 2022) | Set-based adaptation |
| FeatureOnly | Our baseline | Feature similarity only |
| Hybrid | Our baseline | Trajectory + feature |
| TrajOnly | Our baseline | Trajectory only |
| CFObjectFile | Our mechanism | Conflict-first ObjectFile |

## 3. Stress Tests

| Test | Description |
|------|-------------|
| baseline | Normal evaluation on swap-only test |
| feature_noise_0.1/0.3/0.5 | Gaussian noise on features |
| occlusion_0.25/0.5/0.75 | Features zeroed during occlusion |
| conflict_type_A | No-swap + flipped features (feature says swap, trajectory says no-swap) |
| feature_ablation_shuffled | Shuffled features |
| feature_ablation_zeroed | Zeroed features |
| feature_dependency | Drop in swap-only when features shuffled |
| trajectory_dependency | Swap-only with zeroed features |

## 4. Key Expected Outcomes

1. **Slot Attention should fail under conflict** — it uses feature similarity
   for slot assignment, so flipped features should mislead it (like FeatureOnly)
2. **RIMs may be more robust** — independent mechanisms with input attention
   could learn trajectory-based routing
3. **SAVi should show intermediate behavior** — temporal slot persistence helps
   but doesn't add explicit conflict resolution
4. **If all published models fail under conflict**, this validates the claim
   that "current object-centric models lack conflict-resolution structure"

## 5. Implications

- If Slot Attention fails → SVT diagnoses a field-wide structural deficiency
- If RIMs succeeds → competitive attention isn't the answer, but independent
  mechanisms with selective attention might be
- If SAVi succeeds → temporal slot persistence partially addresses the issue
