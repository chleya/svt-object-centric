# SVT-v3.1: Minimal Learned Models

## 1. Purpose

v3.1 is a **minimal learned model stress test**, not a model competition. The goal is to test whether three simple learned models can simultaneously achieve:
1. Prediction skill
2. Swap-only identity tracking
3. OOD dynamics transfer
4. Crossing/occlusion robustness

No modifications to the v3 benchmark. No complex architectures.

## 2. Models

| Model | Type | Uses Position | Uses Feature | Object-Centric |
|-------|------|--------------|-------------|----------------|
| LastVelocityBaseline | baseline | yes | no | no |
| RawTrajectoryKNN | baseline | yes | no | no |
| RawDeltaKNN | baseline | yes | no | no |
| TranslationNormalizedKNN | baseline | yes | no | no |
| FeatureAwareIdentityBaseline | baseline | no | yes | no |
| RandomIdentityBaseline | baseline | no | no | no |
| MLPPositionOnly | learned | yes | no | no |
| MLPPositionFeature | learned | yes | yes | no |
| ObjectCentricFeatureModel | learned | yes | yes | yes |

PyTorch status: available

## 3. Identity Policy

Per v2.4/v3 policy:
- **identity_swap_only** is the primary identity metric
- identity_overall is diagnostic only (inflated by no-swap episodes)
- no-swap accuracy ≠ identity tracking
- All no_swap_bias_flag=True entries must be flagged

## 4. Main Results

### Prediction Skill vs Identity Swap-Only

| Model | Clean Skill | Identity Swap-Only | No-Swap Bias Gap | Bias Flag |
|-------|------------|-------------------|-----------------|-----------|
| LastVelocityBaseline | -25.2114 | 0.0400 | 0.7336 | True |
| RawTrajectoryKNN | 0.4890 | 0.6000 | 0.2679 | True |
| RawDeltaKNN | 0.7287 | 0.2000 | 0.6019 | True |
| TranslationNormalizedKNN | 0.4826 | 0.4000 | 0.4396 | True |
| FeatureAwareIdentityBaseline | nan | 1.0000 | 0.0000 | False |
| RandomIdentityBaseline | nan | 0.6000 | -0.0811 | False |
| MLPPositionOnly | 0.6159 | 0.0000 | 0.7642 | True |
| MLPPositionFeature | 0.6586 | 0.0000 | 0.7642 | True |
| ObjectCentricFeatureModel | 0.5587 | 0.0000 | 0.7642 | True |

### OOD Transfer

| Model | ID Skill | OOD Skill | OOD Drop | OOD Identity Swap-Only |
|-------|---------|----------|---------|----------------------|
| LastVelocityBaseline | 0.6532 | -8.1762 | 8.8293 | nan |
| RawTrajectoryKNN | 0.6420 | 0.4693 | 0.1728 | nan |
| RawDeltaKNN | 0.8568 | 0.6981 | 0.1587 | nan |
| TranslationNormalizedKNN | 0.6669 | 0.5774 | 0.0896 | nan |
| MLPPositionOnly | 0.8869 | 0.5947 | 0.2922 | 0.0000 |
| MLPPositionFeature | 0.8887 | 0.6402 | 0.2486 | 0.0000 |
| ObjectCentricFeatureModel | 0.9073 | 0.4996 | 0.4077 | 0.0000 |

## 5. Interpretation

### Position + Feature vs Position Only
- MLPPositionFeature identity_swap_only vs MLPPositionOnly: Feature does NOT help
- Feature input alone does not improve identity tracking — model may not be utilizing features effectively

### Object-Centric vs Plain MLP
- ObjectCentricFeatureModel vs MLPPositionFeature: Object-centric does NOT help
- Object-centric inductive bias does not significantly improve identity tracking at this scale

### RawDeltaKNN Prediction/Identity Tradeoff
- RawDeltaKNN clean_skill: 0.7287, identity_swap_only: 0.2000
- Delta-Output Identity Paradox continues: high prediction, low identity

### No-Swap Bias in Learned Models
- Some learned models show no-swap bias

### OOD Skill Drop
- OOD skill drop is significant — force field change is detectable

## 6. Failure Cases
- Minimal learned models did not solve swap-only identity under v3 benchmark.
- Best learned identity_swap_only = 0.0000, barely above random (0.5)

## 7. Final Recommendation

**add_stronger_object_centric_bias**
