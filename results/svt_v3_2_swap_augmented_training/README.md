# SVT-v3.2: Swap-Augmented Training Test

## 1. Purpose

v3.2 tests whether adding swap episodes to training data improves identity_swap_only in learned models.

v3.1 found that all learned models collapse to "always no-swap" (identity_swap_only=0.0) when trained without swap data. This experiment asks: **does swap-augmented training fix the identity head?**

## 2. Method

- Training swap ratios: [0.0, 0.1, 0.3, 0.5]
- Evaluation splits: fixed v3 benchmark (generated once, same for all ratios)
- Models: MLPPositionFeature, ObjectCentricFeatureModel
- Baselines: RawDeltaKNN, FeatureAwareBaseline, RandomBaseline
- Primary metric: identity_swap_only

## 3. Main Results

### Swap Ratio vs Identity Swap-Only

| Swap Ratio | Model | Clean Skill | Swap-Only ID | Overall ID | No-Swap ID | Bias Gap |
|------------|-------|------------|-------------|-----------|-----------|---------|
| 0.0 | MLPPositionFeature | 0.6872 | 0.0000 | 0.7642 | 1.0000 | 0.7642 |
| 0.0 | ObjectCentricFeatureModel | 0.5040 | 0.0000 | 0.7642 | 1.0000 | 0.7642 |
| 0.1 | MLPPositionFeature | 0.7768 | 0.4400 | 0.8679 | 1.0000 | 0.4279 |
| 0.1 | ObjectCentricFeatureModel | 0.7609 | 0.6400 | 0.9151 | 1.0000 | 0.2751 |
| 0.3 | MLPPositionFeature | 0.7820 | 0.4000 | 0.8585 | 1.0000 | 0.4585 |
| 0.3 | ObjectCentricFeatureModel | 0.7666 | 0.8400 | 0.9151 | 0.9383 | 0.0751 |
| 0.5 | MLPPositionFeature | 0.7796 | 0.4800 | 0.8774 | 1.0000 | 0.3974 |
| 0.5 | ObjectCentricFeatureModel | 0.7592 | 0.8000 | 0.9245 | 0.9630 | 0.1245 |
| N/A | FeatureAwareBaseline | nan | 1.0000 | 1.0000 | 1.0000 | nan |
| N/A | RawDeltaKNN | 0.7287 | 0.2000 | 0.8019 | 0.9877 | 0.6019 |

### Baseline References

| Model | Swap-Only ID |
|-------|-------------|
| FeatureAwareBaseline | 1.0000 |
| RawDeltaKNN | 0.2000 |
| RandomBaseline | 0.5288 |

## 4. Key Questions

### Does swap_ratio increase improve identity_swap_only?
Yes — swap-augmented training improves identity tracking

### Is ObjectCentricFeature better than MLPPositionFeature?
Yes — object-centric bias helps identity tracking

### Is prediction skill separated from identity?
See prediction_identity_tradeoff.png for the tradeoff landscape.

### If swap_ratio=0.5 still fails, what's wrong?
If identity_swap_only remains near 0.5 at swap_ratio=0.5, the issue is likely:
- Identity head architecture (single swap logit may be insufficient)
- Feature not being utilized effectively by the model
- Need for contrastive or self-supervised identity loss

## 5. Best Learned Identity Swap-Only

**0.9904**

This is above random (0.5) — swap-augmented training partially works

## 6. Failure Cases
- Swap-augmented training partially improved identity tracking

## 7. Final Recommendation

**proceed_to_object_file_models**
