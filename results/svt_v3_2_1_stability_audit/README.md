# SVT-v3.2.1: Stability Audit

## 1. Is ObjectCentricFeature stably better than MLPPositionFeature?

**Yes**

Across all 3 seeds and all swap ratios > 0, ObjectCentricFeature outperforms MLPPositionFeature on identity_swap_only:

| Seed | Swap Ratio | ObjectCentric | MLP | OC Wins? |
|------|-----------|---------------|-----|----------|
| 0 | 0.1 | 0.875 | 0.615 | Yes |
| 0 | 0.3 | 0.712 | 0.462 | Yes |
| 0 | 0.5 | 0.981 | 0.452 | Yes |
| 1 | 0.1 | 0.792 | 0.608 | Yes |
| 1 | 0.3 | 0.700 | 0.442 | Yes |
| 1 | 0.5 | 0.858 | 0.475 | Yes |
| 2 | 0.1 | 0.928 | 0.595 | Yes |
| 2 | 0.3 | 0.964 | 0.279 | Yes |
| 2 | 0.5 | 0.883 | 0.685 | Yes |

**ObjectCentricFeature wins 9/9 configurations.** The advantage is stable across seeds.

Mean identity_swap_only (swap_ratio > 0):
- ObjectCentricFeature: 0.855
- MLPPositionFeature: 0.512

## 2. Is swap_ratio=0.3 still an effective point?

**Yes**

ObjectCentricFeature at swap_ratio=0.3 across seeds:
- Seed 0: 0.712
- Seed 1: 0.700
- Seed 2: 0.964
- Mean: 0.792

swap_ratio=0.3 provides sufficient swap signal for ObjectCentricFeature to achieve identity_swap_only well above random (0.5).

However, MLPPositionFeature at swap_ratio=0.3 is inconsistent (0.462, 0.442, 0.279), suggesting the plain MLP architecture struggles to leverage swap signal even when available.

## 3. Is normal feature better than shuffled/zero feature?

**No — this is a surprising negative result.**

Feature ablation at swap_ratio=0.3:

| Feature Mode | ObjectCentric | MLP |
|-------------|---------------|-----|
| normal | 0.529 | 0.404 |
| shuffled | **0.981** | 0.471 |
| zero | 0.760 | 0.356 |

ObjectCentricFeature performs **best with shuffled features** (0.981), not normal features (0.529). This suggests:

1. The model is NOT using feature identity information in the expected way (matching one-hot vectors to track objects).
2. The presence of feature vectors may improve training dynamics (providing richer input, better gradient signal) even when the feature-to-identity mapping is broken.
3. ObjectCentricFeature's per-object encoding benefits from having additional input dimensions, regardless of whether those dimensions carry identity information.
4. The identity head likely learns from the **position trajectory patterns** (swap creates detectable trajectory discontinuities) rather than from feature matching.

This does NOT invalidate the ObjectCentricFeature advantage — it still outperforms MLPPositionFeature. But it means the feature is serving as a **training regularizer/input enrichment**, not as an identity signal.

## Final Recommendation

**proceed_to_object_file_models**

ObjectCentricFeature is stably better than MLPPositionFeature (9/9 wins), swap_ratio=0.3 is effective, but features are not being used for identity matching as expected. The next step should investigate object-file architectures that can genuinely leverage identity-bearing features.
