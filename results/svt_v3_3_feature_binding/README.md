# SVT-v3.3: Feature-Binding Sanity Test

## 1. Does ObjectCentric truly depend on features?

**No.**

Feature ablation on ObjectCentricFeatureAssignment (swap_ratio=0.3):

| Feature Mode | Identity Swap-Only | Assignment Accuracy |
|-------------|-------------------|-------------------|
| normal | 0.7596 | 0.7596 |
| shuffled | **0.7596** | **0.7596** |
| zero | 0.6923 | 0.6923 |
| random_wrong | 0.7500 | 0.7500 |

Shuffled features produce **identical** results to normal features (0.7596 = 0.7596). Zero features only drop slightly (0.6923). Random wrong features barely drop (0.7500).

**The model is NOT using feature identity information.** It is relying on trajectory patterns (swap-induced trajectory discontinuities) to predict assignment, not feature-to-identity binding.

## 2. Is normal significantly higher than shuffled/zero?

**No.**

Feature dependency scores:
- MLPPositionFeatureAssignment: **-0.0962** (negative! zero features actually outperform normal)
- ObjectCentricFeatureAssignment: **0.0000** (normal = shuffled, exactly equal)

This is a definitive negative result. The assignment head learns trajectory-based heuristics, not feature binding.

## 3. Why Features Don't Help

The N×N assignment head with CrossEntropy loss should theoretically learn to use features for identity matching. But it doesn't, because:

1. **Trajectory shortcuts are available**: Swap episodes create detectable trajectory discontinuities at the swap point. The model learns to detect these discontinuities instead of matching features.

2. **Features are redundant with positions**: In the current setup, features are one-hot vectors that correlate perfectly with initial position ordering. The model can infer "which object is which" from position patterns alone.

3. **Assignment loss doesn't force feature usage**: CrossEntropy on the assignment matrix only requires correct predictions, not feature-dependent predictions. The model finds the easiest path to correct predictions, which is trajectory-based.

## 4. Assignment Results Across Swap Ratios

| Swap Ratio | Model | Assignment Acc (swap_only) | Clean Skill |
|------------|-------|---------------------------|-------------|
| 0.0 | MLP+Assign | 0.0000 | 0.6474 |
| 0.0 | ObjCentric+Assign | 0.0000 | 0.4911 |
| 0.1 | MLP+Assign | 0.6923 | 0.7228 |
| 0.1 | ObjCentric+Assign | **0.9231** | 0.6883 |
| 0.3 | MLP+Assign | 0.4423 | 0.6927 |
| 0.3 | ObjCentric+Assign | **0.9423** | 0.5817 |
| 0.5 | MLP+Assign | 0.4519 | 0.7140 |
| 0.5 | ObjCentric+Assign | **0.8654** | 0.6321 |

ObjectCentricFeatureAssignment achieves up to 0.9423 assignment accuracy at swap_ratio=0.3, but this is **entirely trajectory-based**, not feature-based.

## 5. What Would Fix This

To achieve genuine feature binding, the model needs:
- **Contrastive feature binding loss**: Force the model to match observed features to future features, not just predict assignment from trajectories
- **Feature-only identity test**: Evaluate identity using ONLY features (no position information)
- **Decoupled prediction**: Separate trajectory prediction from identity prediction so the identity head cannot access position information

## 6. Final Recommendation

**add_contrastive_feature_binding_loss**

The current architecture does not achieve genuine feature binding. The identity prediction relies on trajectory shortcuts, not feature-to-identity mapping. A contrastive loss that explicitly forces feature matching is needed.
