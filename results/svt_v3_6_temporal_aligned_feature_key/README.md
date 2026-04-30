# SVT-v3.6: Temporal-Aligned Feature Key Fix

## 1. Purpose

v3.5.1 proved mean pooling destroys swap-pre identity info (75% ceiling).
Oracle with obs=first achieves 100%. v3.6 uses first-timestep pooling.

## 2. Results

| Model | Beta | Clean Skill | Swap-Only ID | Assignment Acc |
|-------|------|------------|-------------|---------------|
| FeatureOnly | N/A | 0.6691 | 1.0000 | 1.0000 |
| Hybrid_b1.0 | 1.0 | 0.6911 | 0.9904 | 0.9904 |
| Hybrid_b2.0 | 2.0 | 0.6852 | 1.0000 | 1.0000 |

## 3. Feature Ablation

| Model | Beta | Feature Mode | Swap-Only ID |
|-------|------|-------------|-------------|
| FeatureOnly | N/A | normal | 1.0000 |
| FeatureOnly | N/A | shuffled | 0.5096 |
| FeatureOnly | N/A | zero | 0.0000 |
| FeatureOnly | N/A | random_wrong | 0.1250 |
| Hybrid_b1.0 | 1.0 | normal | 0.9904 |
| Hybrid_b1.0 | 1.0 | shuffled | 0.5385 |
| Hybrid_b1.0 | 1.0 | zero | 0.4327 |
| Hybrid_b1.0 | 1.0 | random_wrong | 0.3365 |
| Hybrid_b2.0 | 2.0 | normal | 1.0000 |
| Hybrid_b2.0 | 2.0 | shuffled | 0.5096 |
| Hybrid_b2.0 | 2.0 | zero | 0.1538 |
| Hybrid_b2.0 | 2.0 | random_wrong | 0.1250 |

## 4. Pathway Decomposition

| Model | Beta | Feature Mode | Traj Swap | Feat Swap | Hybrid Swap |
|-------|------|-------------|-----------|-----------|-------------|
| Hybrid_b1.0 | 1.0 | normal | 0.4038 | 1.0000 | 0.9904 |
| Hybrid_b1.0 | 1.0 | shuffled | 0.4038 | 0.5096 | 0.5385 |
| Hybrid_b1.0 | 1.0 | zero | 0.4327 | 0.0000 | 0.4327 |
| Hybrid_b1.0 | 1.0 | random_wrong | 0.4135 | 0.1635 | 0.3365 |
| Hybrid_b2.0 | 2.0 | normal | 0.0865 | 1.0000 | 1.0000 |
| Hybrid_b2.0 | 2.0 | shuffled | 0.0865 | 0.5096 | 0.5096 |
| Hybrid_b2.0 | 2.0 | zero | 0.1538 | 0.0000 | 0.1538 |
| Hybrid_b2.0 | 2.0 | random_wrong | 0.1154 | 0.1538 | 0.1250 |
| Hybrid_b1.0 | 1.0 | dep_score | 0.0000 | 0.4904 | 0.4519 |
| Hybrid_b2.0 | 2.0 | dep_score | 0.0000 | 0.4904 | 0.4904 |

## 5. Answers

### Q1: Did FeatureOnly improve from 0.75 to near 1.0?

FeatureOnly swap-only identity: **1.0000**

YES - first-timestep pooling fixed the ceiling.

### Q2: Is normal significantly higher than shuffled/zero/wrong?

normal=1.0000, shuffled=0.5096, zero=0.0000

YES - normal dominates all ablated conditions.

### Q3: Does Hybrid's high score come from feature pathway or trajectory shortcut?

Traj dep_score: 0.0000, Feat dep_score: 0.4904

Feature pathway dominates.

### Q4: Final Recommendation

**proceed_to_minimal_object_file**
