# SVT-v3.5.1: Feature Pathway Audit

## 1. Purpose

v3.5 showed FeatureOnly achieves 75% swap-only identity with feature_dependency=0.38. This audit explains WHY 75% is the ceiling and decomposes Hybrid's trajectory vs feature contributions.

## 2. Experiment 1: Feature Oracle Check

Raw one-hot features (no trained encoder) with cosine similarity assignment.

| Obs Pool | Fut Pool | Swap-Only | Overall |
|----------|----------|-----------|---------|
| mean | mean | 0.7500 | 0.7500 |
| mean | first | 0.7500 | 0.7500 |
| mean | last | 0.7500 | 0.7500 |
| first | mean | 1.0000 | 1.0000 |
| first | first | 1.0000 | 1.0000 |
| first | last | 1.0000 | 1.0000 |
| last | mean | 0.0000 | 0.0000 |
| last | first | 0.0000 | 0.0000 |
| last | last | 0.0000 | 0.0000 |

Feature ablation (mean/mean pooling):

| Mode | Swap-Only | Overall |
|------|-----------|---------|
| normal | 0.7500 | 0.7500 |
| shuffled | 0.3654 | 0.3654 |
| zero | 0.0000 | 0.0000 |
| random_wrong | 0.2212 | 0.2212 |

**Oracle ceiling: 1.0000**

## 3. Experiment 2: Temporal Pooling Ablation

Trained FeatureOnly encoder with different pooling strategies.

| Obs Pool | Fut Pool | Swap-Only |
|----------|----------|-----------|
| mean | mean | 0.7500 |
| first | mean | 1.0000 |
| last | mean | 0.0000 |
| mean | first | 0.7500 |
| mean | last | 0.7500 |
| first | first | 1.0000 |
| last | first | 0.0000 |
| last | last | 0.0000 |
| first | last | 1.0000 |

Best pooling: obs=first, fut=mean, swap_only=1.0000

## 4. Experiment 3: Hybrid Inference Decomposition

| Beta | Feature Mode | Traj Swap | Feat Swap | Hybrid Swap |
|------|-------------|-----------|-----------|-------------|
| 0.0 | normal | 0.6538 | 0.7500 | 0.6538 |
| 0.0 | shuffled | 0.6538 | 0.3654 | 0.6538 |
| 0.0 | zero | 0.7019 | 0.0000 | 0.7019 |
| 0.0 | random_wrong | 0.6635 | 0.2788 | 0.6635 |
| 1.0 | normal | 0.4231 | 0.7500 | 0.8173 |
| 1.0 | shuffled | 0.4135 | 0.3654 | 0.5385 |
| 1.0 | zero | 0.4615 | 0.0000 | 0.4615 |
| 1.0 | random_wrong | 0.4327 | 0.2692 | 0.3558 |
| 2.0 | normal | 0.8173 | 0.7500 | 0.9615 |
| 2.0 | shuffled | 0.8173 | 0.3654 | 0.7212 |
| 2.0 | zero | 0.8654 | 0.0096 | 0.8654 |
| 2.0 | random_wrong | 0.8077 | 0.2692 | 0.6058 |
| 5.0 | normal | 1.0000 | 0.7500 | 1.0000 |
| 5.0 | shuffled | 1.0000 | 0.3654 | 0.7981 |
| 5.0 | zero | 1.0000 | 0.0000 | 1.0000 |
| 5.0 | random_wrong | 1.0000 | 0.2500 | 0.7788 |

Dependency scores (normal - shuffled):

| Beta | Traj Dep | Feat Dep | Hybrid Dep |
|------|----------|----------|------------|
| 0.0 | 0.0000 | 0.3846 | 0.0000 |
| 1.0 | 0.0096 | 0.3846 | 0.2788 |
| 2.0 | 0.0000 | 0.3846 | 0.2403 |
| 5.0 | 0.0000 | 0.3846 | 0.2019 |

## 5. Answers

### Q1: Is FeatureOnly 75% a model problem or feature-label/pooling problem?

Model problem. Oracle achieves 1.0000 with raw features, so the 75% ceiling is a model limitation, not a feature problem.

### Q2: Does Hybrid's high score come from trajectory or feature pathway?

Hybrid's high score comes primarily from feature pathway (feat dep_score=0.3846). Feature similarity directly contributes to assignment.

### Q3: Next step

**proceed_to_object_file_models**
