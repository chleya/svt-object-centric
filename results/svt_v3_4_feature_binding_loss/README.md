# SVT-v3.4: Feature-Binding Loss Test

## 1. Purpose

v3.3 showed that assignment heads achieve high identity accuracy but do NOT depend on features (normal = shuffled). v3.4 adds a contrastive feature binding loss that forces observed/future feature alignment.

## 2. Method

- Feature embedding head: z_obs = encoder(obs_features), z_fut = encoder(fut_features)
- Cosine similarity matrix: [B, N, N]
- Binding loss: CrossEntropy on similarity matrix with identity_labels
- Total loss: mse + identity_ce + lambda_bind * binding_loss
- Lambda sweep: [0.0, 0.1, 1.0, 5.0]
- Swap ratio: 0.3

## 3. Binding Loss Sweep

| Lambda | Model | Clean Skill | Swap-Only ID | Assignment Acc |
|--------|-------|------------|-------------|---------------|
| 0.0 | MLPFeatureBinding | 0.6984 | 0.3462 | 0.3462 |
| 0.0 | ObjectCentricFeatureBinding | 0.5593 | 0.7788 | 0.7788 |
| 0.1 | MLPFeatureBinding | 0.6975 | 0.5000 | 0.5000 |
| 0.1 | ObjectCentricFeatureBinding | 0.5956 | 0.7981 | 0.7981 |
| 1.0 | MLPFeatureBinding | 0.6598 | 0.4038 | 0.4038 |
| 1.0 | ObjectCentricFeatureBinding | 0.6136 | 0.9038 | 0.9038 |
| 5.0 | MLPFeatureBinding | 0.6906 | 0.4615 | 0.4615 |
| 5.0 | ObjectCentricFeatureBinding | 0.5557 | 0.8462 | 0.8462 |

## 4. Feature Dependency

| Lambda | Model | Normal | Shuffled | Zero | Wrong | Dep Score | Feature Dep? |
|--------|-------|--------|----------|------|-------|-----------|-------------|
| 0.0 | MLPFeatureBinding | 0.3462 | 0.3462 | 0.4231 | 0.3846 | 0.0000 | False |
| 0.0 | ObjectCentricFeatureBinding | 0.7788 | 0.7788 | 0.7308 | 0.7500 | 0.0000 | False |
| 0.1 | MLPFeatureBinding | 0.5000 | 0.5192 | 0.5288 | 0.5192 | -0.0192 | False |
| 0.1 | ObjectCentricFeatureBinding | 0.7981 | 0.7885 | 0.7404 | 0.7788 | 0.0096 | False |
| 1.0 | MLPFeatureBinding | 0.4038 | 0.4038 | 0.4519 | 0.4231 | 0.0000 | False |
| 1.0 | ObjectCentricFeatureBinding | 0.9038 | 0.9135 | 0.8462 | 0.8750 | -0.0097 | False |
| 5.0 | MLPFeatureBinding | 0.4615 | 0.4519 | 0.5000 | 0.4808 | 0.0096 | False |
| 5.0 | ObjectCentricFeatureBinding | 0.8462 | 0.8462 | 0.6154 | 0.7885 | 0.0000 | False |

## 5. Key Finding

Best feature dependency score: **0.0096**

Contrastive binding loss does NOT create feature dependency.

normal ≈ shuffled, meaning the model still relies on trajectory shortcuts despite the binding loss.

## 6. Final Recommendation

**add_feature_reconstruction_loss**
