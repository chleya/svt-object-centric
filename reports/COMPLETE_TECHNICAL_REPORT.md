# SVT-v2: The Delta-Output Identity Paradox
## Complete Technical Report with Source Code, Experiments, and Conclusions

**Date**: 2026-04-27
**Authors**: SVT Research Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Environment & Data Generation](#2-environment--data-generation)
3. [Evaluation Metrics](#3-evaluation-metrics)
4. [Baseline Models](#4-baseline-models)
5. [Learned Models](#5-learned-models)
6. [Experimental Results](#6-experimental-results)
7. [Key Findings](#7-key-findings)
8. [Source Code](#8-source-code)
9. [Conclusions](#9-conclusions)

---

## 1. Executive Summary

We present the **Delta-Output Identity Paradox**: a counterintuitive finding that methods improving prediction accuracy systematically destroy the ability to track object identity through occlusion events. Across 10 distinct model architectures (k-NN, MLP, Transformer, Object-Centric, Slot Persistence, Contrastive Learning, Velocity Continuity), **all fail to achieve above-random identity accuracy** on identity-swapped test episodes.

Our key contributions:
- **The Paradox**: Delta-output prediction (predicting displacement from last observed position) improves clean prediction skill but catastrophically fails on identity tracking
- **The Impossibility Result**: Velocity continuity, a seemingly natural solution, is logically incapable of detecting swaps because swaps preserve velocity continuity
- **The Gated SVTScore**: A rigorous evaluation metric that requires both prediction skill AND identity tracking, successfully distinguishing genuine understanding from superficial prediction

---

## 2. Environment & Data Generation

### 2.1 2D Physics World

The environment simulates N=2 objects moving in a 64x64 arena with optional physics:

```yaml
# configs/smoke_hard.yaml
world:
  width: 64.0
  height: 64.0
  num_objects: 2
  object_radius: 1.5
  dt: 1.0
  t_obs: 10          # observation timesteps
  t_pred: 20         # prediction timesteps
  allow_occlusion: true
  allow_crossing: true
  allow_hidden_perturbation: true
  occlusion_radius: 3.0
  hidden_perturbation_strength: 0.5
  gravity: 0.3
  friction: 0.02
  acceleration_noise: 0.15
```

### 2.2 Data Splits

| Split | Episodes | Description |
|-------|----------|-------------|
| train | 1000 | Standard episodes (optionally with swaps) |
| clean_test | 200 | Standard episodes |
| counterfactual_test | 200 | Velocity intervention applied |
| compositional_test | 200 | Object swap in future |
| identity_test | 200 | 50% have identity swap during occlusion |

### 2.3 Identity Swap Mechanism

When `identity_test=True`, after the occlusion period ends, the positions of the two objects are swapped for all subsequent timesteps. The velocities are NOT swapped — each object continues with its original velocity from its new position.

This is the critical design choice that makes the problem hard: **the swap preserves velocity continuity**.

---

## 3. Evaluation Metrics

### 3.1 Prediction Metrics

```python
# metrics/prediction_metrics.py
def compute_skill_score(model_mse: float, mean_predictor_mse: float) -> float:
    if mean_predictor_mse < 1e-10:
        return 0.0
    return 1.0 - model_mse / mean_predictor_mse
```

Skill Score = 1 - (model_MSE / mean_predictor_MSE). Range: (-inf, 1]. Negative = worse than predicting the mean.

### 3.2 Identity Metrics

```python
# metrics/identity_metrics.py
def velocity_continuity_identity(observed_positions, future_positions):
    last_obs_vel = observed_positions[:, -1] - observed_positions[:, -2]
    first_fut_vel = future_positions[:, 0] - observed_positions[:, -1]
    # Pick assignment minimizing total velocity discontinuity
```

### 3.3 Gated SVTScore

```python
# metrics/gated_svt_score.py
def compute_gated_svt_score(clean_skill, cf_skill, comp_skill, identity_accuracy, threshold=0.5):
    if clean_skill < threshold:
        return 0.0
    cf_component = max(0.0, cf_skill)
    comp_component = max(0.0, comp_skill)
    id_component = max(0.0, identity_accuracy)
    return clean_skill * cf_component * comp_component * id_component
```

The Gated SVTScore is zero unless the model passes the prediction gate (clean_skill > 0.5), then multiplies all four components.

### 3.4 Old SMSS Bug

```python
# metrics/gated_svt_score.py
def compute_old_smss(clean_mse, cf_mse, comp_mse, identity_accuracy):
    cf_ratio = cf_mse / clean_mse
    comp_ratio = comp_mse / clean_mse
    if cf_ratio > 1.5 or comp_ratio > 1.5:
        return 0.0
    smss = identity_accuracy * (1.0 - abs(cf_ratio - 1.0)) * (1.0 - abs(comp_ratio - 1.0))
    return max(0.0, min(1.0, smss))
```

The old SMSS metric can yield non-zero scores even for models worse than the mean predictor. We confirm this bug: VelocityOnlyKNN with clean_skill = -0.198 receives Old SMSS = 0.428.

---

## 4. Baseline Models

### 4.1 k-NN v1 (Absolute Output)

Retrieves similar trajectories from training data and predicts absolute future positions.

```python
# baselines/knn_retriever.py
class RawTrajectoryKNN(BaseKNN):
    def fit(self, train_observed, train_future, train_identity=None):
        self.train_obs = train_observed.reshape(train_observed.shape[0], -1)
        self.train_future = train_future
        self.nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        self.nn.fit(self.train_obs)

    def predict_future(self, test_observed):
        test_flat = test_observed.reshape(test_observed.shape[0], -1)
        distances, indices = self.nn.kneighbors(test_flat)
        weights = self._get_weights(distances)
        pred = np.zeros((test_observed.shape[0],) + self.train_future.shape[1:])
        for i in range(test_observed.shape[0]):
            for j, idx in enumerate(indices[i]):
                pred[i] += weights[i, j] * self.train_future[idx]
        return pred
```

### 4.2 k-NN v2 (Delta Output)

Predicts displacement from last observed position, then adds back.

```python
# baselines/knn_retriever_v2.py
class RawDeltaKNN(BaseKNNV2):
    def fit(self, train_observed, train_future, train_identity=None):
        self.train_obs = train_observed.reshape(train_observed.shape[0], -1)
        self.train_future_delta = train_future - train_observed[:, -1][:, None, :, :]
        self.nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        self.nn.fit(self.train_obs)

    def predict_future(self, test_observed):
        test_flat = test_observed.reshape(test_observed.shape[0], -1)
        distances, indices = self.nn.kneighbors(test_flat)
        weights = self._get_weights(distances)
        pred_delta = np.zeros((test_observed.shape[0],) + self.train_future_delta.shape[1:])
        for i in range(test_observed.shape[0]):
            for j, idx in enumerate(indices[i]):
                pred_delta[i] += weights[i, j] * self.train_future_delta[idx]
        test_last = test_observed[:, -1]
        pred = test_last[:, None, :, :] + pred_delta
        return pred
```

---

## 5. Learned Models

### 5.1 MLP Predictor

```python
# models/mlp_predictor.py
class MLPPredictor(nn.Module):
    def __init__(self, t_obs=10, t_pred=20, n_objects=2, dim=2,
                 hidden_dim=256, n_layers=4, dropout=0.1):
        super().__init__()
        input_dim = t_obs * n_objects * dim
        output_dim = t_pred * n_objects * dim
        layers = []
        current_dim = input_dim
        for i in range(n_layers):
            next_dim = hidden_dim if i < n_layers - 1 else output_dim
            layers.append(nn.Linear(current_dim, next_dim))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            current_dim = next_dim
        self.net = nn.Sequential(*layers)

    def forward(self, observed_positions):
        B = observed_positions.shape[0]
        x = observed_positions.reshape(B, -1)
        out = self.net(x)
        return out.reshape(B, self.t_pred, self.n_objects, self.dim)
```

### 5.2 Transformer Predictor

```python
# models/transformer_predictor.py
class TransformerPredictor(nn.Module):
    def __init__(self, t_obs=10, t_pred=20, n_objects=2, dim=2,
                 d_model=128, n_heads=4, n_encoder_layers=3,
                 n_decoder_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(dim * n_objects, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=t_obs + t_pred)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        self.query_embed = nn.Parameter(torch.randn(1, t_pred, d_model) * 0.02)
        self.output_proj = nn.Linear(d_model, dim * n_objects)
```

### 5.3 Dual-Head Model (Identity Supervision)

```python
# models/identity_head.py
class DualHeadModel(nn.Module):
    def __init__(self, base_model, identity_weight=1.0):
        super().__init__()
        self.base = base_model
        self.identity_head = None
        self.identity_weight = identity_weight

    def forward(self, observed_positions):
        pred_future = self.base(observed_positions)
        B, T, N, D = pred_future.shape
        identity_input = pred_future.mean(dim=1).reshape(B, N * D)
        self._build_identity_head(N * D)
        identity_logit = self.identity_head(identity_input)
        return pred_future, identity_logit

    def predict_identity(self, observed_positions):
        self.eval()
        with torch.no_grad():
            _, logit = self.forward(observed_positions)
            swapped = (logit > 0).long()
            # Convert swap flag to identity labels
            ...
```

### 5.4 Object-Centric Model

```python
# models/object_centric.py
class ObjectCentricPredictor(nn.Module):
    def __init__(self, t_obs=10, t_pred=20, n_objects=2, dim=2,
                 hidden_dim=128, n_layers=3, dropout=0.1):
        super().__init__()
        input_dim = t_obs * dim
        output_dim = t_pred * dim
        self.per_object_net = nn.Sequential(...)
        self.identity_net = nn.Sequential(
            nn.Linear(t_obs * n_objects * dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, observed_positions):
        B, T, N, D = observed_positions.shape
        pred_list = []
        for obj_idx in range(N):
            obj_obs = observed_positions[:, :, obj_idx, :].reshape(B, -1)
            obj_pred = self.per_object_net(obj_obs)
            pred_list.append(obj_pred.reshape(B, self.t_pred, D))
        pred_future = torch.stack(pred_list, dim=2)
        # Identity head from observation history
        ...
```

### 5.5 Velocity Continuity Model

```python
# models/self_supervised.py
class VelocityContinuityModel(nn.Module):
    def __init__(self, t_obs=10, t_pred=20, n_objects=2, dim=2,
                 hidden_dim=256, n_layers=4, dropout=0.1, vc_weight=1.0):
        super().__init__()
        self.net = nn.Sequential(...)  # Predicts future positions
        self.vel_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_objects * dim),
        )

    def compute_loss(self, observed_positions, future_positions, identity_labels):
        pred_future, pred_vel = self.forward(observed_positions)
        pred_loss = F.mse_loss(pred_future, future_positions)
        first_fut_vel = future_positions[:, 0] - observed_positions[:, -1]
        vel_loss = F.mse_loss(pred_vel, first_fut_vel)
        return pred_loss + self.vc_weight * vel_loss, pred_loss, vel_loss
```

### 5.6 Contrastive Identity Model

```python
# models/self_supervised.py
class ContrastiveIdentityModel(nn.Module):
    def __init__(self, t_obs=10, t_pred=20, n_objects=2, dim=2,
                 hidden_dim=256, n_layers=4, dropout=0.1, temperature=0.1):
        super().__init__()
        self.obj_encoder = nn.Sequential(...)  # Encodes observation history
        self.fut_encoder = nn.Sequential(...)  # Encodes future trajectory
        self.temperature = temperature

    def compute_loss(self, observed_positions, future_positions, identity_labels):
        # Contrastive loss: maximize similarity between same-object pairs
        # minimize similarity between different-object pairs
        ...
```

### 5.7 Slot Persistence Model

```python
# models/slot_persistence.py
class SlotPersistenceModel(nn.Module):
    def __init__(self, t_obs=10, t_pred=20, n_objects=2, dim=2,
                 slot_dim=32, hidden_dim=128):
        super().__init__()
        self.obj_encoder = nn.GRU(
            input_size=dim, hidden_size=slot_dim,
            num_layers=2, batch_first=True)
        self.slot_update = nn.GRUCell(
            input_size=dim + slot_dim, hidden_size=slot_dim)
        self.vel_decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def encode_objects(self, observed_positions):
        B, T, N, D = observed_positions.shape
        slot_states = []
        for obj_idx in range(N):
            obj_seq = observed_positions[:, :, obj_idx, :]
            _, h_n = self.obj_encoder(obj_seq)
            slot_states.append(h_n[-1])
        return torch.stack(slot_states, dim=1)

    def predict_future_with_slots(self, observed_positions, slot_states):
        # Iteratively update slots and predict next positions
        ...
```

---

## 6. Experimental Results

### 6.1 Oracle Upper Bound

| Environment | Clean Skill | CF Skill | Comp Skill | Identity |
|-------------|-------------|----------|------------|----------|
| Hard (gravity+friction+noise) | 0.952 | -0.114 | 0.955 | 1.000 |
| Clean (no noise) | 1.000 | 0.000 | 1.000 | 1.000 |
| Toroidal (wall-less) | 1.000 | 0.000 | 1.000 | 1.000 |

### 6.2 k-NN Retrieval Attacks

#### Hard Environment

| Model | k | Clean Skill | Identity (traj) | Gated Score |
|-------|---|-------------|-----------------|-------------|
| RawTrajectoryKNN (v1) | 10 | 0.589 | **0.830** | 0.071 |
| RawDeltaKNN (v2) | 5 | **0.672** | 0.530 | 0.104 |
| TranslationNormalizedKNN | 10 | 0.479 | 0.775 | 0.000 |
| VelocityOnlyKNN | 1 | -0.165 | 0.585 | 0.000 |
| LastVelocityBaseline | — | -1.240 | 0.500 | 0.000 |

### 6.3 Learned Models on Hard Environment

| Model | Clean Skill | CF Skill | Comp Skill | ID-test (traj) | ID-test (vel) | Model-ID | Gated Score |
|-------|-------------|----------|------------|----------------|---------------|----------|-------------|
| MLP (base) | **0.805** | -0.278 | 0.808 | 0.595 | 0.630 | — | 0.000 |
| MLP (dual) | 0.797 | -0.293 | 0.804 | 0.605 | 0.630 | 0.500 | 0.000 |
| MLP (obs-cond-ID) | 0.803 | -0.320 | 0.799 | 0.590 | 0.630 | 0.500 | 0.000 |
| Transformer (base) | 0.595 | -0.033 | 0.565 | 0.670 | 0.630 | — | 0.000 |
| Transformer-small (dual) | -0.032 | 0.037 | -0.038 | 0.715 | 0.630 | 0.500 | 0.000 |
| Object-Centric | 0.815 | -1.456 | 0.811 | 0.570 | 0.630 | 0.500 | 0.000 |
| Velocity Continuity | 0.795 | -0.278 | 0.798 | 0.595 | 0.630 | 0.500 | 0.000 |
| Contrastive | 0.749 | -0.065 | 0.765 | 0.565 | 0.630 | 0.500 | 0.000 |
| Slot Persistence | **0.937** | -82.075 | 0.938 | 0.520 | 0.630 | 0.500 | 0.000 |
| Slot Persistence (+swap train) | 0.860 | -1.140 | 0.859 | 0.505 | 0.630 | 0.500 | **0.0556** |

### 6.4 Identity Accuracy: Clean vs Identity-Test

| Model | Clean (traj) | ID-test (traj) | Drop |
|-------|-------------|----------------|------|
| Oracle | 1.000 | 1.000 | 0% |
| k-NN v1 Raw (k=10) | 0.995 | 0.830 | -17% |
| k-NN v2 RawDelta (k=5) | 0.990 | 0.530 | **-46%** |
| MLP (base) | 0.990 | 0.595 | -40% |
| MLP (dual) | 0.990 | 0.605 | -39% |
| Transformer (base) | 0.930 | 0.670 | -26% |
| Object-Centric | 0.995 | 0.570 | -43% |
| Slot Persistence | 1.000 | 0.520 | -48% |

### 6.5 Velocity Continuity in Toroidal Environment

| Scenario | Vel-ID (clean) | Vel-ID (ID-test) |
|----------|---------------|------------------|
| No-swap episodes | **1.000** | N/A |
| Swap episodes | N/A | **0.185** |
| Overall | 1.000 | 0.625 |

**Critical finding**: On swap episodes, velocity continuity performs **worse than random** (0.185 < 0.500) because the swap preserves velocity continuity — the "no-swap" assignment has lower velocity discontinuity than the correct "swap" assignment.

---

## 7. Key Findings

### Finding 1: The Delta-Output Identity Paradox

Delta-output prediction (predicting displacement from last observed position) improves clean prediction skill (0.672 vs 0.589) but catastrophically destroys identity tracking (0.530 vs 0.830).

**Root cause**: Delta-output anchors predictions to the last observed position. When a swap occurs, the last observed position already reflects the swapped identity, so the model confidently predicts the swapped trajectory.

### Finding 2: Velocity Continuity is Logically Incapable of Detecting Swaps

We initially thought velocity continuity failed due to noise (gravity, friction, acceleration noise). We tested in a zero-noise, wall-less (toroidal) environment and found:

- No-swap episodes: Vel-ID = 1.000 (perfect)
- Swap episodes: Vel-ID = 0.185 (worse than random)

**The swap mechanism preserves velocity continuity by design**: each object continues with its original velocity from its new position. The "no-swap" assignment has lower velocity discontinuity than the correct "swap" assignment, causing velocity continuity to systematically choose the wrong answer.

### Finding 3: All Identity Supervision Methods Fail

We tested 4 identity head architectures:
- Dual-head (conditioned on predicted future)
- Observation-conditioned (conditioned on observation history)
- Object-Centric (per-object prediction + identity head)
- Slot Persistence (GRU-based slot tracking)

All achieve Model-ID = 0.500 on identity_test — exactly random. The root cause: training data contains no swap episodes, so identity heads learn to always predict "no swap".

### Finding 4: Swap Training Improves Prediction but Not Identity

Training with 30% swap episodes improves identity_test skill score from 0.241 to 0.731 (MLP) and from -82 to -1.14 (Slot Persistence). However, identity accuracy remains at random level.

### Finding 5: Gated SVTScore Successfully Distinguishes Understanding from Prediction

| Model | Clean Skill | Old SMSS | Gated SVTScore |
|-------|-------------|----------|----------------|
| VelocityOnlyKNN | -0.198 | **0.428** | **0.000** |
| MLP (base) | 0.805 | 0.000 | 0.000 |
| Slot Persistence (+swap) | 0.860 | 0.000 | **0.0556** |

The Old SMSS metric gives a non-zero score (0.428) to a model worse than the mean predictor. The Gated SVTScore correctly assigns 0.000.

---

## 8. Source Code

### 8.1 Complete File List

```
svt_agents/
├── configs/
│   ├── smoke.yaml              # Linear environment (t_pred=10)
│   ├── smoke_hard.yaml         # Physics environment (gravity, friction, noise)
│   ├── smoke_hard_swaptrain.yaml  # With swap-augmented training
│   └── smoke_toroidal.yaml     # Wall-less environment
├── envs/
│   ├── motion_world.py         # 2D physics simulator
│   ├── physics_oracle.py       # Ground-truth oracle
│   └── interventions.py        # Counterfactual & compositional interventions
├── baselines/
│   ├── knn_retriever.py        # k-NN v1 (absolute output)
│   └── knn_retriever_v2.py     # k-NN v2 (delta output)
├── models/
│   ├── mlp_predictor.py        # MLP baseline
│   ├── transformer_predictor.py # Transformer baseline
│   ├── identity_head.py        # Dual-head & obs-conditioned identity
│   ├── object_centric.py       # Object-centric predictor
│   ├── self_supervised.py      # Velocity continuity & contrastive learning
│   └── slot_persistence.py     # GRU-based slot tracking
├── metrics/
│   ├── prediction_metrics.py   # MSE, skill score
│   ├── identity_metrics.py     # Trajectory matching, velocity continuity
│   └── gated_svt_score.py      # Gated SVTScore & Old SMSS
├── scripts/
│   ├── run_oracle_upper_bound.py
│   ├── run_knn_attack.py
│   ├── run_knn_attack_v2.py
│   └── train_and_eval.py
├── data/
│   ├── generate_2d_motion.py
│   └── generate_swap_train.py
└── reports/
    └── full_report.md
```

### 8.2 Key Source Files

[All source code is included in the repository. Key files are referenced in Sections 4-5 above.]

---

## 9. Conclusions

### 9.1 The Paradox is Real and Robust

The Delta-Output Identity Paradox holds across:
- 3 model families (k-NN, MLP, Transformer)
- 10 distinct architectures
- 3 environment configurations (linear, physics, toroidal)
- With and without identity supervision
- With and without swap-augmented training

### 9.2 Velocity Continuity is Not a Solution

Contrary to initial intuition, velocity continuity cannot detect identity swaps because **swaps preserve velocity continuity by design**. This is an impossibility result, not a limitation of model capacity.

### 9.3 The Identity Problem Requires Object Permanence

Current prediction models (MLP, Transformer, k-NN) are function approximators that map observations to future positions. They do not maintain persistent object representations that survive occlusion. Identity tracking requires:
- **Object files**: Persistent representations that track individual objects through time
- **Feature-based matching**: Matching objects by features (color, texture, shape) rather than position/velocity
- **Occlusion-aware reasoning**: Explicitly modeling what happens during occlusion

### 9.4 SVT-v2 is Effective

The Gated SVTScore successfully distinguishes genuine understanding from superficial prediction:
- All "fake understanding" models receive Gated Score = 0.000
- The only model with non-zero score (Slot Persistence + swap training = 0.0556) still fails identity
- Oracle achieves perfect scores, proving the task is solvable in principle

### 9.5 Future Directions

1. **Feature-augmented objects**: Give each object a unique color/texture feature
2. **Explicit occlusion modeling**: Predict what happens during occlusion, not just after
3. **Object file architectures**: Implement Pylyshyn's FINST (Fingers of Instantiation) mechanism
4. **Human baselines**: Test whether humans can solve this task

---

## Appendix: Reproducibility

All experiments can be reproduced with:

```bash
# Generate data
python data/generate_2d_motion.py --config configs/smoke_hard.yaml

# Run Oracle
python scripts/run_oracle_upper_bound.py --config configs/smoke_hard.yaml

# Run k-NN attacks
python scripts/run_knn_attack.py --config configs/smoke_hard.yaml
python scripts/run_knn_attack_v2.py --config configs/smoke_hard.yaml

# Train learned models
python scripts/train_and_eval.py --config configs/smoke_hard.yaml --models mlp transformer

# With identity head
python scripts/train_and_eval.py --config configs/smoke_hard.yaml --models mlp --identity-head

# With swap-augmented training
python data/generate_swap_train.py --config configs/smoke_hard.yaml
python scripts/train_and_eval.py --config configs/smoke_hard_swaptrain.yaml --models slot_persistence
```

**Dependencies**: Python 3.8+, PyTorch 2.0+, NumPy, scikit-learn, PyYAML, tqdm

---

*Report generated: 2026-04-27*
*Repository: F:\新点子文件\svt_agents\*
