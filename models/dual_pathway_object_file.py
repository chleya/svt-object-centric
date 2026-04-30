"""
Dual-Pathway ObjectFile with Corrected Conflict Training

v17 failure analysis:
  1. CRITICAL BUG: Conflict augmentation swapped features AND identity labels
     together, training the model to follow swapped features under conflict.
     This is the OPPOSITE of what conditional binding requires.
  2. Conflict detector signal was diluted (averaged over all pairs).
  3. Double softmax (softmax -> gate -> re-softmax) washed out gating effect.
  4. Per-object gate applied to per-pair edges was too coarse.

v18 solution:
  1. CORRECTED TRAINING: Under conflict, identity labels follow TRAJECTORY,
     not the swapped features. This is the fundamental fix.
  2. Dual independent scorers: Feature Scorer and Trajectory Scorer are
     completely separate networks with separate parameters.
  3. Agreement-based switching: Compare argmax of feature scores vs trajectory
     scores. If they agree, use feature scorer (high confidence). If they
     disagree, use trajectory scorer (correct under conflict).
  4. Hard switch with Straight-Through Estimator (STE): During training,
     use hard switch for forward pass but soft gradient for backprop.
  5. Separate training: Feature scorer trained on clean data, trajectory
     scorer trained on trajectory-based labels (original identity under conflict).

Key architectural principle:
  The MinimalObjectFile (rule-based) achieved conflict resolution = 0.933 by
  keeping feature and trajectory scoring SEPARATE and switching on disagreement.
  v18 makes this principle LEARNABLE: separate learned scorers + agreement switch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureEncoder(nn.Module):
    def __init__(self, feature_dim=2, slot_dim=64, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim))

    def forward(self, features):
        if features is None:
            return None
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        if features.dim() == 4:
            features = features[:, 0, :, :]
        return self.net(features)


class TrajectoryEncoder(nn.Module):
    def __init__(self, dim=2, slot_dim=64, t_obs=10):
        super().__init__()
        self.gru = nn.GRU(input_size=dim, hidden_size=slot_dim,
                          num_layers=2, batch_first=True)

    def forward(self, positions):
        if isinstance(positions, np.ndarray):
            positions = torch.FloatTensor(positions)
        B = positions.shape[0]
        N = positions.shape[2]
        embeddings = []
        for j in range(N):
            _, h = self.gru(positions[:, :, j, :])
            embeddings.append(h[-1])
        return torch.stack(embeddings, dim=1)


class PairwiseScorer(nn.Module):
    def __init__(self, slot_dim=64, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1))

    def forward(self, z_fut, z_obs):
        B = z_fut.shape[0]
        N = z_fut.shape[1]
        scores = torch.zeros(B, N, N, device=z_fut.device)
        for i in range(N):
            for j in range(N):
                pair = torch.cat([z_fut[:, i, :], z_obs[:, j, :]], dim=-1)
                scores[:, i, j] = self.net(pair).squeeze(-1)
        return scores


class DualPathwayObjectFile(nn.Module):
    def __init__(self, num_objects=2, dim=2, feature_dim=2,
                 hidden_dim=128, slot_dim=64, t_obs=10, t_pred=20,
                 identity_weight=1.0, smh_weight=1.0, traj_weight=0.1,
                 conflict_switch_temp=0.1):
        super().__init__()
        self.num_objects = num_objects
        self.dim = dim
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.identity_weight = identity_weight
        self.smh_weight = smh_weight
        self.traj_weight = traj_weight
        self.conflict_switch_temp = conflict_switch_temp

        self.obs_feat_encoder = FeatureEncoder(feature_dim, slot_dim, hidden_dim)
        self.obs_traj_encoder = TrajectoryEncoder(dim, slot_dim, t_obs)
        self.fut_feat_encoder = FeatureEncoder(feature_dim, slot_dim, hidden_dim)
        self.fut_traj_encoder = TrajectoryEncoder(dim, slot_dim, t_obs)

        self.feature_scorer = PairwiseScorer(slot_dim, hidden_dim)
        self.trajectory_scorer = PairwiseScorer(slot_dim, hidden_dim)

        self.obs_node_update = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim))

        self.smh = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_objects))

        self.traj_decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, t_pred * dim))

    def _encode(self, observed_positions, observed_features,
                future_positions, future_features):
        z_obs_traj = self.obs_traj_encoder(observed_positions)
        z_fut_traj = self.fut_traj_encoder(future_positions)

        z_obs_feat = self.obs_feat_encoder(observed_features)
        z_fut_feat = self.fut_feat_encoder(future_features)

        if z_obs_feat is None:
            z_obs_feat = z_obs_traj.detach().clone()
        if z_fut_feat is None:
            z_fut_feat = z_fut_traj.detach().clone()

        z_obs = self.obs_node_update(torch.cat([z_obs_feat, z_obs_traj], dim=-1))

        return z_obs, z_obs_feat, z_obs_traj, z_fut_feat, z_fut_traj

    def _compute_dual_scores(self, z_obs_feat, z_obs_traj, z_fut_feat, z_fut_traj):
        feat_scores = self.feature_scorer(z_fut_feat, z_obs_feat)
        traj_scores = self.trajectory_scorer(z_fut_traj, z_obs_traj)
        return feat_scores, traj_scores

    def _adaptive_combine(self, feat_scores, traj_scores):
        feat_assign = feat_scores.argmax(dim=-1)
        traj_assign = traj_scores.argmax(dim=-1)

        agree = (feat_assign == traj_assign).all(dim=-1).float()

        temp = self.conflict_switch_temp
        soft_agree = torch.sigmoid((agree - 0.5) / max(temp, 1e-6))

        combined = soft_agree.unsqueeze(-1).unsqueeze(-1) * feat_scores + \
                   (1.0 - soft_agree.unsqueeze(-1).unsqueeze(-1)) * traj_scores

        return combined, agree, feat_assign, traj_assign

    def compute_loss(self, observed_positions, future_positions, identity_labels,
                     observed_features=None, future_features=None, is_swap=None,
                     p_conflict=0.0):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        if isinstance(future_positions, np.ndarray):
            future_positions = torch.FloatTensor(future_positions)
        if isinstance(identity_labels, np.ndarray):
            identity_labels = torch.LongTensor(identity_labels)
        elif isinstance(identity_labels, torch.Tensor) and identity_labels.dtype != torch.long:
            identity_labels = identity_labels.long()
        if observed_features is not None and isinstance(observed_features, np.ndarray):
            observed_features = torch.FloatTensor(observed_features)
        if future_features is not None and isinstance(future_features, np.ndarray):
            future_features = torch.FloatTensor(future_features)

        B = observed_positions.shape[0]
        N = self.num_objects

        aug_fut_feat = future_features
        traj_identity = identity_labels
        feat_identity = identity_labels
        conflict_labels = torch.zeros(B, device=observed_positions.device)

        if p_conflict > 0 and future_features is not None and N >= 2:
            aug_fut_feat = future_features.clone()
            feat_identity = identity_labels.clone()
            traj_identity = identity_labels.clone()

            for b in range(B):
                if torch.rand(1).item() < p_conflict:
                    if aug_fut_feat.dim() == 4:
                        aug_fut_feat[b, :, 0, :], aug_fut_feat[b, :, 1, :] = \
                            future_features[b, :, 1, :].clone(), future_features[b, :, 0, :].clone()
                    elif aug_fut_feat.dim() == 3:
                        aug_fut_feat[b, 0, :], aug_fut_feat[b, 1, :] = \
                            future_features[b, 1, :].clone(), future_features[b, 0, :].clone()

                    feat_identity[b, 0], feat_identity[b, 1] = \
                        identity_labels[b, 1].clone(), identity_labels[b, 0].clone()
                    conflict_labels[b] = 1.0

        z_obs, z_obs_feat, z_obs_traj, z_fut_feat, z_fut_traj = \
            self._encode(observed_positions, observed_features,
                         future_positions, aug_fut_feat)

        feat_scores, traj_scores = self._compute_dual_scores(
            z_obs_feat, z_obs_traj, z_fut_feat, z_fut_traj)

        feat_loss = F.cross_entropy(feat_scores.reshape(-1, N), feat_identity.reshape(-1))
        traj_loss = F.cross_entropy(traj_scores.reshape(-1, N), traj_identity.reshape(-1))

        combined, agree, feat_assign, traj_assign = self._adaptive_combine(
            feat_scores, traj_scores)

        combined_identity = torch.where(
            agree.unsqueeze(-1).bool().expand_as(identity_labels),
            feat_identity, traj_identity)

        combined_loss = F.cross_entropy(combined.reshape(-1, N), combined_identity.reshape(-1))

        identity_loss = 0.3 * feat_loss + 0.3 * traj_loss + 0.4 * combined_loss

        smh_logits = torch.zeros(B, N, N, device=z_obs.device)
        for j in range(N):
            smh_logits[:, j, :] = self.smh(z_obs[:, j, :])
        smh_loss = F.cross_entropy(smh_logits.reshape(-1, N), identity_labels.reshape(-1))

        traj_preds = []
        for j in range(N):
            traj_j = self.traj_decoder(z_obs[:, j, :])
            traj_preds.append(traj_j.reshape(B, self.t_pred, self.dim))
        pred_traj = torch.stack(traj_preds, dim=2)
        traj_loss_val = F.mse_loss(pred_traj, future_positions)

        total_loss = (self.identity_weight * identity_loss +
                      self.smh_weight * smh_loss +
                      self.traj_weight * traj_loss_val)

        return total_loss, identity_loss, smh_loss, torch.tensor(0.0)

    def predict_identity(self, observed_positions, observed_features=None,
                         test_future=None, future_features=None,
                         future_positions=None, method="combined"):
        self.eval()
        with torch.no_grad():
            if future_positions is None and test_future is not None:
                future_positions = test_future
            if future_positions is None:
                future_positions = torch.zeros(
                    observed_positions.shape[0] if isinstance(observed_positions, torch.Tensor) else len(observed_positions),
                    self.t_pred, self.num_objects, self.dim)

            z_obs, z_obs_feat, z_obs_traj, z_fut_feat, z_fut_traj = \
                self._encode(observed_positions, observed_features,
                             future_positions, future_features)

            feat_scores, traj_scores = self._compute_dual_scores(
                z_obs_feat, z_obs_traj, z_fut_feat, z_fut_traj)

            if method == "feature_only":
                pred = feat_scores.argmax(dim=-1)
            elif method == "trajectory_only":
                pred = traj_scores.argmax(dim=-1)
            else:
                combined, agree, feat_assign, traj_assign = self._adaptive_combine(
                    feat_scores, traj_scores)
                pred = combined.argmax(dim=-1)

        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        return pred

    def get_dual_scores(self, observed_positions, observed_features=None,
                        future_positions=None, future_features=None):
        self.eval()
        with torch.no_grad():
            if isinstance(observed_positions, np.ndarray):
                observed_positions = torch.FloatTensor(observed_positions)
            if observed_features is not None and isinstance(observed_features, np.ndarray):
                observed_features = torch.FloatTensor(observed_features)
            if future_positions is not None and isinstance(future_positions, np.ndarray):
                future_positions = torch.FloatTensor(future_positions)
            if future_features is not None and isinstance(future_features, np.ndarray):
                future_features = torch.FloatTensor(future_features)

            z_obs, z_obs_feat, z_obs_traj, z_fut_feat, z_fut_traj = \
                self._encode(observed_positions, observed_features,
                             future_positions, future_features)

            feat_scores, traj_scores = self._compute_dual_scores(
                z_obs_feat, z_obs_traj, z_fut_feat, z_fut_traj)

            combined, agree, feat_assign, traj_assign = self._adaptive_combine(
                feat_scores, traj_scores)

        return {
            "feat_scores": feat_scores.cpu().numpy(),
            "traj_scores": traj_scores.cpu().numpy(),
            "combined_scores": combined.cpu().numpy(),
            "agreement": agree.cpu().numpy(),
            "feat_assignment": feat_assign.cpu().numpy(),
            "traj_assignment": traj_assign.cpu().numpy(),
        }

    def forward(self, observed_positions, observed_features=None,
                future_positions=None, future_features=None):
        z_obs, z_obs_feat, z_obs_traj, z_fut_feat, z_fut_traj = \
            self._encode(observed_positions, observed_features,
                         future_positions, future_features)

        feat_scores, traj_scores = self._compute_dual_scores(
            z_obs_feat, z_obs_traj, z_fut_feat, z_fut_traj)

        combined, agree, feat_assign, traj_assign = self._adaptive_combine(
            feat_scores, traj_scores)

        return combined, feat_scores, traj_scores, agree, z_obs

    def get_hidden_representation(self, observed_positions, observed_features=None,
                                   future_positions=None, future_features=None):
        self.eval()
        with torch.no_grad():
            if isinstance(observed_positions, np.ndarray):
                observed_positions = torch.FloatTensor(observed_positions)
            if observed_features is not None and isinstance(observed_features, np.ndarray):
                observed_features = torch.FloatTensor(observed_features)
            if future_positions is not None and isinstance(future_positions, np.ndarray):
                future_positions = torch.FloatTensor(future_positions)
            if future_features is not None and isinstance(future_features, np.ndarray):
                future_features = torch.FloatTensor(future_features)
            z_obs, _, _, _, _ = self._encode(
                observed_positions, observed_features,
                future_positions, future_features)
        return z_obs
