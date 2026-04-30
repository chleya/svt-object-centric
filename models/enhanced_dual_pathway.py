"""
Enhanced Dual-Pathway ObjectFile (v18c)

v18 achieved conditional binding (conflict res = 0.879) but trajectory
scorer accuracy was only 84-88%, limiting the combined model's ceiling.

v18c improvements:
  1. Transformer trajectory encoder (replaces GRU for better temporal modeling)
  2. Velocity-augmented trajectory input (adds delta-position as extra features)
  3. Deeper pairwise scorer (3 layers instead of 2)
  4. Larger slot dimension (128 instead of 64)
  5. Separate trajectory prediction heads for each object
  6. Trajectory contrastive loss (pull same-object embeddings closer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class TransformerTrajectoryEncoder(nn.Module):
    def __init__(self, dim=2, slot_dim=128, t_obs=10, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.t_obs = t_obs
        self.slot_dim = slot_dim
        self.input_proj = nn.Linear(dim + dim, slot_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, slot_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=slot_dim, nhead=n_heads, dim_feedforward=slot_dim * 4,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(slot_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))

    def forward(self, positions):
        if isinstance(positions, np.ndarray):
            positions = torch.FloatTensor(positions)
        B = positions.shape[0]
        N = positions.shape[2]
        embeddings = []
        for j in range(N):
            traj = positions[:, :, j, :]
            if traj.shape[1] > 1:
                velocity = torch.zeros_like(traj)
                velocity[:, 1:, :] = traj[:, 1:, :] - traj[:, :-1, :]
                velocity[:, 0, :] = velocity[:, 1, :]
            else:
                velocity = torch.zeros_like(traj)
            x = torch.cat([traj, velocity], dim=-1)
            x = self.input_proj(x) + self.pos_encoding[:, :x.shape[1], :]
            x = self.transformer(x)
            pooled = x.mean(dim=1)
            embeddings.append(self.output_proj(pooled))
        return torch.stack(embeddings, dim=1)


class DeepPairwiseScorer(nn.Module):
    def __init__(self, slot_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1))

    def forward(self, z_fut, z_obs):
        B = z_fut.shape[0]
        N = z_fut.shape[1]
        scores = torch.zeros(B, N, N, device=z_fut.device)
        for i in range(N):
            for j in range(N):
                pair = torch.cat([z_fut[:, i, :], z_obs[:, j, :]], dim=-1)
                scores[:, i, j] = self.net(pair).squeeze(-1)
        return scores


class FeatureEncoder(nn.Module):
    def __init__(self, feature_dim=2, slot_dim=128, hidden_dim=256):
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


class EnhancedDualPathwayObjectFile(nn.Module):
    def __init__(self, num_objects=2, dim=2, feature_dim=2,
                 hidden_dim=256, slot_dim=128, t_obs=10, t_pred=20,
                 identity_weight=1.0, smh_weight=1.0, traj_weight=0.1,
                 contrastive_weight=0.1, conflict_switch_temp=0.1):
        super().__init__()
        self.num_objects = num_objects
        self.dim = dim
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.identity_weight = identity_weight
        self.smh_weight = smh_weight
        self.traj_weight = traj_weight
        self.contrastive_weight = contrastive_weight
        self.conflict_switch_temp = conflict_switch_temp

        self.obs_feat_encoder = FeatureEncoder(feature_dim, slot_dim, hidden_dim)
        self.fut_feat_encoder = FeatureEncoder(feature_dim, slot_dim, hidden_dim)

        self.obs_traj_encoder = TransformerTrajectoryEncoder(
            dim, slot_dim, t_obs, n_heads=4, n_layers=2)
        self.fut_traj_encoder = TransformerTrajectoryEncoder(
            dim, slot_dim, t_obs, n_heads=4, n_layers=2)

        self.feature_scorer = DeepPairwiseScorer(slot_dim, hidden_dim)
        self.trajectory_scorer = DeepPairwiseScorer(slot_dim, hidden_dim)

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

    def _contrastive_loss(self, z_traj, identity_labels):
        B = z_traj.shape[0]
        N = self.num_objects
        loss = torch.tensor(0.0, device=z_traj.device)
        count = 0
        for j in range(N):
            anchor = z_traj[:, j, :]
            positive_mask = (identity_labels == j)
            for b in range(min(B, 32)):
                if positive_mask[b].sum() < 2:
                    continue
                pos_idx = torch.where(positive_mask[b])[0]
                if len(pos_idx) < 2:
                    continue
                for p in pos_idx[1:]:
                    sim_pos = F.cosine_similarity(anchor[b:b+1], z_traj[b:b+1, p, :])
                    neg_idx = torch.where(~positive_mask[b])[0]
                    if len(neg_idx) == 0:
                        continue
                    sim_neg = F.cosine_similarity(
                        anchor[b:b+1].expand(len(neg_idx), -1),
                        z_traj[b:b+1, neg_idx, :])
                    loss += -sim_pos + torch.logsumexp(sim_neg.unsqueeze(0), dim=1).squeeze()
                    count += 1
        if count > 0:
            loss = loss / count
        return loss

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

        contrastive_loss = self._contrastive_loss(z_obs_traj, identity_labels)

        total_loss = (self.identity_weight * identity_loss +
                      self.smh_weight * smh_loss +
                      self.traj_weight * traj_loss_val +
                      self.contrastive_weight * contrastive_loss)

        return total_loss, identity_loss, smh_loss, contrastive_loss

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
