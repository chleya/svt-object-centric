"""
Dual-Pathway Wrapper for Published Object-Centric Models

v18 finding: DualPathwayObjectFile achieves conditional binding (0.912).
v18d question: Can the same principle be applied to PUBLISHED models?

Key insight from v18: The dual-pathway principle requires:
  1. A feature-based identity scorer (already exists in published models)
  2. A trajectory-based identity scorer (needs to be added)
  3. Agreement-based switching (simple logic)

This wrapper adds a trajectory scorer to any published model and
combines predictions using agreement-based switching, exactly
like DualPathwayObjectFile but as a retrofit.

If this works, it demonstrates that the dual-pathway principle
is general and can be applied to ANY object-centric model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TrajectoryIdentityHead(nn.Module):
    def __init__(self, dim=2, slot_dim=64, hidden_dim=128, t_obs=10, num_objects=2,
                 use_proximity=True):
        super().__init__()
        self.num_objects = num_objects
        self.use_proximity = use_proximity
        self.obs_traj_encoder = nn.GRU(input_size=dim, hidden_size=slot_dim,
                                        num_layers=2, batch_first=True)
        self.fut_traj_encoder = nn.GRU(input_size=dim, hidden_size=slot_dim,
                                        num_layers=2, batch_first=True)
        self.scorer = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1))
        if use_proximity:
            self.prox_scorer = nn.Sequential(
                nn.Linear(slot_dim * 2 + 1, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1))
            self.prox_gate = nn.Sequential(
                nn.Linear(1, hidden_dim // 4), nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1), nn.Sigmoid())

    def _compute_min_distances(self, positions):
        B = positions.shape[0]
        N = positions.shape[2]
        min_dists = torch.full((B, N), 1e6, device=positions.device)
        for j in range(N):
            for k in range(N):
                if j == k:
                    continue
                dist = torch.sqrt(
                    ((positions[:, :, j, :] - positions[:, :, k, :]) ** 2).sum(dim=-1) + 1e-8)
                min_dists[:, j] = torch.min(min_dists[:, j], dist.min(dim=1)[0])
        return min_dists

    def forward(self, observed_positions, future_positions):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        if isinstance(future_positions, np.ndarray):
            future_positions = torch.FloatTensor(future_positions)
        B = observed_positions.shape[0]
        N = self.num_objects

        z_obs_list, z_fut_list = [], []
        for j in range(N):
            _, h_obs = self.obs_traj_encoder(observed_positions[:, :, j, :])
            z_obs_list.append(h_obs[-1])
            _, h_fut = self.fut_traj_encoder(future_positions[:, :, j, :])
            z_fut_list.append(h_fut[-1])
        z_obs = torch.stack(z_obs_list, dim=1)
        z_fut = torch.stack(z_fut_list, dim=1)

        scores = torch.zeros(B, N, N, device=z_obs.device)
        obs_dists = self._compute_min_distances(observed_positions) if self.use_proximity else None
        fut_dists = self._compute_min_distances(future_positions) if self.use_proximity else None

        for i in range(N):
            for j in range(N):
                pair = torch.cat([z_fut[:, i, :], z_obs[:, j, :]], dim=-1)
                base_score = self.scorer(pair).squeeze(-1)

                if self.use_proximity and obs_dists is not None and fut_dists is not None:
                    avg_dist = (obs_dists[:, j] + fut_dists[:, i]) / 2.0
                    norm_dist = avg_dist.unsqueeze(-1) / 64.0
                    prox_pair = torch.cat([pair, norm_dist], dim=-1)
                    enhanced_score = self.prox_scorer(prox_pair).squeeze(-1)
                    gate = self.prox_gate(norm_dist).squeeze(-1)
                    scores[:, i, j] = gate * enhanced_score + (1 - gate) * base_score
                else:
                    scores[:, i, j] = base_score
        return scores


class DualPathwayWrapper(nn.Module):
    def __init__(self, base_model, dim=2, slot_dim=64, hidden_dim=128,
                 t_obs=10, num_objects=2, conflict_switch_temp=0.1,
                 identity_weight=1.0, traj_identity_weight=1.0,
                 traj_weight=0.1, use_proximity=True):
        super().__init__()
        self.base_model = base_model
        self.num_objects = num_objects
        self.dim = dim
        self.t_obs = t_obs
        self.conflict_switch_temp = conflict_switch_temp
        self.identity_weight = identity_weight
        self.traj_identity_weight = traj_identity_weight
        self.traj_weight = traj_weight

        self.traj_identity_head = TrajectoryIdentityHead(
            dim, slot_dim, hidden_dim, t_obs, num_objects,
            use_proximity=use_proximity)

    def _get_feature_scores(self, observed_positions, observed_features,
                            future_positions, future_features):
        self.base_model.eval()
        with torch.no_grad():
            try:
                result = self.base_model(observed_positions, observed_features, future_features)
                if isinstance(result, tuple) and len(result) >= 2:
                    assignment_logits = result[1]
                else:
                    assignment_logits = result
            except TypeError:
                try:
                    result = self.base_model(observed_positions, observed_features)
                    if isinstance(result, tuple) and len(result) >= 2:
                        assignment_logits = result[1]
                    else:
                        assignment_logits = result
                except TypeError:
                    assignment_logits = None

        if assignment_logits is None:
            B = observed_positions.shape[0] if isinstance(observed_positions, torch.Tensor) else len(observed_positions)
            return torch.zeros(B, self.num_objects, self.num_objects)

        if isinstance(assignment_logits, np.ndarray):
            assignment_logits = torch.FloatTensor(assignment_logits)

        return assignment_logits

    def _adaptive_combine(self, feat_scores, traj_scores):
        feat_assign = feat_scores.argmax(dim=-1)
        traj_assign = traj_scores.argmax(dim=-1)

        agree = (feat_assign == traj_assign).all(dim=-1).float()

        temp = self.conflict_switch_temp
        soft_agree = torch.sigmoid((agree - 0.5) / max(temp, 1e-6))

        combined = soft_agree.unsqueeze(-1).unsqueeze(-1) * feat_scores + \
                   (1.0 - soft_agree.unsqueeze(-1).unsqueeze(-1)) * traj_scores

        return combined, agree

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

        base_loss, mse_loss, assign_loss, _ = self.base_model.compute_loss(
            observed_positions, future_positions, feat_identity,
            observed_features, aug_fut_feat)

        traj_scores = self.traj_identity_head(observed_positions, future_positions)
        traj_id_loss = F.cross_entropy(traj_scores.reshape(-1, N), traj_identity.reshape(-1))

        feat_scores = self._get_feature_scores(observed_positions, observed_features,
                                                future_positions, aug_fut_feat)
        combined, agree = self._adaptive_combine(feat_scores, traj_scores)

        combined_identity = torch.where(
            agree.unsqueeze(-1).bool().expand_as(identity_labels),
            feat_identity, traj_identity)
        combined_loss = F.cross_entropy(combined.reshape(-1, N), combined_identity.reshape(-1))

        total_loss = (self.identity_weight * (0.3 * assign_loss + 0.3 * traj_id_loss + 0.4 * combined_loss) +
                      self.traj_weight * mse_loss)

        return total_loss, assign_loss, traj_id_loss, combined_loss

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
                    20, self.num_objects, self.dim)

            if isinstance(future_positions, np.ndarray):
                future_positions = torch.FloatTensor(future_positions)

            feat_scores = self._get_feature_scores(
                observed_positions, observed_features, future_positions, future_features)
            traj_scores = self.traj_identity_head(observed_positions, future_positions)

            if method == "feature_only":
                pred = feat_scores.argmax(dim=-1)
            elif method == "trajectory_only":
                pred = traj_scores.argmax(dim=-1)
            else:
                combined, agree = self._adaptive_combine(feat_scores, traj_scores)
                pred = combined.argmax(dim=-1)

        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        return pred

    def forward(self, observed_positions, observed_features=None,
                future_positions=None, future_features=None):
        if future_positions is None:
            future_positions = torch.zeros(
                observed_positions.shape[0] if isinstance(observed_positions, torch.Tensor) else len(observed_positions),
                20, self.num_objects, self.dim)

        feat_scores = self._get_feature_scores(
            observed_positions, observed_features, future_positions, future_features)
        traj_scores = self.traj_identity_head(observed_positions, future_positions)
        combined, agree = self._adaptive_combine(feat_scores, traj_scores)

        return combined, feat_scores, traj_scores, agree

    def get_hidden_representation(self, observed_positions, observed_features=None,
                                   future_positions=None, future_features=None):
        self.eval()
        with torch.no_grad():
            if isinstance(observed_positions, np.ndarray):
                observed_positions = torch.FloatTensor(observed_positions)
            if future_positions is None:
                future_positions = torch.zeros(
                    observed_positions.shape[0], 20, self.num_objects, self.dim)
            if isinstance(future_positions, np.ndarray):
                future_positions = torch.FloatTensor(future_positions)
            z = self.traj_identity_head(observed_positions, future_positions)
        return z
