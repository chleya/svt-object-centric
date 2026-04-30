"""
Conflict-Augmented Learned ObjectFile (Direction F)

Key insight from Direction D's failure: the gating network always chose
the feature channel because it never experienced conflict during training.

Solution: augment training with conflict scenarios by randomly flipping
future features. This forces the gating network to learn when features
are unreliable and trajectory should be used instead.

Training strategy:
    1. With probability p_conflict: flip future features -> create conflict
    2. With probability p_drop: zero out features -> force trajectory usage
    3. With probability (1 - p_conflict - p_drop): normal training

This is essentially curriculum learning for conflict resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.learned_object_file import (
    FeatureChannel, TrajectoryChannel,
    ConflictDetector, GatingNetwork,
)


class ConflictAugmentedLearnedObjectFile(nn.Module):
    def __init__(self, num_objects=2, dim=2, feature_dim=2,
                 hidden_dim=128, slot_dim=64,
                 t_obs=10, t_pred=20,
                 identity_weight=1.0, conflict_weight=0.5,
                 channel_aux_weight=0.3, temperature=1.0,
                 p_conflict=0.3, p_feature_drop=0.1):
        super().__init__()
        self.num_objects = num_objects
        self.dim = dim
        self.feature_dim = feature_dim
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.identity_weight = identity_weight
        self.conflict_weight = conflict_weight
        self.channel_aux_weight = channel_aux_weight
        self.temperature = temperature
        self.p_conflict = p_conflict
        self.p_feature_drop = p_feature_drop

        self.feature_channel = FeatureChannel(
            feature_dim=feature_dim, num_objects=num_objects,
            hidden_dim=hidden_dim, slot_dim=slot_dim)

        self.trajectory_channel = TrajectoryChannel(
            num_objects=num_objects, dim=dim,
            hidden_dim=hidden_dim, slot_dim=slot_dim,
            t_obs=t_obs, t_pred=t_pred)

        self.conflict_detector = ConflictDetector(
            num_objects=num_objects, hidden_dim=hidden_dim // 2)

        self.gating_network = GatingNetwork(
            num_objects=num_objects, hidden_dim=hidden_dim // 2)

    def _augment_features(self, fut_feat, p_conflict, p_drop):
        B, N = fut_feat.shape[0], self.num_objects
        aug_fut_feat = fut_feat.clone()
        conflict_labels = torch.zeros(B, device=fut_feat.device)
        drop_labels = torch.zeros(B, device=fut_feat.device)

        for b in range(B):
            r = torch.rand(1).item()
            if r < p_conflict:
                if N >= 2:
                    perm = torch.randperm(N)
                    while (perm == torch.arange(N)).all():
                        perm = torch.randperm(N)
                    aug_fut_feat[b] = fut_feat[b, perm]
                    conflict_labels[b] = 1.0
            elif r < p_conflict + p_drop:
                aug_fut_feat[b] = 0.0
                drop_labels[b] = 1.0

        return aug_fut_feat, conflict_labels, drop_labels

    def forward(self, observed_positions, observed_features=None,
                future_positions=None, future_features=None):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        if future_positions is None:
            future_positions = torch.zeros(
                observed_positions.shape[0], self.t_pred, self.num_objects, self.dim)
        if isinstance(future_positions, np.ndarray):
            future_positions = torch.FloatTensor(future_positions)

        B = observed_positions.shape[0]

        traj_logits = self.trajectory_channel(observed_positions, future_positions)

        feat_logits = torch.zeros(B, self.num_objects, self.num_objects,
                                   device=observed_positions.device)
        if observed_features is not None and future_features is not None:
            if isinstance(observed_features, np.ndarray):
                observed_features = torch.FloatTensor(observed_features)
            if isinstance(future_features, np.ndarray):
                future_features = torch.FloatTensor(future_features)

            obs_pooled = observed_features[:, 0, :, :] if observed_features.dim() == 4 else observed_features
            fut_pooled = future_features[:, 0, :, :] if future_features.dim() == 4 else future_features

            feat_logits = self.feature_channel(obs_pooled, fut_pooled)

        conflict_prob = self.conflict_detector(feat_logits, traj_logits)

        gate_weights = self.gating_network(feat_logits, traj_logits, conflict_prob)

        feat_probs = F.softmax(feat_logits / self.temperature, dim=-1)
        traj_probs = F.softmax(traj_logits / self.temperature, dim=-1)

        combined_probs = gate_weights * feat_probs + (1 - gate_weights) * traj_probs
        combined_logits = torch.log(combined_probs + 1e-8)

        return combined_logits, feat_logits, traj_logits, conflict_prob, gate_weights

    def compute_loss(self, observed_positions, future_positions, identity_labels,
                     observed_features=None, future_features=None,
                     is_swap=None):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        if isinstance(future_positions, np.ndarray):
            future_positions = torch.FloatTensor(future_positions)
        if isinstance(identity_labels, np.ndarray):
            identity_labels = torch.LongTensor(identity_labels)
        elif isinstance(identity_labels, torch.Tensor) and identity_labels.dtype != torch.long:
            identity_labels = identity_labels.long()

        if observed_features is not None and future_features is not None:
            if isinstance(observed_features, np.ndarray):
                observed_features = torch.FloatTensor(observed_features)
            if isinstance(future_features, np.ndarray):
                future_features = torch.FloatTensor(future_features)

            obs_pooled = observed_features[:, 0, :, :] if observed_features.dim() == 4 else observed_features
            fut_pooled = future_features[:, 0, :, :] if future_features.dim() == 4 else future_features

            aug_fut, conflict_labels, drop_labels = self._augment_features(
                fut_pooled, self.p_conflict, self.p_feature_drop)

            N = self.num_objects
            B = identity_labels.shape[0]
            traj_identity = torch.arange(N, device=identity_labels.device).unsqueeze(0).expand(B, -1)

            combined_logits, feat_logits, traj_logits, conflict_prob, gate_weights = self.forward(
                observed_positions, observed_features, future_positions, aug_fut)
        else:
            N = self.num_objects
            traj_identity = torch.arange(N, device=identity_labels.device).unsqueeze(0).expand(identity_labels.shape[0], -1)

            combined_logits, feat_logits, traj_logits, conflict_prob, gate_weights = self.forward(
                observed_positions, observed_features, future_positions, future_features)
            conflict_labels = torch.zeros(identity_labels.shape[0], device=identity_labels.device)

        combined_loss = F.cross_entropy(
            combined_logits.reshape(-1, N), identity_labels.reshape(-1))

        feat_aux_loss = F.cross_entropy(
            feat_logits.reshape(-1, N), identity_labels.reshape(-1))

        traj_aux_loss = F.cross_entropy(
            traj_logits.reshape(-1, N), traj_identity.reshape(-1))

        traj_pred = self.trajectory_channel.predict_trajectory(observed_positions)
        mse_loss = F.mse_loss(traj_pred, future_positions)

        conflict_loss = F.binary_cross_entropy(conflict_prob, conflict_labels)

        gate_traj_weight = (1 - gate_weights).mean()
        gate_balance_loss = (gate_traj_weight - 0.3).abs()

        total_loss = (self.identity_weight * combined_loss +
                      self.channel_aux_weight * feat_aux_loss +
                      self.channel_aux_weight * 2.0 * traj_aux_loss +
                      0.1 * mse_loss +
                      self.conflict_weight * conflict_loss +
                      0.1 * gate_balance_loss)

        return total_loss, combined_loss, feat_aux_loss, traj_aux_loss

    def predict_future(self, observed_positions, observed_features=None):
        self.eval()
        with torch.no_grad():
            if isinstance(observed_positions, np.ndarray):
                observed_positions = torch.FloatTensor(observed_positions)
            traj_pred = self.trajectory_channel.predict_trajectory(observed_positions)
        if isinstance(observed_positions, np.ndarray):
            return traj_pred.cpu().numpy()
        return traj_pred

    def predict_identity(self, observed_positions, observed_features=None,
                         test_future=None, future_features=None,
                         future_positions=None, method="combined"):
        self.eval()
        with torch.no_grad():
            if future_positions is None and test_future is not None:
                future_positions = test_future
            if future_positions is None:
                future_positions = np.zeros(
                    (len(observed_positions), self.t_pred, self.num_objects, self.dim),
                    dtype=np.float32) if isinstance(observed_positions, np.ndarray) else \
                    torch.zeros(observed_positions.shape[0], self.t_pred, self.num_objects, self.dim)

            combined_logits, feat_logits, traj_logits, conflict_prob, gate_weights = self.forward(
                observed_positions, observed_features, future_positions, future_features)

            if method == "combined":
                logits = combined_logits
            elif method == "feature":
                logits = feat_logits
            elif method == "trajectory":
                logits = traj_logits
            else:
                logits = combined_logits

            pred_assignment = logits.argmax(dim=-1)

        if isinstance(pred_assignment, torch.Tensor):
            pred_assignment = pred_assignment.cpu().numpy()
        return pred_assignment
