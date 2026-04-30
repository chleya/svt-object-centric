"""
Learned ObjectFile with Structural Inductive Bias (Direction D)

A neural network that has the structural biases of ObjectFile
(separate feature/trajectory channels + conflict detection)
but learns update rules from data.

Architecture:
    1. Feature Channel: encodes observed/future features -> feature identity logits
    2. Trajectory Channel: encodes observed positions + predicted trajectory -> trajectory identity logits
    3. Conflict Detector: compares feature and trajectory logits -> conflict probability
    4. Gating Network: given conflict probability, combines feature and trajectory logits
    5. Auxiliary losses: ensure both channels learn useful signals independently
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureChannel(nn.Module):
    def __init__(self, feature_dim=2, num_objects=2, hidden_dim=128, slot_dim=64):
        super().__init__()
        self.num_objects = num_objects
        self.obs_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        self.fut_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        self.logit_net = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_feat, fut_feat):
        B = obs_feat.shape[0]
        N = self.num_objects

        z_obs = self.obs_encoder(obs_feat)
        z_fut = self.fut_encoder(fut_feat)

        logits = torch.zeros(B, N, N, device=obs_feat.device)
        for i in range(N):
            for j in range(N):
                pair = torch.cat([z_fut[:, i, :], z_obs[:, j, :]], dim=-1)
                logits[:, i, j] = self.logit_net(pair).squeeze(-1)

        return logits


class TrajectoryChannel(nn.Module):
    def __init__(self, num_objects=2, dim=2, hidden_dim=128, slot_dim=64,
                 t_obs=10, t_pred=20):
        super().__init__()
        self.num_objects = num_objects
        self.dim = dim
        self.t_obs = t_obs
        self.t_pred = t_pred

        self.obs_encoder = nn.GRU(
            input_size=dim,
            hidden_size=slot_dim,
            num_layers=2,
            batch_first=True,
        )

        self.fut_encoder = nn.GRU(
            input_size=dim,
            hidden_size=slot_dim,
            num_layers=2,
            batch_first=True,
        )

        self.logit_net = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.trajectory_decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, t_pred * dim),
        )

    def forward(self, obs_pos, fut_pos):
        B = obs_pos.shape[0]
        N = self.num_objects

        z_obs_list = []
        for j in range(N):
            obs_j = obs_pos[:, :, j, :]
            _, h_j = self.obs_encoder(obs_j)
            z_obs_list.append(h_j[-1])
        z_obs = torch.stack(z_obs_list, dim=1)

        z_fut_list = []
        for i in range(N):
            fut_i = fut_pos[:, :, i, :]
            _, h_i = self.fut_encoder(fut_i)
            z_fut_list.append(h_i[-1])
        z_fut = torch.stack(z_fut_list, dim=1)

        logits = torch.zeros(B, N, N, device=obs_pos.device)
        for i in range(N):
            for j in range(N):
                pair = torch.cat([z_fut[:, i, :], z_obs[:, j, :]], dim=-1)
                logits[:, i, j] = self.logit_net(pair).squeeze(-1)

        return logits

    def predict_trajectory(self, obs_pos):
        B = obs_pos.shape[0]
        N = self.num_objects

        z_obs_list = []
        for j in range(N):
            obs_j = obs_pos[:, :, j, :]
            _, h_j = self.obs_encoder(obs_j)
            z_obs_list.append(h_j[-1])
        z_obs = torch.stack(z_obs_list, dim=1)

        traj_preds = []
        for j in range(N):
            traj_j = self.trajectory_decoder(z_obs[:, j, :])
            traj_preds.append(traj_j.reshape(B, self.t_pred, self.dim))
        return torch.stack(traj_preds, dim=2)


class ConflictDetector(nn.Module):
    def __init__(self, num_objects=2, hidden_dim=64):
        super().__init__()
        self.num_objects = num_objects

        self.detector = nn.Sequential(
            nn.Linear(num_objects * num_objects * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, feat_logits, traj_logits):
        B = feat_logits.shape[0]
        N = self.num_objects

        feat_probs = F.softmax(feat_logits, dim=-1)
        traj_probs = F.softmax(traj_logits, dim=-1)

        combined = torch.cat([
            feat_probs.reshape(B, -1),
            traj_probs.reshape(B, -1),
        ], dim=-1)

        conflict_logit = self.detector(combined)
        conflict_prob = torch.sigmoid(conflict_logit).squeeze(-1)

        return conflict_prob


class GatingNetwork(nn.Module):
    def __init__(self, num_objects=2, hidden_dim=64):
        super().__init__()
        self.num_objects = num_objects

        self.gate = nn.Sequential(
            nn.Linear(num_objects * num_objects * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objects * num_objects),
        )

    def forward(self, feat_logits, traj_logits, conflict_prob):
        B = feat_logits.shape[0]

        feat_probs = F.softmax(feat_logits, dim=-1)
        traj_probs = F.softmax(traj_logits, dim=-1)

        context = torch.cat([
            feat_probs.reshape(B, -1),
            traj_probs.reshape(B, -1),
            conflict_prob.unsqueeze(-1),
        ], dim=-1)

        gate_weights = torch.sigmoid(self.gate(context))
        gate_weights = gate_weights.reshape(B, self.num_objects, self.num_objects)

        return gate_weights


class LearnedObjectFile(nn.Module):
    """
    Learned ObjectFile with structural inductive bias.

    Key structural biases:
    1. Separate feature and trajectory processing channels
    2. Explicit conflict detection
    3. Learned gating based on conflict signal
    4. Auxiliary losses to prevent channel collapse

    This should learn better update rules than hand-crafted ones
    while maintaining the structural properties that make ObjectFile work.
    """

    def __init__(self, num_objects=2, dim=2, feature_dim=2,
                 hidden_dim=128, slot_dim=64,
                 t_obs=10, t_pred=20,
                 identity_weight=1.0, conflict_weight=0.5,
                 channel_aux_weight=0.3, temperature=1.0):
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

        combined_logits, feat_logits, traj_logits, conflict_prob, gate_weights = self.forward(
            observed_positions, observed_features, future_positions, future_features)

        N = self.num_objects
        combined_loss = F.cross_entropy(
            combined_logits.reshape(-1, N), identity_labels.reshape(-1))

        feat_aux_loss = F.cross_entropy(
            feat_logits.reshape(-1, N), identity_labels.reshape(-1))

        traj_identity = torch.arange(N, device=identity_labels.device).unsqueeze(0).expand(identity_labels.shape[0], -1)
        traj_aux_loss = F.cross_entropy(
            traj_logits.reshape(-1, N), traj_identity.reshape(-1))

        traj_pred = self.trajectory_channel.predict_trajectory(observed_positions)
        mse_loss = F.mse_loss(traj_pred, future_positions)

        conflict_target = torch.zeros(observed_positions.shape[0], device=observed_positions.device)
        if is_swap is not None:
            if isinstance(is_swap, np.ndarray):
                is_swap = torch.FloatTensor(is_swap)
            feat_pred = feat_logits.argmax(dim=-1)
            traj_pred_id = traj_logits.argmax(dim=-1)
            actual_conflict = (feat_pred != traj_pred_id).any(dim=-1).float()
            conflict_target = actual_conflict

        conflict_loss = F.binary_cross_entropy(conflict_prob, conflict_target)

        total_loss = (self.identity_weight * combined_loss +
                      self.channel_aux_weight * feat_aux_loss +
                      self.channel_aux_weight * traj_aux_loss +
                      0.1 * mse_loss +
                      self.conflict_weight * conflict_loss)

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

    def predict_identity_with_info(self, observed_positions, observed_features=None,
                                    future_positions=None, future_features=None):
        self.eval()
        with torch.no_grad():
            combined_logits, feat_logits, traj_logits, conflict_prob, gate_weights = self.forward(
                observed_positions, observed_features, future_positions, future_features)

            pred_assignment = combined_logits.argmax(dim=-1)
            feat_assignment = feat_logits.argmax(dim=-1)
            traj_assignment = traj_logits.argmax(dim=-1)

        results = {
            'combined': pred_assignment.cpu().numpy() if isinstance(pred_assignment, torch.Tensor) else pred_assignment,
            'feature': feat_assignment.cpu().numpy() if isinstance(feat_assignment, torch.Tensor) else feat_assignment,
            'trajectory': traj_assignment.cpu().numpy() if isinstance(traj_assignment, torch.Tensor) else traj_assignment,
            'conflict_prob': conflict_prob.cpu().numpy() if isinstance(conflict_prob, torch.Tensor) else conflict_prob,
            'gate_weights': gate_weights.cpu().numpy() if isinstance(gate_weights, torch.Tensor) else gate_weights,
        }
        return results
