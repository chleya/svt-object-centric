"""
Gated Graph-Structured ObjectFile

v16 finding: S4 + conflict training is not sufficient — softmax edge
networks cannot learn conditional weight modulation because feature signal
dominates.

v17 solution: Independent conflict detector that GATES edge weights.
  - Conflict detector: separate network that measures feature-trajectory disagreement
  - Gate: when conflict is detected, override feature edge weight and boost trajectory edge weight
  - This is the ConflictFirst gate idea implemented at the graph level

Key architectural difference from GraphObjectFile:
  - GraphObjectFile: edge weights = softmax(relation_scorer) → always dominated by feature
  - GatedGraphObjectFile: edge weights = softmax(relation_scorer) * gate(conflict_signal)
    → gate can override feature-dominated weights when conflict is detected
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ObjectNodeEncoder(nn.Module):
    def __init__(self, feature_dim=2, dim=2, slot_dim=64, hidden_dim=128, t_obs=10):
        super().__init__()
        self.t_obs = t_obs
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, slot_dim))
        self.traj_encoder = nn.GRU(input_size=dim, hidden_size=slot_dim, num_layers=2, batch_first=True)
        self.node_update = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, slot_dim))

    def forward(self, positions, features=None):
        if isinstance(positions, np.ndarray):
            positions = torch.FloatTensor(positions)
        if features is not None and isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        B = positions.shape[0]
        N = positions.shape[2]
        feat_pooled = features[:, 0, :, :] if (features is not None and features.dim() == 4) else features
        z_feat_list, z_traj_list = [], []
        for j in range(N):
            obs_feat_j = feat_pooled[:, j, :] if feat_pooled is not None else torch.zeros(B, self.feature_encoder[0].in_features, device=positions.device)
            z_feat_list.append(self.feature_encoder(obs_feat_j))
            _, h_traj = self.traj_encoder(positions[:, :, j, :])
            z_traj_list.append(h_traj[-1])
        z_feat = torch.stack(z_feat_list, dim=1)
        z_traj = torch.stack(z_traj_list, dim=1)
        z_node = self.node_update(torch.cat([z_feat, z_traj], dim=-1))
        return z_node, z_feat, z_traj


class ConflictDetector(nn.Module):
    def __init__(self, slot_dim=64, hidden_dim=64):
        super().__init__()
        self.feature_match = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.trajectory_match = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, z_fut_feat, z_obs_feat, z_fut_traj, z_obs_traj):
        B = z_fut_feat.shape[0]
        N = z_fut_feat.shape[1]

        feat_scores = torch.zeros(B, N, device=z_fut_feat.device)
        traj_scores = torch.zeros(B, N, device=z_fut_feat.device)

        for i in range(N):
            for j in range(N):
                feat_s = torch.sigmoid(self.feature_match(
                    torch.cat([z_fut_feat[:, i, :], z_obs_feat[:, j, :]], dim=-1))).squeeze(-1)
                traj_s = torch.sigmoid(self.trajectory_match(
                    torch.cat([z_fut_traj[:, i, :], z_obs_traj[:, j, :]], dim=-1))).squeeze(-1)
                if j < N:
                    feat_scores[:, i] += feat_s
                    traj_scores[:, i] += traj_s

        feat_scores = feat_scores / N
        traj_scores = traj_scores / N

        disagreement = (feat_scores - traj_scores).abs()
        conflict_signal = torch.sigmoid(disagreement * 5.0)

        return conflict_signal, feat_scores, traj_scores


class GatedGraphObjectFile(nn.Module):
    def __init__(self, num_objects=2, dim=2, feature_dim=2,
                 hidden_dim=128, slot_dim=64, t_obs=10, t_pred=20,
                 n_relation_types=3, identity_weight=1.0, smh_weight=1.0,
                 traj_weight=0.1, conflict_gate_weight=0.5):
        super().__init__()
        self.num_objects = num_objects
        self.dim = dim
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.identity_weight = identity_weight
        self.smh_weight = smh_weight
        self.traj_weight = traj_weight
        self.conflict_gate_weight = conflict_gate_weight

        self.obs_encoder = ObjectNodeEncoder(feature_dim, dim, slot_dim, hidden_dim, t_obs)
        self.fut_encoder = ObjectNodeEncoder(feature_dim, dim, slot_dim, hidden_dim, t_obs)

        self.edge_scorer = nn.Sequential(
            nn.Linear(slot_dim * 4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_relation_types))
        self.edge_value = nn.Sequential(
            nn.Linear(slot_dim * 4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, slot_dim))

        self.conflict_detector = ConflictDetector(slot_dim, hidden_dim // 2)

        self.message_aggregator = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, slot_dim))
        self.assignment_head = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.smh = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, num_objects))
        self.traj_decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, t_pred * dim))

    def _encode_inputs(self, observed_positions, observed_features, future_positions, future_features):
        z_obs, z_obs_feat, z_obs_traj = self.obs_encoder(observed_positions, observed_features)
        z_fut, z_fut_feat, z_fut_traj = self.fut_encoder(future_positions, future_features)
        return z_obs, z_obs_feat, z_obs_traj, z_fut, z_fut_feat, z_fut_traj

    def _gated_graph_pass(self, z_obs, z_obs_feat, z_obs_traj, z_fut, z_fut_feat, z_fut_traj):
        B = z_obs.shape[0]
        N = self.num_objects

        conflict_signal, feat_scores, traj_scores = self.conflict_detector(
            z_fut_feat, z_obs_feat, z_fut_traj, z_obs_traj)

        edge_weights = torch.zeros(B, N, N, self.edge_scorer[-1].out_features, device=z_obs.device)
        edge_messages = torch.zeros(B, N, N, z_obs.shape[-1], device=z_obs.device)

        for i in range(N):
            for j in range(N):
                pair = torch.cat([z_fut[:, i, :], z_obs[:, j, :],
                                  z_fut_feat[:, i, :] - z_obs_feat[:, j, :],
                                  z_fut_traj[:, i, :] - z_obs_traj[:, j, :]], dim=-1)
                w = F.softmax(self.edge_scorer(pair), dim=-1)

                gate_val = conflict_signal[:, i]
                w_gated = w.clone()
                if w_gated.shape[-1] >= 3:
                    w_gated[:, 0] = w[:, 0] * (1.0 - gate_val)
                    w_gated[:, 1] = w[:, 1] * (1.0 + gate_val)
                    w_gated[:, 2] = w[:, 2] * (1.0 + gate_val * 2.0)
                    w_gated = F.softmax(w_gated + 1e-8, dim=-1)

                v = self.edge_value(pair)
                edge_weights[:, i, j] = w_gated
                edge_messages[:, i, j] = v

        aggregated_fut = torch.zeros_like(z_fut)
        for i in range(N):
            msg = torch.zeros(B, z_fut.shape[-1], device=z_fut.device)
            for j in range(N):
                w_sum = edge_weights[:, i, j, :].sum(dim=-1, keepdim=True)
                msg = msg + w_sum * edge_messages[:, i, j]
            aggregated_fut[:, i, :] = self.message_aggregator(
                torch.cat([z_fut[:, i, :], msg], dim=-1))

        logits = torch.zeros(B, N, N, device=z_obs.device)
        for i in range(N):
            for j in range(N):
                pair = torch.cat([aggregated_fut[:, i, :], z_obs[:, j, :]], dim=-1)
                logits[:, i, j] = self.assignment_head(pair).squeeze(-1)

        return logits, edge_weights, aggregated_fut, conflict_signal

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
        aug_identity = identity_labels
        conflict_labels = torch.zeros(B, device=observed_positions.device)

        if p_conflict > 0 and future_features is not None and N >= 2:
            aug_fut_feat = future_features.clone()
            aug_identity = identity_labels.clone()
            for b in range(B):
                if torch.rand(1).item() < p_conflict:
                    if aug_fut_feat.dim() == 4:
                        aug_fut_feat[b, :, 0, :], aug_fut_feat[b, :, 1, :] = \
                            future_features[b, :, 1, :].clone(), future_features[b, :, 0, :].clone()
                    elif aug_fut_feat.dim() == 3:
                        aug_fut_feat[b, 0, :], aug_fut_feat[b, 1, :] = \
                            future_features[b, 1, :].clone(), future_features[b, 0, :].clone()
                    aug_identity[b, 0], aug_identity[b, 1] = identity_labels[b, 1].clone(), identity_labels[b, 0].clone()
                    conflict_labels[b] = 1.0

        z_obs, z_obs_feat, z_obs_traj, z_fut, z_fut_feat, z_fut_traj = \
            self._encode_inputs(observed_positions, observed_features, future_positions, aug_fut_feat)

        logits, edge_weights, aggregated_fut, conflict_signal = self._gated_graph_pass(
            z_obs, z_obs_feat, z_obs_traj, z_fut, z_fut_feat, z_fut_traj)

        identity_loss = F.cross_entropy(logits.reshape(-1, N), aug_identity.reshape(-1))

        smh_logits = torch.zeros(B, N, N, device=z_obs.device)
        for j in range(N):
            smh_logits[:, j, :] = self.smh(z_obs[:, j, :])
        smh_loss = F.cross_entropy(smh_logits.reshape(-1, N), identity_labels.reshape(-1))

        traj_preds = []
        for j in range(N):
            traj_j = self.traj_decoder(z_obs[:, j, :])
            traj_preds.append(traj_j.reshape(B, self.t_pred, self.dim))
        pred_traj = torch.stack(traj_preds, dim=2)
        traj_loss = F.mse_loss(pred_traj, future_positions)

        conflict_gate_loss = torch.tensor(0.0, device=observed_positions.device)
        if p_conflict > 0 and conflict_labels.sum() > 0:
            conflict_mask = conflict_labels.bool()
            if conflict_mask.any():
                conflict_conf = conflict_signal[conflict_mask].mean()
                clean_conf = conflict_signal[~conflict_mask].mean() if (~conflict_mask).any() else torch.tensor(0.0)
                conflict_gate_loss = -0.5 * (conflict_conf - clean_conf)

        total_loss = (self.identity_weight * identity_loss +
                      self.smh_weight * smh_loss +
                      self.traj_weight * traj_loss +
                      self.conflict_gate_weight * conflict_gate_loss)

        return total_loss, identity_loss, smh_loss, conflict_gate_loss

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

            z_obs, z_obs_feat, z_obs_traj, z_fut, z_fut_feat, z_fut_traj = \
                self._encode_inputs(observed_positions, observed_features, future_positions, future_features)
            logits, _, _, _ = self._gated_graph_pass(
                z_obs, z_obs_feat, z_obs_traj, z_fut, z_fut_feat, z_fut_traj)
            pred = logits.argmax(dim=-1)
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        return pred

    def forward(self, observed_positions, observed_features=None,
                future_positions=None, future_features=None):
        z_obs, z_obs_feat, z_obs_traj, z_fut, z_fut_feat, z_fut_traj = \
            self._encode_inputs(observed_positions, observed_features, future_positions, future_features)
        logits, edge_weights, aggregated_fut, conflict_signal = self._gated_graph_pass(
            z_obs, z_obs_feat, z_obs_traj, z_fut, z_fut_feat, z_fut_traj)
        return logits, edge_weights, aggregated_fut, conflict_signal

    def get_edge_weights(self, observed_positions, observed_features=None,
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
            z_obs, z_obs_feat, z_obs_traj, z_fut, z_fut_feat, z_fut_traj = \
                self._encode_inputs(observed_positions, observed_features, future_positions, future_features)
            _, edge_weights, _, conflict_signal = self._gated_graph_pass(
                z_obs, z_obs_feat, z_obs_traj, z_fut, z_fut_feat, z_fut_traj)
        return edge_weights.cpu().numpy(), conflict_signal.cpu().numpy()

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
            z_obs, _, _, _, _, _ = self._encode_inputs(
                observed_positions, observed_features, future_positions, future_features)
        return z_obs
