"""
SVT-v3.6 Feature-Similarity Assignment Head Models (Temporal-Aligned)

FeatureOnlyAssignmentHead: assignment purely from feature cosine similarity
HybridTrajectoryFeatureAssignmentHead: traj_logits + beta * feature_logits

v3.6 fix: use first-timestep pooling instead of mean pooling.
v3.5.1 proved mean pooling destroys swap-pre identity info (75% ceiling).
obs=first achieves oracle 100%.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _pool_time(features):
    if features is None:
        return None
    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)
    if features.dim() == 4:
        features = features[:, 0, :, :]
    return features


def _to_long(labels):
    if isinstance(labels, np.ndarray):
        return torch.LongTensor(labels)
    if isinstance(labels, torch.Tensor) and labels.dtype != torch.long:
        return labels.long()
    return labels


def _to_float(t, device=None):
    if isinstance(t, np.ndarray):
        t = torch.FloatTensor(t)
    if device is not None and isinstance(t, torch.Tensor):
        t = t.to(device)
    return t


def compute_feature_similarity_logits(feature_encoder, observed_features, future_features, num_objects, temperature=1.0):
    if observed_features is None or future_features is None:
        return None

    obs_pooled = _pool_time(observed_features)
    fut_pooled = _pool_time(future_features)

    z_obs = feature_encoder(obs_pooled)
    z_fut = feature_encoder(fut_pooled)

    z_obs_norm = F.normalize(z_obs, dim=-1)
    z_fut_norm = F.normalize(z_fut, dim=-1)

    sim_matrix = torch.bmm(z_fut_norm, z_obs_norm.transpose(1, 2))

    return sim_matrix / temperature


class FeatureOnlyAssignmentHead(nn.Module):
    def __init__(self, t_obs=10, t_pred=20, num_objects=2, feature_dim=2,
                 hidden_dim=256, num_layers=3, dropout=0.1,
                 identity_weight=1.0, feat_emb_dim=64, temperature=1.0):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.identity_weight = identity_weight
        self.temperature = temperature

        self.input_dim = t_obs * num_objects * (2 + feature_dim)
        self.output_dim = t_pred * num_objects * 2

        layers = []
        in_dim = self.input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.shared = nn.Sequential(*layers)

        self.trajectory_head = nn.Linear(hidden_dim, self.output_dim)

        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, feat_emb_dim),
            nn.ReLU(),
            nn.Linear(feat_emb_dim, feat_emb_dim),
        )

    def forward(self, observed_positions, observed_features=None, future_features=None):
        observed_positions = _to_float(observed_positions)
        B = observed_positions.shape[0]

        if observed_features is not None:
            observed_features = _to_float(observed_features)
            x = torch.cat([observed_positions, observed_features], dim=-1)
        else:
            padding = torch.zeros(B, self.t_obs, self.num_objects, self.feature_dim,
                                  device=observed_positions.device)
            x = torch.cat([observed_positions, padding], dim=-1)

        x = x.reshape(B, -1)
        shared_out = self.shared(x)

        traj_out = self.trajectory_head(shared_out).reshape(B, self.t_pred, self.num_objects, 2)

        feature_logits = compute_feature_similarity_logits(
            self.feature_encoder, observed_features, future_features,
            self.num_objects, self.temperature)

        if feature_logits is not None:
            assignment_logits = feature_logits
        else:
            assignment_logits = torch.zeros(B, self.num_objects, self.num_objects,
                                            device=observed_positions.device)

        return traj_out, assignment_logits

    def compute_loss(self, observed_positions, future_positions, identity_labels,
                     observed_features=None, future_features=None):
        traj_pred, assignment_logits = self.forward(
            observed_positions, observed_features, future_features)

        future_positions = _to_float(future_positions)
        identity_labels = _to_long(identity_labels)

        mse_loss = F.mse_loss(traj_pred, future_positions)

        assignment_loss = F.cross_entropy(
            assignment_logits.reshape(-1, self.num_objects),
            identity_labels.reshape(-1),
        )

        total_loss = mse_loss + self.identity_weight * assignment_loss
        return total_loss, mse_loss, assignment_loss, torch.tensor(0.0)

    def predict_future(self, observed_positions, observed_features=None):
        self.eval()
        with torch.no_grad():
            traj_pred, _ = self.forward(observed_positions, observed_features)
        if isinstance(observed_positions, np.ndarray):
            return traj_pred.cpu().numpy()
        return traj_pred

    def predict_identity(self, observed_positions, observed_features=None,
                         test_future=None, future_features=None):
        self.eval()
        with torch.no_grad():
            _, assignment_logits = self.forward(
                observed_positions, observed_features, future_features)
        pred_assignment = assignment_logits.argmax(dim=-1)
        if isinstance(pred_assignment, torch.Tensor):
            pred_assignment = pred_assignment.cpu().numpy()
        return pred_assignment


class HybridTrajectoryFeatureAssignmentHead(nn.Module):
    def __init__(self, t_obs=10, t_pred=20, num_objects=2, feature_dim=2,
                 hidden_dim=256, num_layers=3, dropout=0.1,
                 identity_weight=1.0, beta=1.0, feat_emb_dim=64, temperature=1.0):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.identity_weight = identity_weight
        self.beta = beta
        self.temperature = temperature

        self.input_dim = t_obs * num_objects * (2 + feature_dim)
        self.output_dim = t_pred * num_objects * 2

        layers = []
        in_dim = self.input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.shared = nn.Sequential(*layers)

        self.trajectory_head = nn.Linear(hidden_dim, self.output_dim)
        self.traj_assignment_head = nn.Linear(hidden_dim, num_objects * num_objects)

        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, feat_emb_dim),
            nn.ReLU(),
            nn.Linear(feat_emb_dim, feat_emb_dim),
        )

    def forward(self, observed_positions, observed_features=None, future_features=None):
        observed_positions = _to_float(observed_positions)
        B = observed_positions.shape[0]

        if observed_features is not None:
            observed_features = _to_float(observed_features)
            x = torch.cat([observed_positions, observed_features], dim=-1)
        else:
            padding = torch.zeros(B, self.t_obs, self.num_objects, self.feature_dim,
                                  device=observed_positions.device)
            x = torch.cat([observed_positions, padding], dim=-1)

        x = x.reshape(B, -1)
        shared_out = self.shared(x)

        traj_out = self.trajectory_head(shared_out).reshape(B, self.t_pred, self.num_objects, 2)
        traj_logits = self.traj_assignment_head(shared_out).reshape(B, self.num_objects, self.num_objects)

        feature_logits = compute_feature_similarity_logits(
            self.feature_encoder, observed_features, future_features,
            self.num_objects, self.temperature)

        if feature_logits is not None:
            assignment_logits = traj_logits + self.beta * feature_logits
        else:
            assignment_logits = traj_logits

        return traj_out, assignment_logits

    def compute_loss(self, observed_positions, future_positions, identity_labels,
                     observed_features=None, future_features=None):
        traj_pred, assignment_logits = self.forward(
            observed_positions, observed_features, future_features)

        future_positions = _to_float(future_positions)
        identity_labels = _to_long(identity_labels)

        mse_loss = F.mse_loss(traj_pred, future_positions)

        assignment_loss = F.cross_entropy(
            assignment_logits.reshape(-1, self.num_objects),
            identity_labels.reshape(-1),
        )

        total_loss = mse_loss + self.identity_weight * assignment_loss
        return total_loss, mse_loss, assignment_loss, torch.tensor(0.0)

    def predict_future(self, observed_positions, observed_features=None):
        self.eval()
        with torch.no_grad():
            traj_pred, _ = self.forward(observed_positions, observed_features)
        if isinstance(observed_positions, np.ndarray):
            return traj_pred.cpu().numpy()
        return traj_pred

    def predict_identity(self, observed_positions, observed_features=None,
                         test_future=None, future_features=None):
        self.eval()
        with torch.no_grad():
            _, assignment_logits = self.forward(
                observed_positions, observed_features, future_features)
        pred_assignment = assignment_logits.argmax(dim=-1)
        if isinstance(pred_assignment, torch.Tensor):
            pred_assignment = pred_assignment.cpu().numpy()
        return pred_assignment
