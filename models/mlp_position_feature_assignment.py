"""
SVT-v3.3 MLP Position + Feature with Assignment Head

Identity head predicts N×N assignment matrix instead of binary swap.
assignment_logits[b, i, j] = logit that future slot i maps to observed object j.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPPositionFeatureAssignment(nn.Module):
    def __init__(self, t_obs=10, t_pred=20, num_objects=2, feature_dim=2,
                 hidden_dim=256, num_layers=3, dropout=0.1, identity_weight=1.0):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.identity_weight = identity_weight

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
        self.assignment_head = nn.Linear(hidden_dim, num_objects * num_objects)

    def forward(self, observed_positions, observed_features=None):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        B = observed_positions.shape[0]

        if observed_features is not None:
            if isinstance(observed_features, np.ndarray):
                observed_features = torch.FloatTensor(observed_features)
            x = torch.cat([observed_positions, observed_features], dim=-1)
        else:
            padding = torch.zeros(B, self.t_obs, self.num_objects, self.feature_dim)
            x = torch.cat([observed_positions, padding], dim=-1)

        x = x.reshape(B, -1)
        shared_out = self.shared(x)

        traj_out = self.trajectory_head(shared_out).reshape(B, self.t_pred, self.num_objects, 2)
        assignment_logits = self.assignment_head(shared_out).reshape(B, self.num_objects, self.num_objects)

        return traj_out, assignment_logits

    def compute_loss(self, observed_positions, future_positions, identity_labels,
                     observed_features=None):
        traj_pred, assignment_logits = self.forward(observed_positions, observed_features)

        if isinstance(future_positions, np.ndarray):
            future_positions = torch.FloatTensor(future_positions)
        if isinstance(identity_labels, np.ndarray):
            identity_labels = torch.LongTensor(identity_labels)
        elif isinstance(identity_labels, torch.Tensor) and identity_labels.dtype != torch.long:
            identity_labels = identity_labels.long()

        mse_loss = F.mse_loss(traj_pred, future_positions)

        assignment_loss = F.cross_entropy(
            assignment_logits.reshape(-1, self.num_objects),
            identity_labels.reshape(-1),
        )

        total_loss = mse_loss + self.identity_weight * assignment_loss
        return total_loss, mse_loss, assignment_loss

    def predict_future(self, observed_positions, observed_features=None):
        self.eval()
        with torch.no_grad():
            traj_pred, _ = self.forward(observed_positions, observed_features)
        if isinstance(observed_positions, np.ndarray):
            return traj_pred.numpy()
        return traj_pred

    def predict_identity(self, observed_positions, observed_features=None, test_future=None):
        self.eval()
        with torch.no_grad():
            _, assignment_logits = self.forward(observed_positions, observed_features)
        pred_assignment = assignment_logits.argmax(dim=-1)
        if isinstance(pred_assignment, torch.Tensor):
            pred_assignment = pred_assignment.cpu().numpy()
        return pred_assignment
