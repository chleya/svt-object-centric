"""
SVT-v3.1 MLP Position + Feature Model

MLP that takes concatenated positions and features, with:
- Trajectory prediction head
- Identity head (swap logit)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPPositionFeature(nn.Module):
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
        self.identity_head = nn.Linear(hidden_dim, 1)

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
        swap_logit = self.identity_head(shared_out).squeeze(-1)

        return traj_out, swap_logit

    def compute_loss(self, observed_positions, future_positions, identity_labels,
                     observed_features=None):
        traj_pred, swap_logit = self.forward(observed_positions, observed_features)

        if isinstance(future_positions, np.ndarray):
            future_positions = torch.FloatTensor(future_positions)
        if isinstance(identity_labels, np.ndarray):
            identity_labels = torch.FloatTensor(identity_labels)

        mse_loss = F.mse_loss(traj_pred, future_positions)

        target_swap = (identity_labels[:, 0] == 1).float()
        identity_loss = F.binary_cross_entropy_with_logits(swap_logit, target_swap)

        total_loss = mse_loss + self.identity_weight * identity_loss
        return total_loss, mse_loss, identity_loss

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
            _, swap_logit = self.forward(observed_positions, observed_features)
        swap_prob = torch.sigmoid(swap_logit)
        is_swap = swap_prob > 0.5

        B = is_swap.shape[0]
        N = self.num_objects
        ids = np.tile(np.arange(N), (B, 1))
        if isinstance(is_swap, torch.Tensor):
            is_swap = is_swap.cpu().numpy()
        for i in range(B):
            if is_swap[i]:
                ids[i] = np.array([1, 0])
        return ids
