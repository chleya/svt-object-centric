"""
SVT-v3.1 Object-Centric Feature Model

Per-object encoding with shared MLP, trajectory head, and identity head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ObjectCentricFeatureModel(nn.Module):
    def __init__(self, t_obs=10, t_pred=20, num_objects=2, feature_dim=2,
                 obj_hidden_dim=128, shared_hidden_dim=256, identity_weight=1.0):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.identity_weight = identity_weight

        self.obj_input_dim = t_obs * (2 + feature_dim)
        self.obj_encoder = nn.Sequential(
            nn.Linear(self.obj_input_dim, obj_hidden_dim),
            nn.ReLU(),
            nn.Linear(obj_hidden_dim, obj_hidden_dim),
            nn.ReLU(),
        )

        self.obj_emb_dim = obj_hidden_dim
        self.combined_dim = num_objects * obj_hidden_dim

        self.trajectory_decoder = nn.Sequential(
            nn.Linear(obj_hidden_dim, obj_hidden_dim),
            nn.ReLU(),
            nn.Linear(obj_hidden_dim, t_pred * 2),
        )

        self.identity_head = nn.Sequential(
            nn.Linear(self.combined_dim, shared_hidden_dim),
            nn.ReLU(),
            nn.Linear(shared_hidden_dim, 1),
        )

    def encode_objects(self, observed_positions, observed_features=None):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        B = observed_positions.shape[0]

        if observed_features is not None:
            if isinstance(observed_features, np.ndarray):
                observed_features = torch.FloatTensor(observed_features)
            x = torch.cat([observed_positions, observed_features], dim=-1)
        else:
            padding = torch.zeros(B, self.t_obs, self.num_objects, self.feature_dim)
            if observed_positions.is_cuda:
                padding = padding.to(observed_positions.device)
            x = torch.cat([observed_positions, padding], dim=-1)

        obj_embs = []
        for i in range(self.num_objects):
            obj_input = x[:, :, i, :].reshape(B, -1)
            emb = self.obj_encoder(obj_input)
            obj_embs.append(emb)

        return torch.stack(obj_embs, dim=1)

    def forward(self, observed_positions, observed_features=None):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        B = observed_positions.shape[0]

        obj_embs = self.encode_objects(observed_positions, observed_features)

        traj_preds = []
        for i in range(self.num_objects):
            traj_i = self.trajectory_decoder(obj_embs[:, i])
            traj_i = traj_i.reshape(B, self.t_pred, 2)
            traj_preds.append(traj_i)
        traj_out = torch.stack(traj_preds, dim=2)

        combined = obj_embs.reshape(B, -1)
        swap_logit = self.identity_head(combined).squeeze(-1)

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
