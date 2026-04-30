"""
Recurrent Independent Mechanisms (RIMs) (Goyal et al., 2020)

Adapted for SVT: K independent recurrent modules with input attention.
Each RIM selectively attends to input elements, creating a natural
object-centric decomposition without requiring image inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RIMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, x, h):
        return self.rnn(x, h)


class RIMsModel(nn.Module):
    """
    RIMs adapted for SVT's (position, feature) input format.

    Architecture:
        1. Per-timestep: project (pos, feat) pairs -> d-dim vectors
        2. K independent RIMs, each with input attention over N objects
        3. RIMs process the temporal sequence with selective attention
        4. Trajectory decoder: per-RIM MLP -> predicted future trajectory
        5. Identity head: feature similarity -> assignment logits
    """

    def __init__(
        self,
        t_obs=10,
        t_pred=20,
        num_objects=2,
        dim=2,
        feature_dim=2,
        num_rims=2,
        rim_dim=64,
        hidden_dim=128,
        top_k=1,
        identity_weight=1.0,
        temperature=1.0,
    ):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.num_objects = num_objects
        self.dim = dim
        self.feature_dim = feature_dim
        self.num_rims = num_rims
        self.rim_dim = rim_dim
        self.top_k = top_k
        self.identity_weight = identity_weight
        self.temperature = temperature

        self.input_proj = nn.Linear(dim + feature_dim, rim_dim)

        self.rim_cells = nn.ModuleList([
            RIMCell(rim_dim, rim_dim) for _ in range(num_rims)
        ])

        self.input_attention = nn.ModuleList([
            nn.Linear(rim_dim, 1) for _ in range(num_rims)
        ])

        self.trajectory_decoder = nn.Sequential(
            nn.Linear(rim_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, t_pred * dim),
        )

        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, rim_dim),
            nn.ReLU(),
            nn.Linear(rim_dim, rim_dim),
        )

    def _get_input_attn_mask(self, rim_idx, h_rim, inputs):
        B, N, D = inputs.shape
        scores = self.input_attention[rim_idx](inputs).squeeze(-1)
        if self.top_k < N:
            _, top_indices = scores.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(1, top_indices, True)
            scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.bmm(attn_weights.unsqueeze(1), inputs).squeeze(1)
        return attended

    def forward(self, observed_positions, observed_features=None, future_features=None):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        B, T, N, D = observed_positions.shape

        if observed_features is not None:
            if isinstance(observed_features, np.ndarray):
                observed_features = torch.FloatTensor(observed_features)
                if observed_positions.is_cuda:
                    observed_features = observed_features.to(observed_positions.device)
            x = torch.cat([observed_positions, observed_features], dim=-1)
        else:
            padding = torch.zeros(B, T, N, self.feature_dim, device=observed_positions.device)
            x = torch.cat([observed_positions, padding], dim=-1)

        projected = self.input_proj(x)

        h_states = [torch.zeros(B, self.rim_dim, device=observed_positions.device)
                     for _ in range(self.num_rims)]

        for t in range(T):
            timestep_input = projected[:, t, :, :]
            for i, (rim_cell, h) in enumerate(zip(self.rim_cells, h_states)):
                attended = self._get_input_attn_mask(i, h, timestep_input)
                h_states[i] = rim_cell(attended, h)

        rim_outputs = torch.stack(h_states, dim=1)

        traj_preds = []
        for i in range(self.num_rims):
            traj_i = self.trajectory_decoder(rim_outputs[:, i])
            traj_i = traj_i.reshape(B, self.t_pred, self.dim)
            traj_preds.append(traj_i)
        traj_out = torch.stack(traj_preds, dim=2)

        assignment_logits = self._compute_assignment_logits(observed_features, future_features)

        return traj_out, assignment_logits

    def _compute_assignment_logits(self, observed_features=None, future_features=None):
        N = self.num_objects

        if future_features is None or observed_features is None:
            return None

        if isinstance(observed_features, np.ndarray):
            observed_features = torch.FloatTensor(observed_features)
        if isinstance(future_features, np.ndarray):
            future_features = torch.FloatTensor(future_features)

        obs_pooled = observed_features[:, 0, :, :] if observed_features.dim() == 4 else observed_features
        fut_pooled = future_features[:, 0, :, :] if future_features.dim() == 4 else future_features

        z_obs = self.feature_encoder(obs_pooled)
        z_fut = self.feature_encoder(fut_pooled)

        z_obs_norm = F.normalize(z_obs, dim=-1)
        z_fut_norm = F.normalize(z_fut, dim=-1)

        sim_matrix = torch.bmm(z_fut_norm, z_obs_norm.transpose(1, 2))

        return sim_matrix / self.temperature

    def compute_loss(self, observed_positions, future_positions, identity_labels,
                     observed_features=None, future_features=None):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        if isinstance(future_positions, np.ndarray):
            future_positions = torch.FloatTensor(future_positions)
        if isinstance(identity_labels, np.ndarray):
            identity_labels = torch.LongTensor(identity_labels)
        elif isinstance(identity_labels, torch.Tensor) and identity_labels.dtype != torch.long:
            identity_labels = identity_labels.long()

        traj_pred, assignment_logits = self.forward(
            observed_positions, observed_features, future_features)

        mse_loss = F.mse_loss(traj_pred, future_positions)

        if assignment_logits is not None and future_features is not None:
            assignment_loss = F.cross_entropy(
                assignment_logits.reshape(-1, self.num_objects),
                identity_labels.reshape(-1),
            )
        else:
            assignment_loss = torch.tensor(0.0, device=observed_positions.device)

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
        if assignment_logits is None:
            B = observed_positions.shape[0] if isinstance(observed_positions, torch.Tensor) else len(observed_positions)
            N = self.num_objects
            return np.tile(np.arange(N), (B, 1))
        pred_assignment = assignment_logits.argmax(dim=-1)
        if isinstance(pred_assignment, torch.Tensor):
            pred_assignment = pred_assignment.cpu().numpy()
        return pred_assignment
