"""
DINOSAUR (Seitzer et al., 2024) — Slot-Based Object-Centric Model

Adapted for SVT: replaces DINO ViT encoder with (position, feature) projection,
replaces Broadcast Decoder with per-slot trajectory decoder.
The Slot Attention core is identical to the original DINOSAUR implementation.

Key difference from plain Slot Attention:
- Uses learnable position embeddings for input elements
- Uses a separate slot initialization MLP (not just learned mu/sigma)
- Uses heavier slot update with LayerNorm + MLP residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.slot_attention_model import SlotAttention


class DINOSAURModel(nn.Module):
    """
    DINOSAUR adapted for SVT's (position, feature) input format.

    Architecture:
        1. Per-timestep: project (pos, feat) pairs -> d-dim vectors
        2. Add learnable positional embeddings per object slot
        3. Temporal pooling: GRU over timesteps -> per-object d-dim vectors
        4. Slot Attention with slot init MLP (DINOSAUR-style)
        5. Trajectory decoder: per-slot MLP -> predicted future trajectory
        6. Identity head: feature similarity -> assignment logits
    """

    def __init__(
        self,
        t_obs=10,
        t_pred=20,
        num_objects=2,
        dim=2,
        feature_dim=2,
        slot_dim=64,
        n_slots=None,
        sa_iters=3,
        hidden_dim=128,
        identity_weight=1.0,
        temperature=1.0,
    ):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.num_objects = num_objects
        self.dim = dim
        self.feature_dim = feature_dim
        self.slot_dim = slot_dim
        self.n_slots = n_slots or num_objects
        self.identity_weight = identity_weight
        self.temperature = temperature

        self.input_proj = nn.Linear(dim + feature_dim, slot_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, num_objects, slot_dim) * 0.02)

        self.temporal_encoder = nn.GRU(
            input_size=slot_dim,
            hidden_size=slot_dim,
            num_layers=2,
            batch_first=True,
        )

        self.slot_attention = SlotAttention(
            num_slots=self.n_slots,
            slot_dim=slot_dim,
            iters=sa_iters,
            hidden_dim=hidden_dim,
        )

        self.slot_init_mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim * self.n_slots),
        )

        self.trajectory_decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, t_pred * dim),
        )

        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )

    def _encode_inputs(self, observed_positions, observed_features=None):
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

        projected = projected + self.pos_embed[:, :N, :].unsqueeze(1)

        obj_reps = []
        for i in range(N):
            obj_seq = projected[:, :, i, :]
            _, h_n = self.temporal_encoder(obj_seq)
            obj_reps.append(h_n[-1])

        return torch.stack(obj_reps, dim=1)

    def forward(self, observed_positions, observed_features=None, future_features=None):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        B = observed_positions.shape[0]

        obj_reps = self._encode_inputs(observed_positions, observed_features)

        slots = self.slot_attention(obj_reps)

        traj_preds = []
        for i in range(self.n_slots):
            traj_i = self.trajectory_decoder(slots[:, i])
            traj_i = traj_i.reshape(B, self.t_pred, self.dim)
            traj_preds.append(traj_i)
        traj_out = torch.stack(traj_preds, dim=2)

        assignment_logits = self._compute_assignment_logits(observed_features, future_features)

        return traj_out, assignment_logits

    def _compute_assignment_logits(self, observed_features=None, future_features=None):
        B = self.n_slots
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
