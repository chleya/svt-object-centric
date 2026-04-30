"""
Slot-Based Identity Matching for Published Object-Centric Models

Instead of using feature similarity for identity assignment,
this uses the slot representations directly: encode both observed
and future data into slots, then match by slot similarity.

This tests whether slot representations carry identity information
beyond what features alone provide.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SlotBasedIdentityWrapper(nn.Module):
    """
    Wraps any model that produces slot representations.
    Uses slot similarity (not feature similarity) for identity assignment.

    The model must expose:
        - encode_to_slots(observed_positions, observed_features) -> slots_obs
        - encode_future_to_slots(future_positions, future_features) -> slots_fut
    Or we can use the forward method if it returns slots.
    """

    def __init__(self, base_model, model_name, num_objects=2, feature_dim=2,
                 slot_dim=64, t_obs=10, t_pred=20, identity_weight=1.0,
                 temperature=1.0):
        super().__init__()
        self.base_model = base_model
        self.model_name = model_name
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.slot_dim = slot_dim
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.identity_weight = identity_weight
        self.temperature = temperature

        self.future_input_proj = nn.Linear(2 + feature_dim, slot_dim)

        self.future_temporal_encoder = nn.GRU(
            input_size=slot_dim,
            hidden_size=slot_dim,
            num_layers=2,
            batch_first=True,
        )

        self.slot_match_head = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )

    def encode_obs_slots(self, observed_positions, observed_features=None):
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

        projected = self.base_model.input_proj(x)

        if hasattr(self.base_model, 'pos_embed'):
            projected = projected + self.base_model.pos_embed[:, :N, :].unsqueeze(1)

        obj_reps = []
        for i in range(N):
            obj_seq = projected[:, :, i, :]
            _, h_n = self.base_model.temporal_encoder(obj_seq)
            obj_reps.append(h_n[-1])
        obj_reps = torch.stack(obj_reps, dim=1)

        slots = self.base_model.slot_attention(obj_reps)
        return slots

    def encode_fut_slots(self, future_positions, future_features=None):
        if isinstance(future_positions, np.ndarray):
            future_positions = torch.FloatTensor(future_positions)
        B, T, N, D = future_positions.shape

        if future_features is not None:
            if isinstance(future_features, np.ndarray):
                future_features = torch.FloatTensor(future_features)
                if future_positions.is_cuda:
                    future_features = future_features.to(future_positions.device)
            if future_features.dim() == 4 and future_features.shape[1] != T:
                step = max(1, future_features.shape[1] // T)
                indices = torch.arange(0, future_features.shape[1], step)[:T]
                future_features = future_features[:, indices, :, :]
            x = torch.cat([future_positions, future_features], dim=-1)
        else:
            padding = torch.zeros(B, T, N, self.feature_dim, device=future_positions.device)
            x = torch.cat([future_positions, padding], dim=-1)

        projected = self.future_input_proj(x)

        obj_reps = []
        for i in range(N):
            obj_seq = projected[:, :, i, :]
            _, h_n = self.future_temporal_encoder(obj_seq)
            obj_reps.append(h_n[-1])
        obj_reps = torch.stack(obj_reps, dim=1)

        slots = self.base_model.slot_attention(obj_reps)
        return slots

    def compute_slot_assignment_logits(self, obs_slots, fut_slots):
        z_obs = self.slot_match_head(obs_slots)
        z_fut = self.slot_match_head(fut_slots)

        z_obs_norm = F.normalize(z_obs, dim=-1)
        z_fut_norm = F.normalize(z_fut, dim=-1)

        sim_matrix = torch.bmm(z_fut_norm, z_obs_norm.transpose(1, 2))
        return sim_matrix / self.temperature

    def forward(self, observed_positions, observed_features=None,
                future_positions=None, future_features=None):
        obs_slots = self.encode_obs_slots(observed_positions, observed_features)

        traj_preds = []
        for i in range(self.base_model.n_slots):
            traj_i = self.base_model.trajectory_decoder(obs_slots[:, i])
            traj_i = traj_i.reshape(observed_positions.shape[0], self.t_pred, 2)
            traj_preds.append(traj_i)
        traj_out = torch.stack(traj_preds, dim=2)

        slot_logits = None
        if future_positions is not None:
            fut_slots = self.encode_fut_slots(future_positions, future_features)
            slot_logits = self.compute_slot_assignment_logits(obs_slots, fut_slots)

        feature_logits = None
        if future_features is not None and observed_features is not None:
            feature_logits = self.base_model._compute_assignment_logits(
                observed_features, future_features)

        return traj_out, slot_logits, feature_logits

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

        traj_pred, slot_logits, feature_logits = self.forward(
            observed_positions, observed_features, future_positions, future_features)

        mse_loss = F.mse_loss(traj_pred, future_positions)

        slot_loss = torch.tensor(0.0, device=observed_positions.device)
        if slot_logits is not None:
            slot_loss = F.cross_entropy(
                slot_logits.reshape(-1, self.num_objects),
                identity_labels.reshape(-1),
            )

        feature_loss = torch.tensor(0.0, device=observed_positions.device)
        if feature_logits is not None:
            feature_loss = F.cross_entropy(
                feature_logits.reshape(-1, self.num_objects),
                identity_labels.reshape(-1),
            )

        total_loss = mse_loss + self.identity_weight * (slot_loss + feature_loss)
        return total_loss, mse_loss, slot_loss, feature_loss

    def predict_future(self, observed_positions, observed_features=None):
        self.eval()
        with torch.no_grad():
            obs_slots = self.encode_obs_slots(observed_positions, observed_features)
            traj_preds = []
            for i in range(self.base_model.n_slots):
                traj_i = self.base_model.trajectory_decoder(obs_slots[:, i])
                traj_i = traj_i.reshape(observed_positions.shape[0] if isinstance(observed_positions, torch.Tensor) else len(observed_positions), self.t_pred, 2)
                traj_preds.append(traj_i)
            traj_out = torch.stack(traj_preds, dim=2)
        if isinstance(observed_positions, np.ndarray):
            return traj_out.cpu().numpy()
        return traj_out

    def predict_identity_slot(self, observed_positions, observed_features=None,
                               future_positions=None, future_features=None):
        self.eval()
        with torch.no_grad():
            _, slot_logits, _ = self.forward(
                observed_positions, observed_features, future_positions, future_features)
        if slot_logits is None:
            B = observed_positions.shape[0] if isinstance(observed_positions, torch.Tensor) else len(observed_positions)
            return np.tile(np.arange(self.num_objects), (B, 1))
        pred = slot_logits.argmax(dim=-1)
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        return pred

    def predict_identity_feature(self, observed_positions, observed_features=None,
                                  future_features=None):
        self.eval()
        with torch.no_grad():
            feature_logits = self.base_model._compute_assignment_logits(
                observed_features, future_features)
        if feature_logits is None:
            B = observed_positions.shape[0] if isinstance(observed_positions, torch.Tensor) else len(observed_positions)
            return np.tile(np.arange(self.num_objects), (B, 1))
        pred = feature_logits.argmax(dim=-1)
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        return pred

    def predict_identity(self, observed_positions, observed_features=None,
                         test_future=None, future_features=None,
                         future_positions=None, method="slot"):
        if method == "slot":
            return self.predict_identity_slot(
                observed_positions, observed_features,
                future_positions=future_positions or test_future,
                future_features=future_features)
        elif method == "feature":
            return self.predict_identity_feature(
                observed_positions, observed_features, future_features=future_features)
        else:
            raise ValueError(f"Unknown method: {method}")
