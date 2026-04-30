"""
Probabilistic Structure Selection ObjectFile (Direction G)

Inspired by Probe-to-Boundary's probabilistic structure selection:
pi_i = sigma(aR + bS + cC + dG - eSpurious)

Instead of hard gating (choose feature OR trajectory), we:
1. Compute per-object binding probabilities for both channels
2. Use a probabilistic mixture based on readability/stability/causality scores
3. Allow the model to "hedge" rather than commit to one channel
4. Use delta-output trajectory prediction (re-anchor to last frame)

Key insight: The trade-off exists because we force hard decisions.
Probabilistic mixing might find a better operating point.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DeltaTrajectoryChannel(nn.Module):
    def __init__(self, num_objects=2, dim=2, hidden_dim=128, slot_dim=64,
                 t_obs=10, t_pred=20):
        super().__init__()
        self.num_objects = num_objects
        self.dim = dim
        self.t_obs = t_obs
        self.t_pred = t_pred

        self.obs_encoder = nn.GRU(
            input_size=dim, hidden_size=slot_dim,
            num_layers=2, batch_first=True)

        self.delta_decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, t_pred * dim),
        )

        self.velocity_encoder = nn.Sequential(
            nn.Linear(dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )

        self.binding_net = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_pos, fut_pos):
        B = obs_pos.shape[0]
        N = self.num_objects

        z_obs_list = []
        last_vel_list = []
        for j in range(N):
            obs_j = obs_pos[:, :, j, :]
            _, h_j = self.obs_encoder(obs_j)
            z_obs_list.append(h_j[-1])
            last_vel_list.append(obs_pos[:, -1, j, :] - obs_pos[:, -2, j, :])
        z_obs = torch.stack(z_obs_list, dim=1)

        delta_preds = []
        for j in range(N):
            delta_j = self.delta_decoder(z_obs[:, j, :])
            delta_j = delta_j.reshape(B, self.t_pred, self.dim)
            delta_preds.append(delta_j)
        delta_out = torch.stack(delta_preds, dim=2)

        last_pos = obs_pos[:, -1, :, :].unsqueeze(1)
        pred_traj = last_pos + delta_out.cumsum(dim=1)

        z_fut_list = []
        for i in range(N):
            fut_i = fut_pos[:, :, i, :]
            _, h_i = self.obs_encoder(fut_i)
            z_fut_list.append(h_i[-1])
        z_fut = torch.stack(z_fut_list, dim=1)

        logits = torch.zeros(B, N, N, device=obs_pos.device)
        for i in range(N):
            for j in range(N):
                pair = torch.cat([z_fut[:, i, :], z_obs[:, j, :]], dim=-1)
                logits[:, i, j] = self.binding_net(pair).squeeze(-1)

        return logits, pred_traj

    def predict_trajectory(self, obs_pos):
        B = obs_pos.shape[0]
        N = self.num_objects

        z_obs_list = []
        for j in range(N):
            obs_j = obs_pos[:, :, j, :]
            _, h_j = self.obs_encoder(obs_j)
            z_obs_list.append(h_j[-1])
        z_obs = torch.stack(z_obs_list, dim=1)

        delta_preds = []
        for j in range(N):
            delta_j = self.delta_decoder(z_obs[:, j, :])
            delta_j = delta_j.reshape(B, self.t_pred, self.dim)
            delta_preds.append(delta_j)
        delta_out = torch.stack(delta_preds, dim=2)

        last_pos = obs_pos[:, -1, :, :].unsqueeze(1)
        pred_traj = last_pos + delta_out.cumsum(dim=1)
        return pred_traj


class ProbabilisticStructureObjectFile(nn.Module):
    """
    ObjectFile with probabilistic structure selection.

    Instead of hard gating, computes per-binding structure scores:
    - Readability (R): how well the binding can be read from representation
    - Stability (S): how robust the binding is to perturbation
    - Causality (C): whether the binding is actually used by the model

    Binding probability: pi = sigma(aR + bS + cC + dG)
    where G is a learned gate signal.
    """

    def __init__(self, num_objects=2, dim=2, feature_dim=2,
                 hidden_dim=128, slot_dim=64,
                 t_obs=10, t_pred=20,
                 identity_weight=1.0, structure_weight=0.5,
                 p_conflict=0.3, p_feature_drop=0.1):
        super().__init__()
        self.num_objects = num_objects
        self.dim = dim
        self.feature_dim = feature_dim
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.identity_weight = identity_weight
        self.structure_weight = structure_weight
        self.p_conflict = p_conflict
        self.p_feature_drop = p_feature_drop

        self.feature_channel = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )

        self.traj_channel = DeltaTrajectoryChannel(
            num_objects=num_objects, dim=dim,
            hidden_dim=hidden_dim, slot_dim=slot_dim,
            t_obs=t_obs, t_pred=t_pred)

        self.feat_binding = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.structure_scorer = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.a_readability = nn.Parameter(torch.tensor(1.0))
        self.b_stability = nn.Parameter(torch.tensor(1.0))
        self.c_causality = nn.Parameter(torch.tensor(1.0))
        self.d_gate = nn.Parameter(torch.tensor(0.0))

    def _compute_feature_logits(self, obs_feat, fut_feat):
        B = obs_feat.shape[0]
        N = self.num_objects

        z_obs = self.feature_channel(obs_feat)
        z_fut = self.feature_channel(fut_feat)

        logits = torch.zeros(B, N, N, device=obs_feat.device)
        for i in range(N):
            for j in range(N):
                pair = torch.cat([z_fut[:, i, :], z_obs[:, j, :]], dim=-1)
                logits[:, i, j] = self.feat_binding(pair).squeeze(-1)

        return logits

    def _compute_structure_scores(self, feat_logits, traj_logits):
        B = feat_logits.shape[0]
        N = self.num_objects

        feat_probs = F.softmax(feat_logits, dim=-1)
        traj_probs = F.softmax(traj_logits, dim=-1)

        readability = (feat_probs.max(dim=-1)[0] + traj_probs.max(dim=-1)[0]) / 2

        feat_noise = feat_logits + torch.randn_like(feat_logits) * 0.1
        traj_noise = traj_logits + torch.randn_like(traj_logits) * 0.1
        feat_probs_n = F.softmax(feat_noise, dim=-1)
        traj_probs_n = F.softmax(traj_noise, dim=-1)
        stability = 1.0 - (feat_probs - feat_probs_n).abs().mean(dim=-1) - \
                    (traj_probs - traj_probs_n).abs().mean(dim=-1)
        stability = torch.clamp(stability, 0, 1)

        agreement = (feat_logits.argmax(dim=-1) == traj_logits.argmax(dim=-1)).float()
        causality = agreement

        gate_signal = torch.sigmoid(self.structure_scorer(
            torch.cat([readability.unsqueeze(-1),
                       stability.unsqueeze(-1),
                       causality.unsqueeze(-1),
                       (1 - agreement).unsqueeze(-1)], dim=-1)
        )).squeeze(-1)

        pi = torch.sigmoid(
            self.a_readability * readability +
            self.b_stability * stability +
            self.c_causality * causality +
            self.d_gate * gate_signal
        )

        return pi

    def _augment_features(self, fut_feat):
        B, N = fut_feat.shape[0], self.num_objects
        aug = fut_feat.clone()
        conflict_labels = torch.zeros(B, device=fut_feat.device)

        for b in range(B):
            r = torch.rand(1).item()
            if r < self.p_conflict:
                if N >= 2:
                    perm = torch.randperm(N)
                    while (perm == torch.arange(N)).all():
                        perm = torch.randperm(N)
                    aug[b] = fut_feat[b, perm]
                    conflict_labels[b] = 1.0
            elif r < self.p_conflict + self.p_feature_drop:
                aug[b] = 0.0

        return aug, conflict_labels

    def forward(self, observed_positions, observed_features=None,
                future_positions=None, future_features=None):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        if isinstance(future_positions, np.ndarray):
            future_positions = torch.FloatTensor(future_positions)

        B = observed_positions.shape[0]
        N = self.num_objects

        traj_logits, pred_traj = self.traj_channel(observed_positions, future_positions)

        feat_logits = torch.zeros(B, N, N, device=observed_positions.device)
        if observed_features is not None and future_features is not None:
            if isinstance(observed_features, np.ndarray):
                observed_features = torch.FloatTensor(observed_features)
            if isinstance(future_features, np.ndarray):
                future_features = torch.FloatTensor(future_features)
            obs_pooled = observed_features[:, 0, :, :] if observed_features.dim() == 4 else observed_features
            fut_pooled = future_features[:, 0, :, :] if future_features.dim() == 4 else future_features
            feat_logits = self._compute_feature_logits(obs_pooled, fut_pooled)

        pi = self._compute_structure_scores(feat_logits, traj_logits)

        feat_probs = F.softmax(feat_logits, dim=-1)
        traj_probs = F.softmax(traj_logits, dim=-1)

        pi_expanded = pi.unsqueeze(-1)
        combined_probs = pi_expanded * feat_probs + (1 - pi_expanded) * traj_probs
        combined_logits = torch.log(combined_probs + 1e-8)

        return combined_logits, feat_logits, traj_logits, pi, pred_traj

    def compute_loss(self, observed_positions, future_positions, identity_labels,
                     observed_features=None, future_features=None, is_swap=None):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        if isinstance(future_positions, np.ndarray):
            future_positions = torch.FloatTensor(future_positions)
        if isinstance(identity_labels, np.ndarray):
            identity_labels = torch.LongTensor(identity_labels)
        elif isinstance(identity_labels, torch.Tensor) and identity_labels.dtype != torch.long:
            identity_labels = identity_labels.long()

        N = self.num_objects
        traj_identity = torch.arange(N, device=identity_labels.device).unsqueeze(0).expand(identity_labels.shape[0], -1)

        if observed_features is not None and future_features is not None:
            if isinstance(observed_features, np.ndarray):
                observed_features = torch.FloatTensor(observed_features)
            if isinstance(future_features, np.ndarray):
                future_features = torch.FloatTensor(future_features)

            obs_pooled = observed_features[:, 0, :, :] if observed_features.dim() == 4 else observed_features
            fut_pooled = future_features[:, 0, :, :] if future_features.dim() == 4 else future_features

            aug_fut, conflict_labels = self._augment_features(fut_pooled)

            combined_logits, feat_logits, traj_logits, pi, pred_traj = self.forward(
                observed_positions, observed_features, future_positions, aug_fut)
        else:
            conflict_labels = torch.zeros(identity_labels.shape[0], device=identity_labels.device)
            combined_logits, feat_logits, traj_logits, pi, pred_traj = self.forward(
                observed_positions, observed_features, future_positions, future_features)

        combined_loss = F.cross_entropy(
            combined_logits.reshape(-1, N), identity_labels.reshape(-1))

        feat_aux_loss = F.cross_entropy(
            feat_logits.reshape(-1, N), identity_labels.reshape(-1))

        traj_aux_loss = F.cross_entropy(
            traj_logits.reshape(-1, N), traj_identity.reshape(-1))

        mse_loss = F.mse_loss(pred_traj, future_positions)

        structure_reg = (pi - 0.5).abs().mean()

        total_loss = (self.identity_weight * combined_loss +
                      0.3 * feat_aux_loss +
                      0.6 * traj_aux_loss +
                      0.1 * mse_loss +
                      self.structure_weight * structure_reg)

        return total_loss, combined_loss, feat_aux_loss, traj_aux_loss

    def predict_future(self, observed_positions, observed_features=None):
        self.eval()
        with torch.no_grad():
            if isinstance(observed_positions, np.ndarray):
                observed_positions = torch.FloatTensor(observed_positions)
            pred_traj = self.traj_channel.predict_trajectory(observed_positions)
        if isinstance(observed_positions, np.ndarray):
            return pred_traj.cpu().numpy()
        return pred_traj

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

            combined_logits, feat_logits, traj_logits, pi, _ = self.forward(
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
        if isinstance(pi, torch.Tensor):
            pi = pi.cpu().numpy()
        return pred_assignment, pi
