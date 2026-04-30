"""
Counterfactual ObjectFile v2 — Fixed Counterfactual Training

v13 bug: sensitivity/counterfactual losses were applied to ALL samples,
including swap episodes where identity was already flipped. This created
conflicting gradients that destroyed identity encoding.

v14 fix: Only apply counterfactual pressures on CLEAN (non-swap) episodes.
  - Clean episode: identity follows trajectory (invariance)
  - Clean episode + swapped features: identity should follow trajectory, not features (sensitivity)
  - Clean episode + swapped trajectory: identity should follow trajectory (counterfactual)
  - Swap episode: identity follows trajectory (normal identity loss, no extra pressure)

This matches the Neural Stage finding: counterfactual training is the
strongest pressure for relation internalization, BUT only when the
counterfactual signal is clean (no confounding).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DualChannelEncoder(nn.Module):
    def __init__(self, num_objects=2, dim=2, feature_dim=2,
                 hidden_dim=128, slot_dim=64, t_obs=10, t_pred=20):
        super().__init__()
        self.num_objects = num_objects
        self.dim = dim
        self.t_obs = t_obs
        self.t_pred = t_pred

        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        self.traj_encoder = nn.GRU(
            input_size=dim, hidden_size=slot_dim,
            num_layers=2, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        self.traj_decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, t_pred * dim),
        )
        self.binding_net = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_object(self, obs_pos_j, obs_feat_j):
        z_feat = self.feature_encoder(obs_feat_j)
        _, h_traj = self.traj_encoder(obs_pos_j)
        z_traj = h_traj[-1]
        z_fused = self.fusion(torch.cat([z_feat, z_traj], dim=-1))
        return z_fused, z_feat, z_traj

    def forward(self, observed_positions, observed_features=None,
                future_positions=None, future_features=None):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        if observed_features is not None and isinstance(observed_features, np.ndarray):
            observed_features = torch.FloatTensor(observed_features)
        if future_positions is not None and isinstance(future_positions, np.ndarray):
            future_positions = torch.FloatTensor(future_positions)
        if future_features is not None and isinstance(future_features, np.ndarray):
            future_features = torch.FloatTensor(future_features)

        B = observed_positions.shape[0]
        N = self.num_objects

        obs_feat_pooled = observed_features[:, 0, :, :] if (observed_features is not None and observed_features.dim() == 4) else observed_features

        z_obs_list = []
        for j in range(N):
            obs_pos_j = observed_positions[:, :, j, :]
            obs_feat_j = obs_feat_pooled[:, j, :] if obs_feat_pooled is not None else torch.zeros(B, self.feature_encoder[0].in_features, device=observed_positions.device)
            z_j, _, _ = self.encode_object(obs_pos_j, obs_feat_j)
            z_obs_list.append(z_j)
        z_obs = torch.stack(z_obs_list, dim=1)

        z_fut_list = []
        if future_features is not None:
            fut_feat_pooled = future_features[:, 0, :, :] if future_features.dim() == 4 else future_features
            for i in range(N):
                z_fut_list.append(self.feature_encoder(fut_feat_pooled[:, i, :]))
        z_fut = torch.stack(z_fut_list, dim=1) if z_fut_list else z_obs

        logits = torch.zeros(B, N, N, device=observed_positions.device)
        for i in range(N):
            for j in range(N):
                pair = torch.cat([z_fut[:, i, :], z_obs[:, j, :]], dim=-1)
                logits[:, i, j] = self.binding_net(pair).squeeze(-1)

        traj_preds = []
        for j in range(N):
            traj_j = self.traj_decoder(z_obs[:, j, :])
            traj_preds.append(traj_j.reshape(B, self.t_pred, self.dim))
        pred_traj = torch.stack(traj_preds, dim=2)

        return logits, z_obs, z_fut, pred_traj

    def predict_trajectory(self, observed_positions, observed_features=None):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        if observed_features is not None and isinstance(observed_features, np.ndarray):
            observed_features = torch.FloatTensor(observed_features)
        B = observed_positions.shape[0]
        N = self.num_objects
        obs_feat_pooled = observed_features[:, 0, :, :] if (observed_features is not None and observed_features.dim() == 4) else observed_features
        z_obs_list = []
        for j in range(N):
            obs_pos_j = observed_positions[:, :, j, :]
            obs_feat_j = obs_feat_pooled[:, j, :] if obs_feat_pooled is not None else torch.zeros(B, self.feature_encoder[0].in_features, device=observed_positions.device)
            z_j, _, _ = self.encode_object(obs_pos_j, obs_feat_j)
            z_obs_list.append(z_j)
        traj_preds = []
        for j in range(N):
            traj_j = self.traj_decoder(z_obs_list[j])
            traj_preds.append(traj_j.reshape(B, self.t_pred, self.dim))
        return torch.stack(traj_preds, dim=2)


class StructureMonitoringHead(nn.Module):
    def __init__(self, slot_dim=64, num_objects=2, hidden_dim=64):
        super().__init__()
        self.num_objects = num_objects
        self.probe = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objects),
        )

    def forward(self, z_obs):
        B = z_obs.shape[0]
        N = self.num_objects
        probe_logits = torch.zeros(B, N, N, device=z_obs.device)
        for j in range(N):
            probe_logits[:, j, :] = self.probe(z_obs[:, j, :])
        return probe_logits


class CounterfactualObjectFileV2(nn.Module):
    def __init__(self, num_objects=2, dim=2, feature_dim=2,
                 hidden_dim=128, slot_dim=64, t_obs=10, t_pred=20,
                 identity_weight=1.0, smh_weight=1.0,
                 invariance_weight=0.5, sensitivity_weight=0.5,
                 counterfactual_weight=1.0, traj_weight=0.1):
        super().__init__()
        self.num_objects = num_objects
        self.dim = dim
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.identity_weight = identity_weight
        self.smh_weight = smh_weight
        self.invariance_weight = invariance_weight
        self.sensitivity_weight = sensitivity_weight
        self.counterfactual_weight = counterfactual_weight
        self.traj_weight = traj_weight

        self.encoder = DualChannelEncoder(
            num_objects=num_objects, dim=dim, feature_dim=feature_dim,
            hidden_dim=hidden_dim, slot_dim=slot_dim,
            t_obs=t_obs, t_pred=t_pred)
        self.smh = StructureMonitoringHead(
            slot_dim=slot_dim, num_objects=num_objects,
            hidden_dim=hidden_dim // 2)

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
        if observed_features is not None and isinstance(observed_features, np.ndarray):
            observed_features = torch.FloatTensor(observed_features)
        if future_features is not None and isinstance(future_features, np.ndarray):
            future_features = torch.FloatTensor(future_features)

        B = observed_positions.shape[0]
        N = self.num_objects

        if is_swap is None:
            is_swap = torch.zeros(B, dtype=torch.bool)
        elif isinstance(is_swap, np.ndarray):
            is_swap = torch.BoolTensor(is_swap)
        clean_mask = ~is_swap

        logits, z_obs, z_fut, pred_traj = self.encoder(
            observed_positions, observed_features, future_positions, future_features)

        identity_loss = F.cross_entropy(logits.reshape(-1, N), identity_labels.reshape(-1))

        smh_logits = self.smh(z_obs)
        smh_loss = F.cross_entropy(smh_logits.reshape(-1, N), identity_labels.reshape(-1))

        traj_loss = F.mse_loss(pred_traj, future_positions)

        noisy_pos = observed_positions + torch.randn_like(observed_positions) * 0.3
        logits_noisy, z_obs_noisy, _, _ = self.encoder(
            noisy_pos, observed_features, future_positions, future_features)
        invariance_loss = F.kl_div(
            F.log_softmax(logits_noisy.reshape(-1, N), dim=-1),
            F.softmax(logits.reshape(-1, N).detach(), dim=-1),
            reduction='batchmean')

        sensitivity_loss = torch.tensor(0.0, device=observed_positions.device)
        if clean_mask.any() and future_features is not None:
            swapped_fut_feat = future_features.clone()
            if swapped_fut_feat.dim() == 4:
                swapped_fut_feat[:, :, 0, :], swapped_fut_feat[:, :, 1, :] = \
                    future_features[:, :, 1, :].clone(), future_features[:, :, 0, :].clone()
            elif swapped_fut_feat.dim() == 3:
                swapped_fut_feat[:, 0, :], swapped_fut_feat[:, 1, :] = \
                    future_features[:, 1, :].clone(), future_features[:, 0, :].clone()

            logits_swapped, _, _, _ = self.encoder(
                observed_positions[clean_mask], observed_features[clean_mask] if observed_features is not None else None,
                future_positions[clean_mask], swapped_fut_feat[clean_mask])

            swapped_id = identity_labels[clean_mask].clone()
            if N >= 2:
                swapped_id[:, 0], swapped_id[:, 1] = identity_labels[clean_mask, 1].clone(), identity_labels[clean_mask, 0].clone()
            sensitivity_loss = F.cross_entropy(logits_swapped.reshape(-1, N), swapped_id.reshape(-1))

        counterfactual_loss = torch.tensor(0.0, device=observed_positions.device)
        if clean_mask.any():
            swapped_obs_pos = observed_positions.clone()
            swapped_obs_pos[:, :, 0, :], swapped_obs_pos[:, :, 1, :] = \
                observed_positions[:, :, 1, :].clone(), observed_positions[:, :, 0, :].clone()

            logits_cf, z_obs_cf, _, _ = self.encoder(
                swapped_obs_pos[clean_mask], observed_features[clean_mask] if observed_features is not None else None,
                future_positions[clean_mask], future_features[clean_mask] if future_features is not None else None)

            cf_id = identity_labels[clean_mask].clone()
            if N >= 2:
                cf_id[:, 0], cf_id[:, 1] = identity_labels[clean_mask, 1].clone(), identity_labels[clean_mask, 0].clone()
            counterfactual_loss = F.cross_entropy(logits_cf.reshape(-1, N), cf_id.reshape(-1))

        total_loss = (self.identity_weight * identity_loss +
                      self.smh_weight * smh_loss +
                      self.invariance_weight * invariance_loss +
                      self.sensitivity_weight * sensitivity_loss +
                      self.counterfactual_weight * counterfactual_loss +
                      self.traj_weight * traj_loss)

        return total_loss, identity_loss, smh_loss, counterfactual_loss

    def forward(self, observed_positions, observed_features=None,
                future_positions=None, future_features=None):
        logits, z_obs, z_fut, pred_traj = self.encoder(
            observed_positions, observed_features, future_positions, future_features)
        smh_logits = self.smh(z_obs)
        return logits, z_obs, smh_logits, pred_traj

    def predict_future(self, observed_positions, observed_features=None):
        self.eval()
        with torch.no_grad():
            if isinstance(observed_positions, np.ndarray):
                observed_positions = torch.FloatTensor(observed_positions)
            pred_traj = self.encoder.predict_trajectory(observed_positions, observed_features)
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
                if isinstance(observed_positions, np.ndarray):
                    future_positions = torch.zeros(observed_positions.shape[0], self.t_pred, self.num_objects, self.dim)
                else:
                    future_positions = torch.zeros(observed_positions.shape[0], self.t_pred, self.num_objects, self.dim)

            logits, z_obs, smh_logits, _ = self.forward(
                observed_positions, observed_features, future_positions, future_features)

            if method == "smh":
                chosen_logits = smh_logits
            else:
                chosen_logits = logits

            pred_assignment = chosen_logits.argmax(dim=-1)
        if isinstance(pred_assignment, torch.Tensor):
            pred_assignment = pred_assignment.cpu().numpy()
        return pred_assignment

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
            _, z_obs, _, _ = self.encoder(
                observed_positions, observed_features, future_positions, future_features)
        return z_obs
