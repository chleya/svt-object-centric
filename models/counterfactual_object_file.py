"""
Counterfactual ObjectFile with Structure Monitoring Head (SMH)

v12 finding: ALL learned models are in State A (identity not readable from representation).
Root cause: models bypass identity encoding by directly matching features/trajectories.

Solution from two projects:
  1. Neural Stage (Relation-Internalization): counterfactual training is the strongest
     pressure for relation internalization (gated_score ~0.981 vs edit-pressure ~0.200)
  2. Probe-to-Boundary: Structure Monitoring Head (SMH) forces identity encoding
     by adding a probe loss that monitors whether identity is in the representation

Architecture:
  A. Dual-channel backbone (feature + trajectory) → shared hidden representation
  B. Structure Monitoring Head (SMH): linear probe that reads identity from hidden
     - Level A (passive): just monitor, report probe accuracy
     - Level B (active): add probe loss to force identity encoding
  C. Counterfactual training: three simultaneous pressures
     - Invariance: same identity under nuisance perturbations (noise, translation)
     - Sensitivity: different identity when binding changes (swap features)
     - Counterfactual: intervene on one channel, identity should follow the other

Key difference from previous models:
  - Previous: identity loss on final logits → model can bypass via direct matching
  - This: identity loss on INTERMEDIATE representation → model MUST encode identity
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
                fut_feat_i = fut_feat_pooled[:, i, :]
                z_fut_feat = self.feature_encoder(fut_feat_i)
                z_fut_list.append(z_fut_feat)
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


class CounterfactualObjectFile(nn.Module):
    def __init__(self, num_objects=2, dim=2, feature_dim=2,
                 hidden_dim=128, slot_dim=64, t_obs=10, t_pred=20,
                 identity_weight=1.0, smh_weight=1.0,
                 invariance_weight=0.5, sensitivity_weight=0.5,
                 counterfactual_weight=1.0, traj_weight=0.1,
                 smh_mode='active'):
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
        self.smh_mode = smh_mode

        self.encoder = DualChannelEncoder(
            num_objects=num_objects, dim=dim, feature_dim=feature_dim,
            hidden_dim=hidden_dim, slot_dim=slot_dim,
            t_obs=t_obs, t_pred=t_pred)

        self.smh = StructureMonitoringHead(
            slot_dim=slot_dim, num_objects=num_objects,
            hidden_dim=hidden_dim // 2)

    def _add_nuisance_noise(self, observed_positions, noise_std=0.5):
        noise = torch.randn_like(observed_positions) * noise_std
        return observed_positions + noise

    def _swap_features(self, future_features):
        if future_features is None:
            return None
        N = self.num_objects
        if N < 2:
            return future_features
        swapped = future_features.clone()
        if swapped.dim() == 4:
            swapped[:, :, 0, :], swapped[:, :, 1, :] = future_features[:, :, 1, :].clone(), future_features[:, :, 0, :].clone()
        elif swapped.dim() == 3:
            swapped[:, 0, :], swapped[:, 1, :] = future_features[:, 1, :].clone(), future_features[:, 0, :].clone()
        return swapped

    def _swap_trajectories(self, observed_positions):
        N = self.num_objects
        if N < 2:
            return observed_positions
        swapped = observed_positions.clone()
        swapped[:, :, 0, :], swapped[:, :, 1, :] = observed_positions[:, :, 1, :].clone(), observed_positions[:, :, 0, :].clone()
        return swapped

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

        logits, z_obs, z_fut, pred_traj = self.encoder(
            observed_positions, observed_features, future_positions, future_features)

        identity_loss = F.cross_entropy(logits.reshape(-1, N), identity_labels.reshape(-1))

        smh_logits = self.smh(z_obs.detach() if self.smh_mode == 'passive' else z_obs)
        smh_loss = F.cross_entropy(smh_logits.reshape(-1, N), identity_labels.reshape(-1))

        traj_loss = F.mse_loss(pred_traj, future_positions)

        noisy_pos = self._add_nuisance_noise(observed_positions, noise_std=0.3)
        logits_noisy, z_obs_noisy, _, _ = self.encoder(
            noisy_pos, observed_features, future_positions, future_features)
        invariance_loss = F.kl_div(
            F.log_softmax(logits_noisy.reshape(-1, N), dim=-1),
            F.softmax(logits.reshape(-1, N).detach(), dim=-1),
            reduction='batchmean')

        swapped_fut_feat = self._swap_features(future_features)
        if swapped_fut_feat is not None:
            logits_swapped, _, _, _ = self.encoder(
                observed_positions, observed_features, future_positions, swapped_fut_feat)
            swapped_identity = identity_labels.clone()
            if N >= 2:
                swapped_identity[:, 0], swapped_identity[:, 1] = identity_labels[:, 1].clone(), identity_labels[:, 0].clone()
            sensitivity_loss = F.cross_entropy(logits_swapped.reshape(-1, N), swapped_identity.reshape(-1))
        else:
            sensitivity_loss = torch.tensor(0.0, device=observed_positions.device)

        swapped_obs_pos = self._swap_trajectories(observed_positions)
        logits_cf, z_obs_cf, _, _ = self.encoder(
            swapped_obs_pos, observed_features, future_positions, future_features)
        cf_identity = identity_labels.clone()
        if N >= 2:
            cf_identity[:, 0], cf_identity[:, 1] = identity_labels[:, 1].clone(), identity_labels[:, 0].clone()
        counterfactual_loss = F.cross_entropy(logits_cf.reshape(-1, N), cf_identity.reshape(-1))

        smh_noisy_logits = self.smh(z_obs_noisy.detach() if self.smh_mode == 'passive' else z_obs_noisy)
        smh_invariance = F.kl_div(
            F.log_softmax(smh_noisy_logits.reshape(-1, N), dim=-1),
            F.softmax(smh_logits.reshape(-1, N).detach(), dim=-1),
            reduction='batchmean')

        total_loss = (self.identity_weight * identity_loss +
                      self.smh_weight * smh_loss +
                      self.invariance_weight * invariance_loss +
                      self.sensitivity_weight * sensitivity_loss +
                      self.counterfactual_weight * counterfactual_loss +
                      self.traj_weight * traj_loss +
                      0.3 * smh_invariance)

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
                future_positions = np.zeros(
                    (observed_positions.shape[0] if isinstance(observed_positions, torch.Tensor) else len(observed_positions),
                     self.t_pred, self.num_objects, self.dim),
                    dtype=np.float32) if isinstance(observed_positions, np.ndarray) else \
                    torch.zeros(observed_positions.shape[0], self.t_pred, self.num_objects, self.dim)

            logits, z_obs, smh_logits, _ = self.forward(
                observed_positions, observed_features, future_positions, future_features)

            if method == "combined":
                chosen_logits = logits
            elif method == "smh":
                chosen_logits = smh_logits
            elif method == "feature":
                chosen_logits = logits
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
