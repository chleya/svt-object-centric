"""
Evidential Deep Learning for ObjectFile Uncertainty Estimation

Uses Dirichlet-based evidential learning (Sensoy et al., 2018) to produce
principled uncertainty estimates for identity assignment.

Instead of rule-based confidence scores, the model learns to output
evidence vectors that parameterize a Dirichlet distribution over
class probabilities. This gives:
- Aleatoric uncertainty: inherent noise in the data
- Epistemic uncertainty: model's lack of knowledge (high in OOD/conflict)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EvidentialIdentityHead(nn.Module):
    """
    Evidential identity assignment head.

    For each (future_object, observed_object) pair, outputs evidence e_ij.
    The Dirichlet concentration parameters are alpha_ij = e_ij + 1.
    Identity probability: p_ij = alpha_ij / sum_j(alpha_ij)
    Epistemic uncertainty: u_i = K / sum_j(alpha_ij) where K = num_objects
    """

    def __init__(self, feature_dim=2, slot_dim=64, num_objects=2, hidden_dim=128):
        super().__init__()
        self.num_objects = num_objects
        self.feature_dim = feature_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )

        self.fut_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )

        self.evidence_net = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.traj_evidence_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _compute_pairwise_evidence(self, obs_feat, fut_feat):
        B = obs_feat.shape[0]
        N = self.num_objects

        z_obs = self.obs_encoder(obs_feat)
        z_fut = self.fut_encoder(fut_feat)

        evidence = torch.zeros(B, N, N, device=obs_feat.device)
        for i in range(N):
            for j in range(N):
                pair = torch.cat([z_fut[:, i, :], z_obs[:, j, :]], dim=-1)
                evidence[:, i, j] = F.softplus(self.evidence_net(pair)).squeeze(-1)

        return evidence

    def _compute_traj_evidence(self, fut_pos, pred_pos, last_vel):
        B = fut_pos.shape[0]
        N = self.num_objects

        evidence = torch.zeros(B, N, N, device=fut_pos.device)
        for i in range(N):
            for j in range(N):
                diff = fut_pos[:, i, :] - pred_pos[:, j, :]
                vel_diff = last_vel[:, j, :] if last_vel is not None else torch.zeros_like(diff)
                pair_feat = torch.cat([diff, vel_diff], dim=-1)
                evidence[:, i, j] = F.softplus(self.traj_evidence_net(pair_feat)).squeeze(-1)

        return evidence

    def forward(self, obs_feat, fut_feat, fut_pos=None, pred_pos=None, last_vel=None):
        feat_evidence = self._compute_pairwise_evidence(obs_feat, fut_feat)

        alpha_feat = feat_evidence + 1
        S_feat = alpha_feat.sum(dim=-1, keepdim=True)
        feat_probs = alpha_feat / S_feat
        feat_epistemic = self.num_objects / S_feat.squeeze(-1)

        result = {
            'feat_evidence': feat_evidence,
            'feat_alpha': alpha_feat,
            'feat_probs': feat_probs,
            'feat_epistemic': feat_epistemic,
        }

        if fut_pos is not None and pred_pos is not None:
            traj_evidence = self._compute_traj_evidence(fut_pos, pred_pos, last_vel)

            alpha_traj = traj_evidence + 1
            S_traj = alpha_traj.sum(dim=-1, keepdim=True)
            traj_probs = alpha_traj / S_traj
            traj_epistemic = self.num_objects / S_traj.squeeze(-1)

            result.update({
                'traj_evidence': traj_evidence,
                'traj_alpha': alpha_traj,
                'traj_probs': traj_probs,
                'traj_epistemic': traj_epistemic,
            })

        return result

    def compute_loss(self, obs_feat, fut_feat, identity_labels,
                     fut_pos=None, pred_pos=None, last_vel=None,
                     kl_weight=0.1):
        if isinstance(identity_labels, np.ndarray):
            identity_labels = torch.LongTensor(identity_labels)
        elif isinstance(identity_labels, torch.Tensor) and identity_labels.dtype != torch.long:
            identity_labels = identity_labels.long()

        result = self.forward(obs_feat, fut_feat, fut_pos, pred_pos, last_vel)

        alpha_feat = result['feat_alpha']
        B, N, _ = alpha_feat.shape

        eye = torch.eye(N, device=alpha_feat.device).unsqueeze(0).expand(B, -1, -1)
        y_onehot = torch.zeros_like(alpha_feat)
        for b in range(B):
            for i in range(N):
                y_onehot[b, i, identity_labels[b, i]] = 1.0

        S = alpha_feat.sum(dim=-1, keepdim=True)
        feat_loss = torch.sum(
            y_onehot * (torch.digamma(S) - torch.digamma(alpha_feat)),
            dim=-1
        ).mean()

        alpha_hat = y_onehot + (1 - y_onehot) * alpha_feat
        beta = torch.ones_like(alpha_hat)
        S_hat = alpha_hat.sum(dim=-1, keepdim=True)
        kl = torch.lgamma(S_hat) - torch.lgamma(alpha_hat).sum(dim=-1, keepdim=True)
        kl = kl + (alpha_hat - 1) * (torch.digamma(alpha_hat) - torch.digamma(S_hat))
        kl = kl - (torch.lgamma(beta).sum(dim=-1, keepdim=True) - torch.lgamma(beta.sum(dim=-1, keepdim=True)))
        kl = kl.sum(dim=-1).mean()

        total_loss = feat_loss + kl_weight * kl

        traj_loss = torch.tensor(0.0, device=obs_feat.device)
        if 'traj_alpha' in result:
            alpha_traj = result['traj_alpha']
            S_t = alpha_traj.sum(dim=-1, keepdim=True)
            traj_loss = torch.sum(
                y_onehot * (torch.digamma(S_t) - torch.digamma(alpha_traj)),
                dim=-1
            ).mean()
            total_loss = total_loss + traj_loss

        return total_loss, feat_loss, traj_loss, kl


class UncertaintyAwareObjectFile:
    """
    ObjectFile with Evidential Deep Learning uncertainty estimation.

    Replaces rule-based confidence with learned evidential uncertainty:
    - Feature confidence: Dirichlet concentration from feature evidence
    - Trajectory confidence: Dirichlet concentration from trajectory evidence
    - Conflict resolution: compare epistemic uncertainties
    - Abstain: when both sources have high epistemic uncertainty
    """

    def __init__(self, evidential_head, traj_model=None, num_objects=2,
                 feature_dim=2, epistemic_threshold=0.5,
                 conflict_strategy="lower_uncertainty"):
        self.evidential_head = evidential_head
        self.traj_model = traj_model
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.epistemic_threshold = epistemic_threshold
        self.conflict_strategy = conflict_strategy

    def _get_predicted_trajectory(self, observed_positions):
        if self.traj_model is None:
            return None
        self.traj_model.eval()
        with torch.no_grad():
            if isinstance(observed_positions, np.ndarray):
                obs_t = torch.FloatTensor(observed_positions)
            else:
                obs_t = observed_positions
            pred = self.traj_model(obs_t)
            if isinstance(pred, tuple):
                pred = pred[0]
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
        return pred

    def predict_identity(self, observed_positions, observed_features,
                         future_positions, future_features,
                         occlusion_mask=None, return_trace=False,
                         return_conflict_info=False):
        single = observed_positions.ndim == 3
        if single:
            observed_positions = observed_positions[np.newaxis]
            observed_features = observed_features[np.newaxis]
            future_positions = future_positions[np.newaxis]
            future_features = future_features[np.newaxis] if future_features is not None else None
            if occlusion_mask is not None:
                occlusion_mask = occlusion_mask[np.newaxis]

        B = observed_positions.shape[0]
        N = self.num_objects
        results = np.zeros((B, N), dtype=int)
        all_confidences = []
        all_sources = []
        all_abstain_flags = []
        all_epistemic = []

        pred_traj_all = self._get_predicted_trajectory(observed_positions)

        self.evidential_head.eval()
        with torch.no_grad():
            for b in range(B):
                obs_feat_t = torch.FloatTensor(observed_features[b, 0, :, :]).unsqueeze(0)

                fut_feat_pooled = future_features[b, 0, :, :] if future_features is not None else np.zeros((N, self.feature_dim))
                fut_feat_t = torch.FloatTensor(fut_feat_pooled).unsqueeze(0)

                fut_pos_t = None
                pred_pos_t = None
                last_vel_t = None

                if pred_traj_all is not None:
                    fut_pos_t = torch.FloatTensor(future_positions[b, 0, :, :]).unsqueeze(0)
                    pred_pos_t = torch.FloatTensor(pred_traj_all[b, 0, :, :]).unsqueeze(0)
                    last_vel_np = observed_positions[b, -1, :, :] - observed_positions[b, -2, :, :]
                    last_vel_t = torch.FloatTensor(last_vel_np).unsqueeze(0)

                ev_result = self.evidential_head(
                    obs_feat_t, fut_feat_t, fut_pos_t, pred_pos_t, last_vel_t)

                feat_probs = ev_result['feat_probs'][0].cpu().numpy()
                feat_epistemic = ev_result['feat_epistemic'][0].cpu().numpy()

                feat_assignment = np.argmax(feat_probs, axis=-1)

                has_traj = 'traj_probs' in ev_result
                if has_traj:
                    traj_probs = ev_result['traj_probs'][0].cpu().numpy()
                    traj_epistemic = ev_result['traj_epistemic'][0].cpu().numpy()
                    traj_assignment = np.argmax(traj_probs, axis=-1)
                else:
                    traj_probs = np.zeros((N, N))
                    traj_epistemic = np.ones(N)
                    traj_assignment = feat_assignment.copy()

                conflict = not np.array_equal(feat_assignment, traj_assignment)

                if not conflict:
                    chosen = feat_assignment.copy()
                    source = "agreement"
                    confidence = float(np.max(feat_probs, axis=-1).mean())
                    abstain = False
                    epistemic = float(feat_epistemic.mean())
                else:
                    avg_feat_epistemic = float(feat_epistemic.mean())
                    avg_traj_epistemic = float(traj_epistemic.mean())

                    if self.conflict_strategy == "lower_uncertainty":
                        if avg_traj_epistemic < avg_feat_epistemic:
                            chosen = traj_assignment.copy()
                            source = "trajectory"
                            confidence = float(np.max(traj_probs, axis=-1).mean())
                            abstain = False
                        elif avg_feat_epistemic < avg_traj_epistemic:
                            chosen = feat_assignment.copy()
                            source = "feature"
                            confidence = float(np.max(feat_probs, axis=-1).mean())
                            abstain = False
                        else:
                            chosen = traj_assignment.copy()
                            source = "uncertain"
                            confidence = 0.1
                            abstain = True

                    elif self.conflict_strategy == "epistemic_gated":
                        if avg_traj_epistemic < self.epistemic_threshold:
                            chosen = traj_assignment.copy()
                            source = "trajectory"
                            confidence = float(np.max(traj_probs, axis=-1).mean())
                            abstain = False
                        elif avg_feat_epistemic < self.epistemic_threshold:
                            chosen = feat_assignment.copy()
                            source = "feature"
                            confidence = float(np.max(feat_probs, axis=-1).mean())
                            abstain = False
                        else:
                            chosen = traj_assignment.copy()
                            source = "uncertain"
                            confidence = 0.1
                            abstain = True

                    else:
                        chosen = traj_assignment.copy()
                        source = "trajectory"
                        confidence = float(np.max(traj_probs, axis=-1).mean()) if has_traj else 0.3
                        abstain = False

                    epistemic = min(avg_feat_epistemic, avg_traj_epistemic)

                results[b] = chosen
                all_confidences.append(confidence)
                all_sources.append(source)
                all_abstain_flags.append(abstain)
                all_epistemic.append(epistemic)

        if return_conflict_info:
            return results, all_confidences, all_sources, all_abstain_flags
        return results
