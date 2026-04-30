"""
SVT-v3.4 Models with Contrastive Feature Binding Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_binding_loss(feature_encoder, observed_features, future_features, identity_labels, num_objects):
    if observed_features is None or future_features is None:
        dev = identity_labels.device if isinstance(identity_labels, torch.Tensor) else "cpu"
        return torch.tensor(0.0, device=dev)

    if isinstance(observed_features, np.ndarray):
        observed_features = torch.FloatTensor(observed_features)
    if isinstance(future_features, np.ndarray):
        future_features = torch.FloatTensor(future_features)
    if isinstance(identity_labels, np.ndarray):
        identity_labels = torch.LongTensor(identity_labels)
    elif isinstance(identity_labels, torch.Tensor) and identity_labels.dtype != torch.long:
        identity_labels = identity_labels.long()

    if observed_features.dim() == 4:
        observed_features = observed_features[:, 0, :, :]
    if future_features.dim() == 4:
        future_features = future_features[:, 0, :, :]

    B = observed_features.shape[0]
    N = num_objects

    z_obs = feature_encoder(observed_features)
    z_fut = feature_encoder(future_features)

    z_obs_norm = F.normalize(z_obs, dim=-1)
    z_fut_norm = F.normalize(z_fut, dim=-1)

    sim_matrix = torch.bmm(z_fut_norm, z_obs_norm.transpose(1, 2))

    binding_loss = F.cross_entropy(
        sim_matrix.reshape(B * N, N),
        identity_labels.reshape(B * N),
    )

    return binding_loss


class MLPFeatureBinding(nn.Module):
    def __init__(self, t_obs=10, t_pred=20, num_objects=2, feature_dim=2,
                 hidden_dim=256, num_layers=3, dropout=0.1,
                 identity_weight=1.0, lambda_bind=1.0, feat_emb_dim=64):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.identity_weight = identity_weight
        self.lambda_bind = lambda_bind

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

        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, feat_emb_dim),
            nn.ReLU(),
            nn.Linear(feat_emb_dim, feat_emb_dim),
        )

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
                     observed_features=None, future_features=None):
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

        binding_loss = compute_binding_loss(
            self.feature_encoder, observed_features, future_features,
            identity_labels, self.num_objects)

        total_loss = mse_loss + self.identity_weight * assignment_loss + self.lambda_bind * binding_loss
        return total_loss, mse_loss, assignment_loss, binding_loss

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


class ObjectCentricFeatureBinding(nn.Module):
    def __init__(self, t_obs=10, t_pred=20, num_objects=2, feature_dim=2,
                 obj_hidden_dim=128, shared_hidden_dim=256,
                 identity_weight=1.0, lambda_bind=1.0, feat_emb_dim=64):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.identity_weight = identity_weight
        self.lambda_bind = lambda_bind

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

        self.assignment_head = nn.Sequential(
            nn.Linear(self.combined_dim, shared_hidden_dim),
            nn.ReLU(),
            nn.Linear(shared_hidden_dim, num_objects * num_objects),
        )

        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, feat_emb_dim),
            nn.ReLU(),
            nn.Linear(feat_emb_dim, feat_emb_dim),
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
        assignment_logits = self.assignment_head(combined).reshape(B, self.num_objects, self.num_objects)

        return traj_out, assignment_logits

    def compute_loss(self, observed_positions, future_positions, identity_labels,
                     observed_features=None, future_features=None):
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

        binding_loss = compute_binding_loss(
            self.feature_encoder, observed_features, future_features,
            identity_labels, self.num_objects)

        total_loss = mse_loss + self.identity_weight * assignment_loss + self.lambda_bind * binding_loss
        return total_loss, mse_loss, assignment_loss, binding_loss

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
