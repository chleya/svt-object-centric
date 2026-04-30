import torch
import torch.nn as nn
import torch.nn.functional as F


class VelocityContinuityModel(nn.Module):
    def __init__(
        self,
        t_obs: int = 10,
        t_pred: int = 20,
        n_objects: int = 2,
        dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        vc_weight: float = 1.0,
    ):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.n_objects = n_objects
        self.dim = dim
        self.vc_weight = vc_weight

        input_dim = t_obs * n_objects * dim
        output_dim = t_pred * n_objects * dim

        layers = []
        current_dim = input_dim
        for i in range(n_layers):
            next_dim = hidden_dim if i < n_layers - 1 else output_dim
            layers.append(nn.Linear(current_dim, next_dim))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            current_dim = next_dim
        self.net = nn.Sequential(*layers)

        self.vel_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_objects * dim),
        )

    def forward(self, observed_positions: torch.Tensor):
        B = observed_positions.shape[0]
        x = observed_positions.reshape(B, -1)
        out = self.net(x)
        pred_future = out.reshape(B, self.t_pred, self.n_objects, self.dim)

        pred_vel = self.vel_head(x).reshape(B, self.n_objects, self.dim)

        return pred_future, pred_vel

    def predict_future(self, observed_positions: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            pred, _ = self.forward(observed_positions)
            return pred

    def predict_identity(self, observed_positions: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            _, pred_vel = self.forward(observed_positions)

            B = observed_positions.shape[0]
            N = observed_positions.shape[2]
            ids = torch.arange(N).unsqueeze(0).expand(B, -1).to(observed_positions.device)

            last_obs_vel = observed_positions[:, -1] - observed_positions[:, -2]

            for i in range(B):
                if N != 2:
                    continue
                dist_no_swap = (
                    torch.norm(last_obs_vel[i, 0] - pred_vel[i, 0]) +
                    torch.norm(last_obs_vel[i, 1] - pred_vel[i, 1])
                )
                dist_swap = (
                    torch.norm(last_obs_vel[i, 0] - pred_vel[i, 1]) +
                    torch.norm(last_obs_vel[i, 1] - pred_vel[i, 0])
                )
                if dist_swap < dist_no_swap:
                    ids[i] = torch.tensor([1, 0], device=ids.device)

            return ids

    def compute_loss(
        self,
        observed_positions: torch.Tensor,
        future_positions: torch.Tensor,
        identity_labels: torch.Tensor = None,
    ):
        pred_future, pred_vel = self.forward(observed_positions)

        pred_loss = F.mse_loss(pred_future, future_positions)

        first_fut_vel = future_positions[:, 0] - observed_positions[:, -1]
        vel_loss = F.mse_loss(pred_vel, first_fut_vel)

        total_loss = pred_loss + self.vc_weight * vel_loss
        return total_loss, pred_loss, vel_loss


class ContrastiveIdentityModel(nn.Module):
    def __init__(
        self,
        t_obs: int = 10,
        t_pred: int = 20,
        n_objects: int = 2,
        dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.n_objects = n_objects
        self.dim = dim
        self.temperature = temperature

        input_dim = t_obs * n_objects * dim
        output_dim = t_pred * n_objects * dim

        layers = []
        current_dim = input_dim
        for i in range(n_layers):
            next_dim = hidden_dim if i < n_layers - 1 else output_dim
            layers.append(nn.Linear(current_dim, next_dim))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            current_dim = next_dim
        self.net = nn.Sequential(*layers)

        self.obj_encoder = nn.Sequential(
            nn.Linear(t_obs * dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        self.fut_encoder = nn.Sequential(
            nn.Linear(t_pred * dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

    def forward(self, observed_positions: torch.Tensor):
        B = observed_positions.shape[0]
        x = observed_positions.reshape(B, -1)
        out = self.net(x)
        pred_future = out.reshape(B, self.t_pred, self.n_objects, self.dim)
        return pred_future

    def predict_future(self, observed_positions: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(observed_positions)

    def predict_identity(self, observed_positions: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            B = observed_positions.shape[0]
            N = observed_positions.shape[2]
            ids = torch.arange(N).unsqueeze(0).expand(B, -1).to(observed_positions.device)

            obj_features = []
            for obj_idx in range(N):
                obj_obs = observed_positions[:, :, obj_idx, :].reshape(B, -1)
                obj_feat = self.obj_encoder(obj_obs)
                obj_feat = F.normalize(obj_feat, dim=-1)
                obj_features.append(obj_feat)

            pred_future = self.forward(observed_positions)
            fut_features = []
            for obj_idx in range(N):
                obj_fut = pred_future[:, :, obj_idx, :].reshape(B, -1)
                fut_feat = self.fut_encoder(obj_fut)
                fut_feat = F.normalize(fut_feat, dim=-1)
                fut_features.append(fut_feat)

            if N == 2:
                for i in range(B):
                    sim_no_swap = (
                        torch.dot(obj_features[0][i], fut_features[0][i]) +
                        torch.dot(obj_features[1][i], fut_features[1][i])
                    )
                    sim_swap = (
                        torch.dot(obj_features[0][i], fut_features[1][i]) +
                        torch.dot(obj_features[1][i], fut_features[0][i])
                    )
                    if sim_swap > sim_no_swap:
                        ids[i] = torch.tensor([1, 0], device=ids.device)

            return ids

    def compute_loss(
        self,
        observed_positions: torch.Tensor,
        future_positions: torch.Tensor,
        identity_labels: torch.Tensor = None,
    ):
        B = observed_positions.shape[0]
        N = observed_positions.shape[2]

        pred_future = self.forward(observed_positions)
        pred_loss = F.mse_loss(pred_future, future_positions)

        obj_features = []
        for obj_idx in range(N):
            obj_obs = observed_positions[:, :, obj_idx, :].reshape(B, -1)
            obj_feat = self.obj_encoder(obj_obs)
            obj_feat = F.normalize(obj_feat, dim=-1)
            obj_features.append(obj_feat)

        fut_features = []
        for obj_idx in range(N):
            obj_fut = future_positions[:, :, obj_idx, :].reshape(B, -1)
            fut_feat = self.fut_encoder(obj_fut)
            fut_feat = F.normalize(fut_feat, dim=-1)
            fut_features.append(fut_feat)

        if identity_labels is not None:
            swap_mask = (identity_labels[:, 0] == 1)
        else:
            swap_mask = torch.zeros(B, dtype=torch.bool, device=observed_positions.device)

        contrastive_loss = torch.tensor(0.0, device=observed_positions.device)
        n_pairs = 0

        for i in range(B):
            if swap_mask[i]:
                pos_sim_0 = torch.dot(obj_features[0][i], fut_features[1][i]) / self.temperature
                pos_sim_1 = torch.dot(obj_features[1][i], fut_features[0][i]) / self.temperature
                neg_sim_0 = torch.dot(obj_features[0][i], fut_features[0][i]) / self.temperature
                neg_sim_1 = torch.dot(obj_features[1][i], fut_features[1][i]) / self.temperature
            else:
                pos_sim_0 = torch.dot(obj_features[0][i], fut_features[0][i]) / self.temperature
                pos_sim_1 = torch.dot(obj_features[1][i], fut_features[1][i]) / self.temperature
                neg_sim_0 = torch.dot(obj_features[0][i], fut_features[1][i]) / self.temperature
                neg_sim_1 = torch.dot(obj_features[1][i], fut_features[0][i]) / self.temperature

            loss_0 = -pos_sim_0 + torch.logsumexp(torch.stack([pos_sim_0, neg_sim_0]), dim=0)
            loss_1 = -pos_sim_1 + torch.logsumexp(torch.stack([pos_sim_1, neg_sim_1]), dim=0)
            contrastive_loss = contrastive_loss + loss_0 + loss_1
            n_pairs += 2

        if n_pairs > 0:
            contrastive_loss = contrastive_loss / n_pairs

        total_loss = pred_loss + contrastive_loss
        return total_loss, pred_loss, contrastive_loss
