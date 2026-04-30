import torch
import torch.nn as nn


class ObservationConditionedIdentityModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        t_obs: int = 10,
        n_objects: int = 2,
        dim: int = 2,
        hidden_dim: int = 64,
        identity_weight: float = 1.0,
    ):
        super().__init__()
        self.base = base_model
        self.t_obs = t_obs
        self.n_objects = n_objects
        self.dim = dim
        self.identity_weight = identity_weight

        obs_feature_dim = t_obs * n_objects * dim
        self.identity_net = nn.Sequential(
            nn.Linear(obs_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observed_positions: torch.Tensor):
        pred_future = self.base(observed_positions)
        B = observed_positions.shape[0]
        obs_flat = observed_positions.reshape(B, -1)
        identity_logit = self.identity_net(obs_flat).squeeze(-1)
        return pred_future, identity_logit

    def predict_future(self, observed_positions: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            pred, _ = self.forward(observed_positions)
            return pred

    def predict_identity(self, observed_positions: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            _, logit = self.forward(observed_positions)
            swapped = (logit > 0).long()
            B = observed_positions.shape[0]
            N = observed_positions.shape[2]
            ids = torch.arange(N).unsqueeze(0).expand(B, -1).to(observed_positions.device)
            for i in range(B):
                if swapped[i]:
                    ids[i] = torch.tensor([1, 0], device=ids.device)
            return ids

    def compute_loss(
        self,
        observed_positions: torch.Tensor,
        future_positions: torch.Tensor,
        identity_labels: torch.Tensor,
    ):
        pred_future, identity_logit = self.forward(observed_positions)

        pred_loss = nn.functional.mse_loss(pred_future, future_positions)

        swap_target = (identity_labels[:, 0] == 1).long()
        id_loss = nn.functional.binary_cross_entropy_with_logits(
            identity_logit, swap_target.float()
        )

        total_loss = pred_loss + self.identity_weight * id_loss
        return total_loss, pred_loss, id_loss


class ObjectCentricPredictor(nn.Module):
    def __init__(
        self,
        t_obs: int = 10,
        t_pred: int = 20,
        n_objects: int = 2,
        dim: int = 2,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.n_objects = n_objects
        self.dim = dim

        input_dim = t_obs * dim
        output_dim = t_pred * dim

        layers = []
        current = input_dim
        for i in range(n_layers):
            nxt = hidden_dim if i < n_layers - 1 else output_dim
            layers.append(nn.Linear(current, nxt))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            current = nxt
        self.per_object_net = nn.Sequential(*layers)

        self.identity_net = nn.Sequential(
            nn.Linear(t_obs * n_objects * dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, observed_positions: torch.Tensor):
        B, T, N, D = observed_positions.shape

        pred_list = []
        for obj_idx in range(N):
            obj_obs = observed_positions[:, :, obj_idx, :].reshape(B, -1)
            obj_pred = self.per_object_net(obj_obs)
            pred_list.append(obj_pred.reshape(B, self.t_pred, D))

        pred_future = torch.stack(pred_list, dim=2)

        obs_flat = observed_positions.reshape(B, -1)
        identity_logit = self.identity_net(obs_flat).squeeze(-1)

        return pred_future, identity_logit

    def predict_future(self, observed_positions: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            pred, _ = self.forward(observed_positions)
            return pred

    def predict_identity(self, observed_positions: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            _, logit = self.forward(observed_positions)
            swapped = (logit > 0).long()
            B = observed_positions.shape[0]
            N = observed_positions.shape[2]
            ids = torch.arange(N).unsqueeze(0).expand(B, -1).to(observed_positions.device)
            for i in range(B):
                if swapped[i]:
                    ids[i] = torch.tensor([1, 0], device=ids.device)
            return ids

    def compute_loss(
        self,
        observed_positions: torch.Tensor,
        future_positions: torch.Tensor,
        identity_labels: torch.Tensor,
    ):
        pred_future, identity_logit = self.forward(observed_positions)

        pred_loss = nn.functional.mse_loss(pred_future, future_positions)

        swap_target = (identity_labels[:, 0] == 1).long()
        id_loss = nn.functional.binary_cross_entropy_with_logits(
            identity_logit, swap_target.float()
        )

        total_loss = pred_loss + id_loss
        return total_loss, pred_loss, id_loss
