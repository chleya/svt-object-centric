import torch
import torch.nn as nn


class MLPPredictor(nn.Module):
    def __init__(
        self,
        t_obs: int = 10,
        t_pred: int = 20,
        n_objects: int = 2,
        dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.n_objects = n_objects
        self.dim = dim

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

    def forward(self, observed_positions: torch.Tensor) -> torch.Tensor:
        B = observed_positions.shape[0]
        x = observed_positions.reshape(B, -1)
        out = self.net(x)
        return out.reshape(B, self.t_pred, self.n_objects, self.dim)

    def predict_future(self, observed_positions: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(observed_positions)
