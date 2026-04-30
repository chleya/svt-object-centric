import torch
import torch.nn as nn
import numpy as np


class IdentityHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)


class DualHeadModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        identity_weight: float = 1.0,
    ):
        super().__init__()
        self.base = base_model
        self.identity_head = None
        self.identity_weight = identity_weight

    def _build_identity_head(self, input_dim: int):
        if self.identity_head is None:
            self.identity_head = IdentityHead(input_dim)

    def forward(self, observed_positions: torch.Tensor):
        pred_future = self.base(observed_positions)
        B, T, N, D = pred_future.shape
        identity_input = pred_future.mean(dim=1).reshape(B, N * D)
        self._build_identity_head(N * D)
        identity_logit = self.identity_head(identity_input)
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
