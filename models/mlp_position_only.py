"""
SVT-v3.1 MLP Position-Only Model

Simple MLP that takes flattened observed positions and predicts future positions.
No feature input, no identity head.
"""

import torch
import torch.nn as nn
import numpy as np


class MLPPositionOnly(nn.Module):
    def __init__(self, t_obs=10, t_pred=20, num_objects=2, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.num_objects = num_objects
        self.input_dim = t_obs * num_objects * 2
        self.output_dim = t_pred * num_objects * 2

        layers = []
        in_dim = self.input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, self.output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, observed_positions):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        B = observed_positions.shape[0]
        x = observed_positions.reshape(B, -1)
        out = self.mlp(x)
        return out.reshape(B, self.t_pred, self.num_objects, 2)

    def predict_future(self, observed_positions, observed_features=None):
        self.eval()
        with torch.no_grad():
            pred = self.forward(observed_positions)
        if isinstance(observed_positions, np.ndarray):
            return pred.numpy()
        return pred

    def predict_identity(self, observed_positions, observed_features=None, test_future=None):
        B = observed_positions.shape[0] if isinstance(observed_positions, np.ndarray) else observed_positions.shape[0]
        N = self.num_objects
        return np.tile(np.arange(N), (B, 1))
