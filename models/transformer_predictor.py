import torch
import torch.nn as nn
import math


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        t_obs: int = 10,
        t_pred: int = 20,
        n_objects: int = 2,
        dim: int = 2,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.n_objects = n_objects
        self.dim = dim
        self.d_model = d_model

        self.input_proj = nn.Linear(dim * n_objects, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=t_obs + t_pred)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        self.query_embed = nn.Parameter(torch.randn(1, t_pred, d_model) * 0.02)
        self.output_proj = nn.Linear(d_model, dim * n_objects)

    def forward(self, observed_positions: torch.Tensor) -> torch.Tensor:
        B = observed_positions.shape[0]

        x = observed_positions.reshape(B, self.t_obs, self.n_objects * self.dim)
        x = self.input_proj(x)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        memory = self.encoder(x)

        query = self.query_embed.expand(B, -1, -1)
        query = query * math.sqrt(self.d_model)
        query = self.pos_encoder(query, offset=self.t_obs)

        out = self.decoder(query, memory)
        out = self.output_proj(out)

        return out.reshape(B, self.t_pred, self.n_objects, self.dim)

    def predict_future(self, observed_positions: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(observed_positions)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        x = x + self.pe[:, offset : offset + x.size(1)]
        return x
