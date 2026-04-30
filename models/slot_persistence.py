import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotPersistenceModel(nn.Module):
    def __init__(
        self,
        t_obs: int = 10,
        t_pred: int = 20,
        n_objects: int = 2,
        dim: int = 2,
        slot_dim: int = 32,
        n_slots: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.n_objects = n_objects
        self.dim = dim
        self.slot_dim = slot_dim
        self.n_slots = n_slots

        self.obj_encoder = nn.GRU(
            input_size=dim,
            hidden_size=slot_dim,
            num_layers=2,
            batch_first=True,
        )

        self.slot_update = nn.GRUCell(
            input_size=dim + slot_dim,
            hidden_size=slot_dim,
        )

        self.pos_decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

        self.vel_decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

        self.matching_head = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_objects(self, observed_positions):
        B, T, N, D = observed_positions.shape
        slot_states = []

        for obj_idx in range(N):
            obj_seq = observed_positions[:, :, obj_idx, :]
            _, h_n = self.obj_encoder(obj_seq)
            slot_state = h_n[-1]
            slot_states.append(slot_state)

        return torch.stack(slot_states, dim=1)

    def predict_future_with_slots(self, observed_positions, slot_states):
        B = observed_positions.shape[0]
        N = self.n_objects

        last_pos = observed_positions[:, -1]
        last_vel = observed_positions[:, -1] - observed_positions[:, -2]

        predictions = []
        current_pos = last_pos
        current_vel = last_vel
        current_slots = slot_states

        for t in range(self.t_pred):
            next_slots = []
            next_positions = []

            for obj_idx in range(N):
                slot_input = torch.cat([current_pos[:, obj_idx], current_slots[:, obj_idx]], dim=-1)
                new_slot = self.slot_update(slot_input, current_slots[:, obj_idx])
                next_slots.append(new_slot)

                vel_delta = self.vel_decoder(new_slot)
                new_vel = current_vel[:, obj_idx] + vel_delta * 0.1
                new_pos = current_pos[:, obj_idx] + new_vel

                next_positions.append(new_pos)
                current_vel[:, obj_idx] = new_vel

            current_pos = torch.stack(next_positions, dim=1)
            current_slots = torch.stack(next_slots, dim=1)
            predictions.append(current_pos)

        return torch.stack(predictions, dim=1)

    def predict_identity_by_slot_matching(self, slot_states, future_positions):
        B = slot_states.shape[0]
        N = self.n_objects
        ids = torch.arange(N).unsqueeze(0).expand(B, -1).to(slot_states.device)

        if N != 2:
            return ids

        first_fut_vel = future_positions[:, 0] - future_positions[:, 1] if future_positions.shape[1] > 1 else future_positions[:, 0]

        for i in range(B):
            pair_00 = torch.cat([slot_states[i, 0], slot_states[i, 0]], dim=-1)
            pair_01 = torch.cat([slot_states[i, 0], slot_states[i, 1]], dim=-1)
            pair_10 = torch.cat([slot_states[i, 1], slot_states[i, 0]], dim=-1)
            pair_11 = torch.cat([slot_states[i, 1], slot_states[i, 1]], dim=-1)

            sim_no_swap = self.matching_head(pair_00).squeeze() + self.matching_head(pair_11).squeeze()
            sim_swap = self.matching_head(pair_01).squeeze() + self.matching_head(pair_10).squeeze()

            if sim_swap > sim_no_swap:
                ids[i] = torch.tensor([1, 0], device=ids.device)

        return ids

    def forward(self, observed_positions):
        slot_states = self.encode_objects(observed_positions)
        pred_future = self.predict_future_with_slots(observed_positions, slot_states)
        return pred_future, slot_states

    def predict_future(self, observed_positions):
        self.eval()
        with torch.no_grad():
            pred, _ = self.forward(observed_positions)
            return pred

    def predict_identity(self, observed_positions):
        self.eval()
        with torch.no_grad():
            _, slot_states = self.forward(observed_positions)

            B = observed_positions.shape[0]
            N = observed_positions.shape[2]
            ids = torch.arange(N).unsqueeze(0).expand(B, -1).to(observed_positions.device)

            if N != 2:
                return ids

            last_obs_vel = observed_positions[:, -1] - observed_positions[:, -2]

            for i in range(B):
                vel_0 = self.vel_decoder(slot_states[i, 0])
                vel_1 = self.vel_decoder(slot_states[i, 1])

                dist_no_swap = (
                    torch.norm(last_obs_vel[i, 0] - vel_0) +
                    torch.norm(last_obs_vel[i, 1] - vel_1)
                )
                dist_swap = (
                    torch.norm(last_obs_vel[i, 0] - vel_1) +
                    torch.norm(last_obs_vel[i, 1] - vel_0)
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
        pred_future, slot_states = self.forward(observed_positions)

        pred_loss = F.mse_loss(pred_future, future_positions)

        vel_reg = torch.tensor(0.0, device=observed_positions.device)
        for obj_idx in range(self.n_objects):
            pred_vel = self.vel_decoder(slot_states[:, obj_idx])
            first_fut_vel = future_positions[:, 0, obj_idx] - observed_positions[:, -1, obj_idx]
            vel_reg = vel_reg + F.mse_loss(pred_vel, first_fut_vel)

        total_loss = pred_loss + 0.5 * vel_reg
        return total_loss, pred_loss, vel_reg
