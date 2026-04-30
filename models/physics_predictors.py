"""
Hamiltonian Neural Network (HNN) Trajectory Predictor (Greydanus et al., 2019)

Learns a Hamiltonian H(q, p) and derives equations of motion from it:
    dq/dt = ∂H/∂p
    dp/dt = -∂H/∂q

This ensures energy conservation by construction, which should improve
OOD generalization for physical systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HamiltonianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class HNNTrajectoryPredictor(nn.Module):
    """
    HNN-based trajectory predictor for SVT.

    Instead of directly predicting future positions (MLP),
    learns the Hamiltonian H(q, p) and integrates equations of motion.
    This should conserve energy and generalize better OOD.
    """

    def __init__(self, t_obs=10, t_pred=20, num_objects=2, dim=2,
                 hidden_dim=128, hnn_hidden=128, dt=1.0):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.num_objects = num_objects
        self.dim = dim
        self.dt = dt

        self.state_dim = 2 * dim * num_objects

        self.hnn = HamiltonianNN(self.state_dim, hnn_hidden)

        self.obs_encoder = nn.GRU(
            input_size=dim * num_objects,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        self.state_init = nn.Linear(hidden_dim, self.state_dim)

    def _get_initial_state(self, observed_positions):
        B, T, N, D = observed_positions.shape
        x = observed_positions.reshape(B, T, N * D)

        _, h_n = self.obs_encoder(x)
        init_state = self.state_init(h_n[-1])

        return init_state

    def _derivatives(self, state):
        dH = torch.autograd.grad(
            self.hnn(state).sum(), state, create_graph=True
        )[0]

        dH_dq = dH[:, :self.state_dim // 2]
        dH_dp = dH[:, self.state_dim // 2:]

        dq_dt = dH_dp
        dp_dt = -dH_dq

        return torch.cat([dq_dt, dp_dt], dim=-1)

    def _rk4_step(self, state, dt):
        k1 = self._derivatives(state)
        k2 = self._derivatives(state + 0.5 * dt * k1)
        k3 = self._derivatives(state + 0.5 * dt * k2)
        k4 = self._derivatives(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _integrate(self, initial_state, n_steps, dt):
        state = initial_state
        positions = [state[:, :self.state_dim // 2]]

        for _ in range(n_steps):
            state = self._rk4_step(state, dt)
            positions.append(state[:, :self.state_dim // 2])

        return torch.stack(positions[1:], dim=1)

    def forward(self, observed_positions):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)

        with torch.enable_grad():
            init_state = self._get_initial_state(observed_positions)
            init_state = init_state.requires_grad_(True)

            positions_flat = self._integrate(init_state, self.t_pred, self.dt)

        B = observed_positions.shape[0]
        traj_out = positions_flat.reshape(B, self.t_pred, self.num_objects, self.dim)
        return traj_out

    def compute_loss(self, observed_positions, future_positions, identity_labels,
                     observed_features=None, future_features=None):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        if isinstance(future_positions, np.ndarray):
            future_positions = torch.FloatTensor(future_positions)

        with torch.enable_grad():
            init_state = self._get_initial_state(observed_positions)
            init_state = init_state.requires_grad_(True)
            positions_flat = self._integrate(init_state, self.t_pred, self.dt)

        B = observed_positions.shape[0]
        pred_traj = positions_flat.reshape(B, self.t_pred, self.num_objects, self.dim)

        mse_loss = F.mse_loss(pred_traj, future_positions)

        if future_positions.shape[1] > 1:
            fut_vel = future_positions[:, 1:] - future_positions[:, :-1]
            pred_vel = pred_traj[:, 1:] - pred_traj[:, :-1]
            vel_loss = F.mse_loss(pred_vel, fut_vel)
        else:
            vel_loss = torch.tensor(0.0, device=observed_positions.device)

        total_loss = mse_loss + 0.1 * vel_loss
        return total_loss, mse_loss, vel_loss, torch.tensor(0.0)

    def predict_future(self, observed_positions, observed_features=None):
        self.eval()
        with torch.enable_grad():
            traj_pred = self.forward(observed_positions)
        if isinstance(observed_positions, np.ndarray):
            return traj_pred.detach().cpu().numpy()
        return traj_pred.detach()

    def predict_identity(self, observed_positions, observed_features=None,
                         test_future=None, future_features=None,
                         future_positions=None):
        self.eval()
        with torch.enable_grad():
            pred_traj = self.forward(observed_positions)

        if isinstance(pred_traj, torch.Tensor):
            pred_traj = pred_traj.detach().cpu().numpy()

        if future_positions is None and test_future is not None:
            future_positions = test_future
        if isinstance(future_positions, torch.Tensor):
            future_positions = future_positions.cpu().numpy()

        B = pred_traj.shape[0]
        N = self.num_objects
        results = np.zeros((B, N), dtype=int)

        if future_positions is None:
            return results

        for b in range(B):
            pred_mean = pred_traj[b].mean(axis=0)
            actual_mean = future_positions[b].mean(axis=0)

            used = set()
            for i in range(N):
                dists = np.linalg.norm(actual_mean[i] - pred_mean, axis=-1)
                dists_m = dists.copy()
                for j in used:
                    dists_m[j] = float('inf')
                best = np.argmin(dists_m)
                results[b, i] = best
                used.add(best)

        return results


class LNNTrajectoryPredictor(nn.Module):
    """
    Lagrangian Neural Network (LNN) Trajectory Predictor (Lutter et al., 2019)

    Learns the Lagrangian L(q, q_dot) and derives equations of motion:
    d/dt(∂L/∂q_dot) - ∂L/∂q = 0

    More flexible than HNN because it doesn't require canonical momentum.
    Can handle dissipative systems (friction, damping).
    """

    def __init__(self, t_obs=10, t_pred=20, num_objects=2, dim=2,
                 hidden_dim=128, lnn_hidden=128, dt=1.0):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.num_objects = num_objects
        self.dim = dim
        self.dt = dt

        self.q_dim = dim * num_objects
        self.state_dim = 2 * dim * num_objects

        self.lagrangian_net = nn.Sequential(
            nn.Linear(self.state_dim, lnn_hidden),
            nn.Tanh(),
            nn.Linear(lnn_hidden, lnn_hidden),
            nn.Tanh(),
            nn.Linear(lnn_hidden, 1),
        )

        self.obs_encoder = nn.GRU(
            input_size=dim * num_objects,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        self.state_init = nn.Linear(hidden_dim, self.state_dim)

    def _get_initial_state(self, observed_positions):
        B, T, N, D = observed_positions.shape
        x = observed_positions.reshape(B, T, N * D)

        _, h_n = self.obs_encoder(x)
        init_state = self.state_init(h_n[-1])

        return init_state

    def _lagrangian(self, state):
        return self.lagrangian_net(state).squeeze(-1)

    def _euler_lagrange_accel(self, state):
        eps = 1e-4

        dL = torch.zeros_like(state)
        for i in range(self.state_dim):
            state_plus = state.clone()
            state_plus[:, i] = state_plus[:, i] + eps
            state_minus = state.clone()
            state_minus[:, i] = state_minus[:, i] - eps
            dL[:, i] = (self._lagrangian(state_plus) - self._lagrangian(state_minus)) / (2 * eps)

        dL_dq = dL[:, :self.q_dim]
        dL_dqdot = dL[:, self.q_dim:]

        q_dot = state[:, self.q_dim:]

        M = torch.zeros(state.shape[0], self.q_dim, self.q_dim, device=state.device)
        for i in range(self.q_dim):
            d2L = torch.zeros_like(state)
            for j in range(self.state_dim):
                state_plus = state.clone()
                state_plus[:, j] = state_plus[:, j] + eps
                state_minus = state.clone()
                state_minus[:, j] = state_minus[:, j] - eps
                dL_plus = torch.zeros_like(state)
                dL_minus = torch.zeros_like(state)
                for k in range(self.state_dim):
                    sp2 = state_plus.clone()
                    sp2[:, k] = sp2[:, k] + eps
                    sm2 = state_plus.clone()
                    sm2[:, k] = sm2[:, k] - eps
                    dL_plus[:, k] = (self._lagrangian(sp2) - self._lagrangian(sm2)) / (2 * eps)
                M[:, i, :] = dL_plus[:, self.q_dim:] - dL_dqdot

        M_reg = M + 1e-3 * torch.eye(self.q_dim, device=state.device).unsqueeze(0)

        rhs = dL_dq - torch.bmm(M, q_dot.unsqueeze(-1)).squeeze(-1)

        try:
            q_ddot = torch.linalg.solve(M_reg, rhs)
        except Exception:
            q_ddot = torch.zeros_like(rhs)

        q_ddot = torch.clamp(q_ddot, -10.0, 10.0)

        return q_ddot

    def _step(self, state, dt):
        q = state[:, :self.q_dim]
        q_dot = state[:, self.q_dim:]

        q_ddot = self._euler_lagrange_accel(state)

        new_q = q + q_dot * dt + 0.5 * q_ddot * dt * dt
        new_q_dot = q_dot + q_ddot * dt

        return torch.cat([new_q, new_q_dot], dim=-1)

    def _integrate(self, initial_state, n_steps, dt):
        state = initial_state
        positions = [state[:, :self.q_dim]]

        for _ in range(n_steps):
            state = self._step(state, dt)
            positions.append(state[:, :self.q_dim])

        return torch.stack(positions[1:], dim=1)

    def forward(self, observed_positions):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)

        init_state = self._get_initial_state(observed_positions)
        init_state = init_state.requires_grad_(True)
        positions_flat = self._integrate(init_state, self.t_pred, self.dt)

        B = observed_positions.shape[0]
        traj_out = positions_flat.reshape(B, self.t_pred, self.num_objects, self.dim)
        return traj_out

    def compute_loss(self, observed_positions, future_positions, identity_labels,
                     observed_features=None, future_features=None):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        if isinstance(future_positions, np.ndarray):
            future_positions = torch.FloatTensor(future_positions)

        init_state = self._get_initial_state(observed_positions)
        positions_flat = self._integrate(init_state, self.t_pred, self.dt)

        B = observed_positions.shape[0]
        pred_traj = positions_flat.reshape(B, self.t_pred, self.num_objects, self.dim)

        mse_loss = F.mse_loss(pred_traj, future_positions)

        total_loss = mse_loss
        return total_loss, mse_loss, torch.tensor(0.0), torch.tensor(0.0)

    def predict_future(self, observed_positions, observed_features=None):
        self.eval()
        with torch.enable_grad():
            traj_pred = self.forward(observed_positions)
        if isinstance(observed_positions, np.ndarray):
            return traj_pred.detach().cpu().numpy()
        return traj_pred.detach()

    def predict_identity(self, observed_positions, observed_features=None,
                         test_future=None, future_features=None,
                         future_positions=None):
        self.eval()
        with torch.enable_grad():
            pred_traj = self.forward(observed_positions)

        if isinstance(pred_traj, torch.Tensor):
            pred_traj = pred_traj.detach().cpu().numpy()

        if future_positions is None and test_future is not None:
            future_positions = test_future
        if isinstance(future_positions, torch.Tensor):
            future_positions = future_positions.cpu().numpy()

        B = pred_traj.shape[0]
        N = self.num_objects
        results = np.zeros((B, N), dtype=int)

        if future_positions is None:
            return results

        for b in range(B):
            pred_mean = pred_traj[b].mean(axis=0)
            actual_mean = future_positions[b].mean(axis=0)

            used = set()
            for i in range(N):
                dists = np.linalg.norm(actual_mean[i] - pred_mean, axis=-1)
                dists_m = dists.copy()
                for j in used:
                    dists_m[j] = float('inf')
                best = np.argmin(dists_m)
                results[b, i] = best
                used.add(best)

        return results
