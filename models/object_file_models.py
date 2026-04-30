"""
SVT-v4 Models: TrajectoryOnly + MinimalObjectFile

TrajectoryOnlyAssignment: identity via nearest predicted position
MinimalObjectFileMechanism: rule-based object file with occlusion handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TrajectoryOnlyAssignment(nn.Module):
    def __init__(self, t_obs=10, t_pred=20, num_objects=2,
                 hidden_dim=256, num_layers=3, dropout=0.1):
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
        self.shared = nn.Sequential(*layers)
        self.trajectory_head = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, observed_positions):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        B = observed_positions.shape[0]
        x = observed_positions.reshape(B, -1)
        shared_out = self.shared(x)
        traj_out = self.trajectory_head(shared_out).reshape(B, self.t_pred, self.num_objects, 2)
        return traj_out

    def compute_loss(self, observed_positions, future_positions, identity_labels,
                     observed_features=None, future_features=None):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        if isinstance(future_positions, np.ndarray):
            future_positions = torch.FloatTensor(future_positions)

        traj_pred = self.forward(observed_positions)
        mse_loss = F.mse_loss(traj_pred, future_positions)
        return mse_loss, mse_loss, torch.tensor(0.0), torch.tensor(0.0)

    def predict_future(self, observed_positions, observed_features=None):
        self.eval()
        with torch.no_grad():
            traj_pred = self.forward(observed_positions)
        if isinstance(observed_positions, np.ndarray):
            return traj_pred.cpu().numpy()
        return traj_pred

    def predict_identity(self, observed_positions, observed_features=None,
                         test_future=None, future_features=None,
                         future_positions=None):
        self.eval()
        with torch.no_grad():
            pred_traj = self.forward(observed_positions)

        if isinstance(pred_traj, torch.Tensor):
            pred_traj = pred_traj.cpu().numpy()

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


class MultiTaskTrajectoryPredictor(nn.Module):
    def __init__(self, t_obs=10, t_pred=20, num_objects=2,
                 hidden_dim=256, num_layers=3, dropout=0.1):
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
        self.shared = nn.Sequential(*layers)
        self.trajectory_head = nn.Linear(hidden_dim, self.output_dim)
        self.velocity_head = nn.Linear(hidden_dim, self.output_dim)
        self.accel_head = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, observed_positions):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        B = observed_positions.shape[0]
        x = observed_positions.reshape(B, -1)
        shared_out = self.shared(x)
        shape = (B, self.t_pred, self.num_objects, 2)
        traj_out = self.trajectory_head(shared_out).reshape(shape)
        vel_out = self.velocity_head(shared_out).reshape(shape)
        accel_out = self.accel_head(shared_out).reshape(shape)
        return traj_out, vel_out, accel_out

    def compute_loss(self, observed_positions, future_positions, identity_labels,
                     observed_features=None, future_features=None):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        if isinstance(future_positions, np.ndarray):
            future_positions = torch.FloatTensor(future_positions)

        traj_pred, vel_pred, accel_pred = self.forward(observed_positions)

        mse_pos = F.mse_loss(traj_pred, future_positions)

        T_pred = self.t_pred
        if T_pred > 1:
            future_vel = future_positions[:, 1:, :, :] - future_positions[:, :-1, :, :]
            vel_target = torch.zeros_like(future_positions)
            vel_target[:, 1:, :, :] = future_vel
            vel_target[:, 0, :, :] = future_vel[:, 0, :, :]
            mse_vel = F.mse_loss(vel_pred[:, 1:, :, :], future_vel)
        else:
            mse_vel = torch.tensor(0.0, device=traj_pred.device)

        if T_pred > 2:
            future_accel = future_vel[:, 1:, :, :] - future_vel[:, :-1, :, :]
            mse_accel = F.mse_loss(accel_pred[:, 2:, :, :], future_accel)
        else:
            mse_accel = torch.tensor(0.0, device=traj_pred.device)

        total_loss = mse_pos + 0.1 * mse_vel + 0.01 * mse_accel
        return total_loss, mse_pos, torch.tensor(0.0), torch.tensor(0.0)

    def predict_future(self, observed_positions, observed_features=None):
        self.eval()
        with torch.no_grad():
            traj_pred, _, _ = self.forward(observed_positions)
        if isinstance(observed_positions, np.ndarray):
            return traj_pred.cpu().numpy()
        return traj_pred

    def predict_identity(self, observed_positions, observed_features=None,
                         test_future=None, future_features=None,
                         future_positions=None):
        self.eval()
        with torch.no_grad():
            pred_traj, _, _ = self.forward(observed_positions)

        if isinstance(pred_traj, torch.Tensor):
            pred_traj = pred_traj.cpu().numpy()

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


class MinimalObjectFileMechanism:
    def __init__(self, num_objects=2, feature_dim=2,
                 feature_weight=1.0, traj_weight=1.0,
                 occlusion_dist_threshold=5.0):
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.feature_weight = feature_weight
        self.traj_weight = traj_weight
        self.occlusion_dist_threshold = occlusion_dist_threshold

    def _compute_scores(self, fut_pos, fut_feat, files, N):
        scores = np.full((N, N), -float('inf'))
        feat_scores = np.full((N, N), -float('inf'))
        traj_scores = np.full((N, N), -float('inf'))

        for i in range(N):
            for j in range(N):
                fs = 0.0
                if fut_feat is not None and fut_feat[i] is not None:
                    fkey = files[j]['identity_key']
                    nf = np.linalg.norm(fut_feat[i])
                    nk = np.linalg.norm(fkey)
                    if nf > 1e-8 and nk > 1e-8:
                        fs = np.dot(fut_feat[i], fkey) / (nf * nk)

                predicted_pos = files[j]['last_pos'] + files[j]['last_vel']
                dist = np.linalg.norm(fut_pos[i] - predicted_pos)
                ts = -dist

                scores[i, j] = self.feature_weight * fs + self.traj_weight * ts
                feat_scores[i, j] = fs
                traj_scores[i, j] = ts

        return scores, feat_scores, traj_scores

    def _greedy_match(self, scores, N, excluded_rows=None):
        if excluded_rows is None:
            excluded_rows = set()
        assignment = np.full(N, -1, dtype=int)
        used_cols = set()
        for i in range(N):
            if i in excluded_rows:
                continue
            valid = scores[i].copy()
            for j in used_cols:
                valid[j] = -float('inf')
            if valid.max() > -float('inf'):
                best = np.argmax(valid)
                assignment[i] = best
                used_cols.add(best)
        return assignment

    def predict_identity(self, observed_positions, observed_features,
                         future_positions, future_features,
                         occlusion_mask=None, return_trace=False):
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
        all_traces = []

        for b in range(B):
            files = []
            for i in range(N):
                files.append({
                    'identity_key': observed_features[b, 0, i, :].copy(),
                    'last_pos': observed_positions[b, -1, i, :].copy(),
                    'last_vel': (observed_positions[b, -1, i, :] -
                                 observed_positions[b, -2, i, :]).copy(),
                    'confidence': 1.0,
                })

            trace = []
            T_pred = future_positions.shape[1]

            for t in range(T_pred):
                is_occluded = np.zeros(N, dtype=bool)
                if occlusion_mask is not None:
                    is_occluded = occlusion_mask[b, t, :]
                else:
                    for i in range(N):
                        for j in range(i + 1, N):
                            d = np.linalg.norm(future_positions[b, t, i, :] -
                                               future_positions[b, t, j, :])
                            if d < self.occlusion_dist_threshold:
                                is_occluded[i] = True
                                is_occluded[j] = True

                fut_pos = [future_positions[b, t, i, :] for i in range(N)]
                fut_feat = None
                if future_features is not None:
                    fut_feat = [future_features[b, t, i, :] if not is_occluded[i] else None
                                for i in range(N)]

                scores, feat_s, traj_s = self._compute_scores(fut_pos, fut_feat, files, N)

                visible = [i for i in range(N) if not is_occluded[i]]
                if visible:
                    assignment = self._greedy_match(scores, N, excluded_rows=set(range(N)) - set(visible))
                    for i in visible:
                        j = assignment[i]
                        if j >= 0:
                            old_pos = files[j]['last_pos'].copy()
                            files[j]['last_vel'] = future_positions[b, t, i, :] - old_pos
                            files[j]['last_pos'] = future_positions[b, t, i, :].copy()

                            if fut_feat[i] is not None:
                                nf = np.linalg.norm(fut_feat[i])
                                nk = np.linalg.norm(files[j]['identity_key'])
                                if nf > 1e-8 and nk > 1e-8:
                                    sim = np.dot(fut_feat[i], files[j]['identity_key']) / (nf * nk)
                                    if sim > 0.9:
                                        files[j]['identity_key'] = fut_feat[i].copy()

                if return_trace:
                    for i in range(N):
                        trace.append({
                            'episode_idx': b,
                            'timestep': t,
                            'object_idx': i,
                            'identity_key': files[i]['identity_key'].tolist(),
                            'last_pos': files[i]['last_pos'].tolist(),
                            'occluded': bool(is_occluded[i]),
                            'confidence': files[i]['confidence'],
                        })

            final_fut_pos = [future_positions[b, -1, i, :] for i in range(N)]
            final_fut_feat = None
            if future_features is not None:
                final_fut_feat = [future_features[b, -1, i, :] for i in range(N)]

            scores, _, _ = self._compute_scores(final_fut_pos, final_fut_feat, files, N)
            assignment = self._greedy_match(scores, N)
            results[b] = assignment

            if return_trace:
                all_traces.append(trace)

        if single:
            results = results[0]

        if return_trace:
            return results, all_traces
        return results

    def predict_identity_with_conflict_info(self, observed_positions, observed_features,
                                             future_positions, future_features):
        single = observed_positions.ndim == 3
        if single:
            observed_positions = observed_positions[np.newaxis]
            observed_features = observed_features[np.newaxis]
            future_positions = future_positions[np.newaxis]
            future_features = future_features[np.newaxis] if future_features is not None else None

        B = observed_positions.shape[0]
        N = self.num_objects
        results = np.zeros((B, N), dtype=int)
        trusted_feature = np.zeros(B, dtype=bool)

        for b in range(B):
            files = []
            for i in range(N):
                files.append({
                    'identity_key': observed_features[b, 0, i, :].copy(),
                    'last_pos': observed_positions[b, -1, i, :].copy(),
                    'last_vel': (observed_positions[b, -1, i, :] -
                                 observed_positions[b, -2, i, :]).copy(),
                    'confidence': 1.0,
                })

            final_fut_pos = [future_positions[b, -1, i, :] for i in range(N)]
            final_fut_feat = [future_features[b, -1, i, :] for i in range(N)] if future_features is not None else None

            scores, feat_scores, traj_scores = self._compute_scores(
                final_fut_pos, final_fut_feat, files, N)

            feat_assignment = self._greedy_match(feat_scores, N)
            traj_assignment = self._greedy_match(traj_scores, N)

            results[b] = self._greedy_match(scores, N)

            feat_agrees = np.array_equal(feat_assignment, traj_assignment)
            if not feat_agrees:
                trusted_feature[b] = np.array_equal(results[b], feat_assignment)
            else:
                trusted_feature[b] = True

        if single:
            results = results[0]
            trusted_feature = trusted_feature[0]

        return results, trusted_feature


class ImprovedObjectFile:
    def __init__(self, traj_model=None, num_objects=2, feature_dim=2,
                 conflict_threshold=0.3, occlusion_decay=0.95,
                 reappearance_boost=0.8):
        self.traj_model = traj_model
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.conflict_threshold = conflict_threshold
        self.occlusion_decay = occlusion_decay
        self.reappearance_boost = reappearance_boost

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
                pred = pred.cpu().numpy()
        return pred

    def _compute_feature_score(self, feat_vec, identity_key):
        nf = np.linalg.norm(feat_vec)
        nk = np.linalg.norm(identity_key)
        if nf < 1e-8 or nk < 1e-8:
            return 0.0, 0.0
        sim = np.dot(feat_vec, identity_key) / (nf * nk)
        confidence = max(0.0, sim)
        return float(sim), float(confidence)

    def _compute_trajectory_score(self, fut_pos, file_state):
        if 'predicted_positions' in file_state:
            pred_pos = file_state['predicted_positions'][-1]
        else:
            pred_pos = file_state['last_pos'] + file_state.get('last_vel', np.zeros(2))
        dist = np.linalg.norm(fut_pos - pred_pos)
        score = -dist
        confidence = np.exp(-dist / 10.0)
        return float(score), float(confidence)

    def _compute_adaptive_weights(self, feat_conf, traj_conf, feat_score, traj_score):
        if feat_conf < 1e-8:
            return 0.0, 1.0
        if traj_conf < 1e-8:
            return 1.0, 0.0

        feat_best_j = None
        traj_best_j = None

        w_f = feat_conf / (feat_conf + traj_conf + 1e-8)
        w_t = traj_conf / (feat_conf + traj_conf + 1e-8)

        return w_f, w_t

    def _detect_conflict(self, feat_assignment, traj_assignment, feat_scores, traj_scores, N):
        if np.array_equal(feat_assignment, traj_assignment):
            return False, "agreement"

        feat_max_scores = np.full(N, -float('inf'))
        traj_max_scores = np.full(N, -float('inf'))
        for i in range(N):
            if feat_assignment[i] >= 0:
                feat_max_scores[i] = feat_scores[i, feat_assignment[i]]
            if traj_assignment[i] >= 0:
                traj_max_scores[i] = traj_scores[i, traj_assignment[i]]

        feat_margin = np.mean(feat_max_scores[feat_max_scores > -float('inf')]) if np.any(feat_max_scores > -float('inf')) else 0
        traj_margin = np.mean(traj_max_scores[traj_max_scores > -float('inf')]) if np.any(traj_max_scores > -float('inf')) else 0

        if abs(feat_margin - traj_margin) < self.conflict_threshold:
            return True, "uncertain"
        elif feat_margin > traj_margin:
            return True, "feature"
        else:
            return True, "trajectory"

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
        all_traces = []
        all_confidences = []
        all_chosen_sources = []

        pred_traj_all = self._get_predicted_trajectory(observed_positions)

        for b in range(B):
            files = []
            for i in range(N):
                f = {
                    'identity_key': observed_features[b, 0, i, :].copy(),
                    'last_pos': observed_positions[b, -1, i, :].copy(),
                    'last_vel': (observed_positions[b, -1, i, :] -
                                 observed_positions[b, -2, i, :]).copy(),
                    'feature_confidence': 1.0,
                    'trajectory_confidence': 1.0,
                    'total_confidence': 1.0,
                    'occluded_steps': 0,
                }
                if pred_traj_all is not None:
                    f['predicted_positions'] = pred_traj_all[b, :, i, :]
                files.append(f)

            trace = []
            ep_confidences = []
            ep_sources = []
            T_pred = future_positions.shape[1]

            for t in range(T_pred):
                is_occluded = np.zeros(N, dtype=bool)
                if occlusion_mask is not None:
                    is_occluded = occlusion_mask[b, t, :]
                else:
                    for i in range(N):
                        for j in range(i + 1, N):
                            d = np.linalg.norm(future_positions[b, t, i, :] -
                                               future_positions[b, t, j, :])
                            if d < 3.0:
                                is_occluded[i] = True
                                is_occluded[j] = True

                fut_pos = [future_positions[b, t, i, :] for i in range(N)]
                fut_feat = None
                if future_features is not None:
                    fut_feat = [future_features[b, t, i, :] if not is_occluded[i] else None
                                for i in range(N)]

                feat_scores = np.full((N, N), -float('inf'))
                traj_scores = np.full((N, N), -float('inf'))
                combined_scores = np.full((N, N), -float('inf'))
                feat_confs = np.zeros(N)
                traj_confs = np.zeros(N)

                for i in range(N):
                    for j in range(N):
                        fs, fc = 0.0, 0.0
                        if fut_feat is not None and fut_feat[i] is not None:
                            fs, fc = self._compute_feature_score(
                                fut_feat[i], files[j]['identity_key'])
                            feat_confs[i] = max(feat_confs[i], fc)

                        ts, tc = self._compute_trajectory_score(fut_pos[i], files[j])
                        traj_confs[i] = max(traj_confs[i], tc)

                        feat_scores[i, j] = fs
                        traj_scores[i, j] = ts

                visible = [i for i in range(N) if not is_occluded[i]]
                for i in visible:
                    w_f, w_t = self._compute_adaptive_weights(
                        feat_confs[i], traj_confs[i],
                        feat_scores[i].max(), traj_scores[i].max())
                    combined_scores[i] = w_f * feat_scores[i] + w_t * traj_scores[i]

                if visible:
                    used = set()
                    assignment = np.full(N, -1, dtype=int)
                    for i in visible:
                        valid = combined_scores[i].copy()
                        for j in used:
                            valid[j] = -float('inf')
                        if valid.max() > -float('inf'):
                            best = np.argmax(valid)
                            assignment[i] = best
                            used.add(best)

                    feat_assign = np.full(N, -1, dtype=int)
                    traj_assign = np.full(N, -1, dtype=int)
                    used_f = set()
                    used_t = set()
                    for i in visible:
                        vf = feat_scores[i].copy()
                        vt = traj_scores[i].copy()
                        for j in used_f:
                            vf[j] = -float('inf')
                        for j in used_t:
                            vt[j] = -float('inf')
                        if vf.max() > -float('inf'):
                            feat_assign[i] = np.argmax(vf)
                            used_f.add(feat_assign[i])
                        if vt.max() > -float('inf'):
                            traj_assign[i] = np.argmax(vt)
                            used_t.add(traj_assign[i])

                    is_conflict, source = self._detect_conflict(
                        feat_assign, traj_assign, feat_scores, traj_scores, N)

                    for i in visible:
                        j = assignment[i]
                        if j >= 0:
                            old_pos = files[j]['last_pos'].copy()
                            files[j]['last_vel'] = future_positions[b, t, i, :] - old_pos
                            files[j]['last_pos'] = future_positions[b, t, i, :].copy()

                            if fut_feat is not None and fut_feat[i] is not None:
                                fs, fc = self._compute_feature_score(
                                    fut_feat[i], files[j]['identity_key'])
                                if fc > 0.9:
                                    files[j]['identity_key'] = fut_feat[i].copy()
                                    files[j]['feature_confidence'] = min(1.0, fc + 0.1)
                                else:
                                    files[j]['feature_confidence'] = fc * 0.9

                            files[j]['trajectory_confidence'] = traj_confs[i]

                            if not is_conflict:
                                files[j]['total_confidence'] = min(1.0,
                                    (feat_confs[i] + traj_confs[i]) / 2 + 0.1)
                            else:
                                files[j]['total_confidence'] = max(0.0,
                                    abs(feat_confs[i] - traj_confs[i]))

                for i in range(N):
                    if is_occluded[i]:
                        files[i]['occluded_steps'] += 1
                        files[i]['trajectory_confidence'] *= self.occlusion_decay
                        files[i]['total_confidence'] *= self.occlusion_decay
                    elif files[i]['occluded_steps'] > 0:
                        files[i]['trajectory_confidence'] = min(1.0,
                            files[i]['trajectory_confidence'] + self.reappearance_boost * 0.5)
                        files[i]['total_confidence'] = min(1.0,
                            files[i]['total_confidence'] + self.reappearance_boost * 0.3)
                        files[i]['occluded_steps'] = 0

                if return_trace:
                    for i in range(N):
                        trace.append({
                            'episode_idx': b,
                            'timestep': t,
                            'object_idx': i,
                            'identity_key_0': files[i]['identity_key'][0],
                            'identity_key_1': files[i]['identity_key'][1],
                            'last_pos_x': files[i]['last_pos'][0],
                            'last_pos_y': files[i]['last_pos'][1],
                            'occluded': bool(is_occluded[i]),
                            'feature_confidence': files[i]['feature_confidence'],
                            'trajectory_confidence': files[i]['trajectory_confidence'],
                            'total_confidence': files[i]['total_confidence'],
                        })

            final_fut_pos = [future_positions[b, -1, i, :] for i in range(N)]
            final_fut_feat = [future_features[b, -1, i, :] for i in range(N)] if future_features is not None else None

            feat_scores_f = np.full((N, N), -float('inf'))
            traj_scores_f = np.full((N, N), -float('inf'))
            feat_confs_f = np.zeros(N)
            traj_confs_f = np.zeros(N)

            for i in range(N):
                for j in range(N):
                    fs, fc = 0.0, 0.0
                    if final_fut_feat is not None and final_fut_feat[i] is not None:
                        fs, fc = self._compute_feature_score(
                            final_fut_feat[i], files[j]['identity_key'])
                        feat_confs_f[i] = max(feat_confs_f[i], fc)
                    feat_scores_f[i, j] = fs

                    ts, tc = self._compute_trajectory_score(final_fut_pos[i], files[j])
                    traj_confs_f[i] = max(traj_confs_f[i], tc)
                    traj_scores_f[i, j] = ts

            combined_f = np.full((N, N), -float('inf'))
            for i in range(N):
                w_f, w_t = self._compute_adaptive_weights(
                    feat_confs_f[i], traj_confs_f[i],
                    feat_scores_f[i].max(), traj_scores_f[i].max())
                combined_f[i] = w_f * feat_scores_f[i] + w_t * traj_scores_f[i]

            used = set()
            for i in range(N):
                valid = combined_f[i].copy()
                for j in used:
                    valid[j] = -float('inf')
                best = np.argmax(valid)
                results[b, i] = best
                used.add(best)

            feat_assign_f = np.full(N, -1, dtype=int)
            traj_assign_f = np.full(N, -1, dtype=int)
            used_f = set()
            used_t = set()
            for i in range(N):
                vf = feat_scores_f[i].copy()
                vt = traj_scores_f[i].copy()
                for j in used_f:
                    vf[j] = -float('inf')
                for j in used_t:
                    vt[j] = -float('inf')
                if vf.max() > -float('inf'):
                    feat_assign_f[i] = np.argmax(vf)
                    used_f.add(feat_assign_f[i])
                if vt.max() > -float('inf'):
                    traj_assign_f[i] = np.argmax(vt)
                    used_t.add(traj_assign_f[i])

            is_conflict, source = self._detect_conflict(
                feat_assign_f, traj_assign_f, feat_scores_f, traj_scores_f, N)

            ep_confidences.append({
                'feat_confs': feat_confs_f.tolist(),
                'traj_confs': traj_confs_f.tolist(),
                'total_confs': [files[i]['total_confidence'] for i in range(N)],
            })
            ep_sources.append(source)

            if return_trace:
                all_traces.append(trace)

        all_confidences.append(ep_confidences)
        all_chosen_sources.append(ep_sources)

        if single:
            results = results[0]

        if return_conflict_info:
            return results, all_confidences, all_chosen_sources
        if return_trace:
            return results, all_traces
        return results


class ConflictFirstObjectFile:
    def __init__(self, traj_model=None, num_objects=2, feature_dim=2,
                 strategy="margin_gated",
                 conflict_margin_threshold=0.1,
                 trajectory_confidence_threshold=0.3,
                 occlusion_decay=0.95,
                 reappearance_boost=0.5,
                 high_conflict_abstain_threshold=2.0,
                 traj_margin_advantage=1.5):
        self.traj_model = traj_model
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.strategy = strategy
        self.conflict_margin_threshold = conflict_margin_threshold
        self.trajectory_confidence_threshold = trajectory_confidence_threshold
        self.occlusion_decay = occlusion_decay
        self.reappearance_boost = reappearance_boost
        self.high_conflict_abstain_threshold = high_conflict_abstain_threshold
        self.traj_margin_advantage = traj_margin_advantage

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
                pred = pred.cpu().numpy()
        return pred

    def _compute_feature_similarity(self, feat_vec, identity_key):
        nf = np.linalg.norm(feat_vec)
        nk = np.linalg.norm(identity_key)
        if nf < 1e-8 or nk < 1e-8:
            return 0.0, True
        sim = np.dot(feat_vec, identity_key) / (nf * nk)
        return float(sim), False

    def _compute_trajectory_distance(self, fut_pos, file_state):
        if 'predicted_positions' in file_state:
            pred_pos = file_state['predicted_positions'][-1]
        else:
            pred_pos = file_state['last_pos'] + file_state.get('last_vel', np.zeros(2))
        dist = np.linalg.norm(fut_pos - pred_pos)
        return float(dist)

    def _greedy_assignment(self, scores, N, excluded_rows=None):
        if excluded_rows is None:
            excluded_rows = set()
        assignment = np.full(N, -1, dtype=int)
        used_cols = set()
        for i in range(N):
            if i in excluded_rows:
                continue
            valid = scores[i].copy()
            for j in used_cols:
                valid[j] = -float('inf')
            if valid.max() > -float('inf'):
                best = np.argmax(valid)
                assignment[i] = best
                used_cols.add(best)
        return assignment

    def _compute_margin(self, scores, assignment, N):
        margins = []
        for i in range(N):
            if assignment[i] < 0:
                continue
            best_score = scores[i, assignment[i]]
            others = [scores[i, j] for j in range(N) if j != assignment[i] and scores[i, j] > -float('inf')]
            if others:
                margin = best_score - max(others)
            else:
                margin = best_score
            margins.append(margin)
        return float(np.mean(margins)) if margins else 0.0

    def _resolve_conflict(self, feat_assignment, traj_assignment,
                          feat_scores, traj_scores, feat_missing_flags,
                          occlusion_flags, trajectory_uncertainties, N):
        conflict = not np.array_equal(feat_assignment, traj_assignment)

        if not conflict:
            return feat_assignment.copy(), "agreement", 0.9, False

        feat_margin = self._compute_margin(feat_scores, feat_assignment, N)
        traj_margin = self._compute_margin(traj_scores, traj_assignment, N)

        any_feature_missing = any(feat_missing_flags)
        any_occluded = any(occlusion_flags)
        avg_traj_uncertainty = float(np.mean(trajectory_uncertainties)) if len(trajectory_uncertainties) > 0 else 0.0
        traj_confidence = np.exp(-avg_traj_uncertainty / 10.0)

        if self.strategy == "prefer_trajectory_on_conflict":
            chosen = traj_assignment.copy()
            source = "trajectory"
            confidence = max(0.3, traj_confidence * 0.8)
            abstain = False

        elif self.strategy == "prefer_feature_on_low_trajectory_confidence":
            if traj_confidence < self.trajectory_confidence_threshold:
                chosen = feat_assignment.copy()
                source = "feature"
                confidence = max(0.3, min(0.7, feat_margin))
                abstain = False
            else:
                chosen = traj_assignment.copy()
                source = "trajectory"
                confidence = max(0.3, traj_confidence * 0.8)
                abstain = False

        elif self.strategy == "abstain_on_high_conflict":
            both_margins_small = (feat_margin < self.high_conflict_abstain_threshold and
                                  traj_margin < self.high_conflict_abstain_threshold)
            if both_margins_small:
                chosen = traj_assignment.copy()
                source = "uncertain"
                confidence = 0.2
                abstain = True
            elif traj_margin > feat_margin:
                chosen = traj_assignment.copy()
                source = "trajectory"
                confidence = max(0.3, min(0.85, traj_margin / 10.0))
                abstain = False
            else:
                chosen = feat_assignment.copy()
                source = "feature"
                confidence = max(0.3, min(0.85, feat_margin))
                abstain = False

        elif self.strategy == "margin_gated":
            if traj_margin > feat_margin * self.traj_margin_advantage:
                chosen = traj_assignment.copy()
                source = "trajectory"
                confidence = max(0.2, traj_confidence * 0.7)
                abstain = False
            elif feat_margin > traj_margin * self.traj_margin_advantage:
                chosen = feat_assignment.copy()
                source = "feature"
                confidence = max(0.2, min(0.7, feat_margin * 0.7))
                abstain = False
            else:
                chosen = traj_assignment.copy()
                source = "uncertain"
                confidence = 0.15
                abstain = True
        else:
            chosen = traj_assignment.copy()
            source = "trajectory"
            confidence = max(0.3, traj_confidence * 0.8)
            abstain = False

        if any_feature_missing and source == "feature":
            chosen = traj_assignment.copy()
            source = "trajectory_fallback"
            confidence = max(0.2, traj_confidence * 0.6)
            abstain = False

        if any_occluded and source == "feature":
            chosen = traj_assignment.copy()
            source = "trajectory_occlusion"
            confidence = max(0.2, traj_confidence * 0.5)
            abstain = False

        return chosen, source, confidence, abstain

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
        all_traces = []
        all_confidences = []
        all_sources = []
        all_abstain_flags = []

        pred_traj_all = self._get_predicted_trajectory(observed_positions)

        for b in range(B):
            files = []
            for i in range(N):
                f = {
                    'identity_key': observed_features[b, 0, i, :].copy(),
                    'last_pos': observed_positions[b, -1, i, :].copy(),
                    'last_vel': (observed_positions[b, -1, i, :] -
                                 observed_positions[b, -2, i, :]).copy(),
                    'feature_confidence': 1.0,
                    'trajectory_confidence': 1.0,
                    'total_confidence': 1.0,
                    'occluded_steps': 0,
                }
                if pred_traj_all is not None:
                    f['predicted_positions'] = pred_traj_all[b, :, i, :]
                files.append(f)

            trace = []
            T_pred = future_positions.shape[1]

            for t in range(T_pred):
                is_occluded = np.zeros(N, dtype=bool)
                if occlusion_mask is not None:
                    is_occluded = occlusion_mask[b, t, :]
                else:
                    for i in range(N):
                        for j in range(i + 1, N):
                            d = np.linalg.norm(future_positions[b, t, i, :] -
                                               future_positions[b, t, j, :])
                            if d < 3.0:
                                is_occluded[i] = True
                                is_occluded[j] = True

                fut_pos = [future_positions[b, t, i, :] for i in range(N)]
                fut_feat = None
                if future_features is not None:
                    fut_feat = [future_features[b, t, i, :] if not is_occluded[i] else None
                                for i in range(N)]

                feat_scores = np.full((N, N), -float('inf'))
                traj_scores = np.full((N, N), -float('inf'))
                feat_missing_flags = [False] * N
                traj_uncertainties = [0.0] * N

                for i in range(N):
                    for j in range(N):
                        sim, missing = self._compute_feature_similarity(
                            fut_feat[i] if fut_feat is not None and fut_feat[i] is not None else np.zeros(self.feature_dim),
                            files[j]['identity_key'])
                        if fut_feat is None or fut_feat[i] is None:
                            feat_missing_flags[i] = True
                            sim = 0.0
                        feat_scores[i, j] = sim

                        dist = self._compute_trajectory_distance(fut_pos[i], files[j])
                        traj_scores[i, j] = -dist
                        if j == 0:
                            traj_uncertainties[i] = dist

                visible = [i for i in range(N) if not is_occluded[i]]

                if visible:
                    feat_assignment = self._greedy_assignment(feat_scores, N)
                    traj_assignment = self._greedy_assignment(traj_scores, N)

                    occlusion_flags = [is_occluded[i] for i in range(N)]

                    chosen, source, confidence, abstain = self._resolve_conflict(
                        feat_assignment, traj_assignment,
                        feat_scores, traj_scores,
                        feat_missing_flags, occlusion_flags,
                        traj_uncertainties, N)

                    for i in visible:
                        j = chosen[i]
                        if j >= 0:
                            old_pos = files[j]['last_pos'].copy()
                            files[j]['last_vel'] = future_positions[b, t, i, :] - old_pos
                            files[j]['last_pos'] = future_positions[b, t, i, :].copy()

                            if fut_feat is not None and fut_feat[i] is not None:
                                sim, _ = self._compute_feature_similarity(
                                    fut_feat[i], files[j]['identity_key'])
                                if sim > 0.9:
                                    files[j]['identity_key'] = fut_feat[i].copy()
                                    files[j]['feature_confidence'] = min(1.0, sim + 0.05)
                                else:
                                    files[j]['feature_confidence'] = sim * 0.9

                            files[j]['trajectory_confidence'] = np.exp(-traj_uncertainties[min(i, len(traj_uncertainties)-1)] / 10.0)
                            files[j]['total_confidence'] = confidence

                for i in range(N):
                    if is_occluded[i]:
                        files[i]['occluded_steps'] += 1
                        files[i]['trajectory_confidence'] *= self.occlusion_decay
                        files[i]['total_confidence'] *= self.occlusion_decay
                    elif files[i]['occluded_steps'] > 0:
                        files[i]['trajectory_confidence'] = min(1.0,
                            files[i]['trajectory_confidence'] + self.reappearance_boost * 0.5)
                        files[i]['total_confidence'] = min(1.0,
                            files[i]['total_confidence'] + self.reappearance_boost * 0.3)
                        files[i]['occluded_steps'] = 0

                if return_trace:
                    for i in range(N):
                        trace.append({
                            'episode_idx': b,
                            'timestep': t,
                            'object_idx': i,
                            'identity_key_0': files[i]['identity_key'][0],
                            'identity_key_1': files[i]['identity_key'][1],
                            'last_pos_x': files[i]['last_pos'][0],
                            'last_pos_y': files[i]['last_pos'][1],
                            'occluded': bool(is_occluded[i]),
                            'feature_confidence': files[i]['feature_confidence'],
                            'trajectory_confidence': files[i]['trajectory_confidence'],
                            'total_confidence': files[i]['total_confidence'],
                        })

            final_fut_pos = [future_positions[b, -1, i, :] for i in range(N)]
            final_fut_feat = [future_features[b, -1, i, :] for i in range(N)] if future_features is not None else None

            feat_scores_f = np.full((N, N), -float('inf'))
            traj_scores_f = np.full((N, N), -float('inf'))
            feat_missing_f = [False] * N
            traj_unc_f = [0.0] * N

            for i in range(N):
                for j in range(N):
                    sim, missing = self._compute_feature_similarity(
                        final_fut_feat[i] if final_fut_feat is not None and final_fut_feat[i] is not None else np.zeros(self.feature_dim),
                        files[j]['identity_key'])
                    if final_fut_feat is None or final_fut_feat[i] is None:
                        feat_missing_f[i] = True
                        sim = 0.0
                    feat_scores_f[i, j] = sim

                    dist = self._compute_trajectory_distance(final_fut_pos[i], files[j])
                    traj_scores_f[i, j] = -dist
                    if j == 0:
                        traj_unc_f[i] = dist

            feat_assignment_f = self._greedy_assignment(feat_scores_f, N)
            traj_assignment_f = self._greedy_assignment(traj_scores_f, N)

            occlusion_flags_f = [False] * N

            chosen_f, source_f, confidence_f, abstain_f = self._resolve_conflict(
                feat_assignment_f, traj_assignment_f,
                feat_scores_f, traj_scores_f,
                feat_missing_f, occlusion_flags_f,
                traj_unc_f, N)

            results[b] = chosen_f

            ep_conf = {
                'feat_confs': [files[i]['feature_confidence'] for i in range(N)],
                'traj_confs': [files[i]['trajectory_confidence'] for i in range(N)],
                'total_confs': [files[i]['total_confidence'] for i in range(N)],
                'final_confidence': confidence_f,
            }
            all_confidences.append(ep_conf)
            all_sources.append(source_f)
            all_abstain_flags.append(abstain_f)

            if return_trace:
                all_traces.append(trace)

        if single:
            results = results[0]

        if return_conflict_info:
            return results, all_confidences, all_sources, all_abstain_flags
        if return_trace:
            return results, all_traces
        return results


class TrajectoryRobustObjectFile:
    def __init__(self, traj_model=None, num_objects=2, feature_dim=2,
                 approach_threshold=5.0, approach_weight=0.5,
                 temporal_decay=0.9, n_voting_steps=5,
                 conflict_strategy="approach_aware",
                 occlusion_decay=0.95, reappearance_boost=0.5):
        self.traj_model = traj_model
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.approach_threshold = approach_threshold
        self.approach_weight = approach_weight
        self.temporal_decay = temporal_decay
        self.n_voting_steps = n_voting_steps
        self.conflict_strategy = conflict_strategy
        self.occlusion_decay = occlusion_decay
        self.reappearance_boost = reappearance_boost

    def _get_predicted_positions(self, observed_positions):
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
                pred = pred.cpu().numpy()
        return pred

    def _compute_approach_signal(self, observed_positions):
        B = observed_positions.shape[0]
        N = self.num_objects
        approach_signals = np.zeros(B)

        for b in range(B):
            dists = []
            for t in range(observed_positions.shape[1]):
                d = np.linalg.norm(observed_positions[b, t, 0, :] -
                                   observed_positions[b, t, 1, :])
                dists.append(d)

            if len(dists) >= 2:
                early_dist = np.mean(dists[:len(dists)//2])
                late_dist = np.mean(dists[len(dists)//2:])
                approach_signals[b] = late_dist - early_dist
            else:
                approach_signals[b] = 0.0

        return approach_signals

    def _compute_feature_similarity(self, feat_vec, identity_key):
        nf = np.linalg.norm(feat_vec)
        nk = np.linalg.norm(identity_key)
        if nf < 1e-8 or nk < 1e-8:
            return 0.0
        sim = np.dot(feat_vec, identity_key) / (nf * nk)
        return float(sim)

    def _compute_trajectory_distance(self, fut_pos, file_state):
        if 'predicted_positions' in file_state:
            pred_pos = file_state['predicted_positions'][-1]
        else:
            pred_pos = file_state['last_pos'] + file_state.get('last_vel', np.zeros(2))
        dist = np.linalg.norm(fut_pos - pred_pos)
        return float(dist)

    def _greedy_assignment(self, scores, N):
        assignment = np.full(N, -1, dtype=int)
        used_cols = set()
        for i in range(N):
            valid = scores[i].copy()
            for j in used_cols:
                valid[j] = -float('inf')
            if valid.max() > -float('inf'):
                best = np.argmax(valid)
                assignment[i] = best
                used_cols.add(best)
        return assignment

    def _compute_margin(self, scores, assignment, N):
        margins = []
        for i in range(N):
            if assignment[i] < 0:
                continue
            best_score = scores[i, assignment[i]]
            others = [scores[i, j] for j in range(N) if j != assignment[i] and scores[i, j] > -float('inf')]
            if others:
                margin = best_score - max(others)
            else:
                margin = best_score
            margins.append(margin)
        return float(np.mean(margins)) if margins else 0.0

    def _resolve_with_approach(self, feat_assignment, traj_assignment,
                                feat_scores, traj_scores,
                                approach_signal, feat_missing, occluded, N):
        conflict = not np.array_equal(feat_assignment, traj_assignment)

        if not conflict:
            return feat_assignment.copy(), "agreement", 0.9, False

        feat_margin = self._compute_margin(feat_scores, feat_assignment, N)
        traj_margin = self._compute_margin(traj_scores, traj_assignment, N)

        objects_approaching = approach_signal < -self.approach_threshold

        if self.conflict_strategy == "approach_aware":
            if objects_approaching:
                chosen = feat_assignment.copy()
                source = "feature_approach"
                confidence = max(0.3, min(0.85, feat_margin * 0.8))
                abstain = False
            else:
                if traj_margin > feat_margin * 1.2:
                    chosen = traj_assignment.copy()
                    source = "trajectory"
                    confidence = max(0.2, min(0.7, traj_margin / 10.0))
                    abstain = False
                elif feat_margin > traj_margin * 1.2:
                    chosen = feat_assignment.copy()
                    source = "feature_margin"
                    confidence = max(0.2, min(0.7, feat_margin * 0.7))
                    abstain = False
                else:
                    chosen = traj_assignment.copy()
                    source = "uncertain"
                    confidence = 0.15
                    abstain = True

        elif self.conflict_strategy == "approach_veto":
            if objects_approaching:
                chosen = feat_assignment.copy()
                source = "feature_approach_veto"
                confidence = max(0.4, min(0.85, feat_margin * 0.8))
                abstain = False
            else:
                chosen = traj_assignment.copy()
                source = "trajectory_no_approach"
                confidence = max(0.3, min(0.8, traj_margin / 10.0))
                abstain = False
        else:
            chosen = traj_assignment.copy()
            source = "trajectory"
            confidence = 0.5
            abstain = False

        if any(feat_missing) and source.startswith("feature"):
            chosen = traj_assignment.copy()
            source = "trajectory_fallback"
            confidence = max(0.2, confidence * 0.5)
            abstain = False

        if any(occluded) and source.startswith("feature"):
            chosen = traj_assignment.copy()
            source = "trajectory_occlusion"
            confidence = max(0.2, confidence * 0.5)
            abstain = False

        return chosen, source, confidence, abstain

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

        approach_signals = self._compute_approach_signal(observed_positions)
        pred_traj_all = self._get_predicted_positions(observed_positions)

        for b in range(B):
            files = []
            for i in range(N):
                f = {
                    'identity_key': observed_features[b, 0, i, :].copy(),
                    'last_pos': observed_positions[b, -1, i, :].copy(),
                    'last_vel': (observed_positions[b, -1, i, :] -
                                 observed_positions[b, -2, i, :]).copy(),
                    'feature_confidence': 1.0,
                    'trajectory_confidence': 1.0,
                    'total_confidence': 1.0,
                    'occluded_steps': 0,
                }
                if pred_traj_all is not None:
                    f['predicted_positions'] = pred_traj_all[b, :, i, :]
                files.append(f)

            T_pred = future_positions.shape[1]
            vote_steps = np.linspace(0, T_pred - 1, min(self.n_voting_steps, T_pred), dtype=int)

            feat_votes = np.zeros((N, N))
            traj_votes = np.zeros((N, N))

            for vi, t in enumerate(vote_steps):
                weight = self.temporal_decay ** vi

                is_occluded = np.zeros(N, dtype=bool)
                if occlusion_mask is not None:
                    is_occluded = occlusion_mask[b, t, :]

                fut_pos = [future_positions[b, t, i, :] for i in range(N)]
                fut_feat = None
                if future_features is not None:
                    fut_feat = [future_features[b, t, i, :] if not is_occluded[i] else None
                                for i in range(N)]

                feat_scores = np.full((N, N), -float('inf'))
                traj_scores = np.full((N, N), -float('inf'))

                for i in range(N):
                    for j in range(N):
                        feat_vec = fut_feat[i] if fut_feat is not None and fut_feat[i] is not None else np.zeros(self.feature_dim)
                        sim = self._compute_feature_similarity(feat_vec, files[j]['identity_key'])
                        if fut_feat is None or fut_feat[i] is None:
                            sim = 0.0
                        feat_scores[i, j] = sim

                        dist = self._compute_trajectory_distance(fut_pos[i], files[j])
                        traj_scores[i, j] = -dist

                feat_assign = self._greedy_assignment(feat_scores, N)
                traj_assign = self._greedy_assignment(traj_scores, N)

                for i in range(N):
                    if feat_assign[i] >= 0:
                        feat_votes[i, feat_assign[i]] += weight * max(0.01, feat_scores[i, feat_assign[i]])
                    if traj_assign[i] >= 0:
                        traj_votes[i, traj_assign[i]] += weight * max(0.01, -traj_scores[i, traj_assign[i]] + 100)

            feat_assignment = self._greedy_assignment(feat_votes, N)
            traj_assignment = self._greedy_assignment(traj_votes, N)

            final_fut_pos = [future_positions[b, -1, i, :] for i in range(N)]
            final_fut_feat = [future_features[b, -1, i, :] for i in range(N)] if future_features is not None else None

            feat_scores_f = np.full((N, N), -float('inf'))
            traj_scores_f = np.full((N, N), -float('inf'))
            feat_missing_f = [False] * N
            occluded_f = [False] * N

            for i in range(N):
                for j in range(N):
                    feat_vec = final_fut_feat[i] if final_fut_feat is not None and final_fut_feat[i] is not None else np.zeros(self.feature_dim)
                    sim = self._compute_feature_similarity(feat_vec, files[j]['identity_key'])
                    if final_fut_feat is None or final_fut_feat[i] is None:
                        feat_missing_f[i] = True
                        sim = 0.0
                    feat_scores_f[i, j] = sim

                    dist = self._compute_trajectory_distance(final_fut_pos[i], files[j])
                    traj_scores_f[i, j] = -dist

            feat_assign_f = self._greedy_assignment(feat_scores_f, N)
            traj_assign_f = self._greedy_assignment(traj_scores_f, N)

            chosen, source, confidence, abstain = self._resolve_with_approach(
                feat_assign_f, traj_assign_f,
                feat_scores_f, traj_scores_f,
                approach_signals[b], feat_missing_f, occluded_f, N)

            results[b] = chosen

            ep_conf = {
                'feat_confs': [files[i]['feature_confidence'] for i in range(N)],
                'traj_confs': [files[i]['trajectory_confidence'] for i in range(N)],
                'total_confs': [files[i]['total_confidence'] for i in range(N)],
                'final_confidence': confidence,
                'approach_signal': float(approach_signals[b]),
            }
            all_confidences.append(ep_conf)
            all_sources.append(source)
            all_abstain_flags.append(abstain)

        if single:
            results = results[0]

        if return_conflict_info:
            return results, all_confidences, all_sources, all_abstain_flags
        if return_trace:
            return results, []
        return results


class LearnedTrajObjectFile:
    def __init__(self, traj_model, num_objects=2, feature_dim=2,
                 feature_weight=1.0, traj_weight=1.0):
        self.traj_model = traj_model
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.feature_weight = feature_weight
        self.traj_weight = traj_weight

    def _get_predicted_positions(self, observed_positions):
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
                pred = pred.cpu().numpy()
        return pred

    def predict_identity(self, observed_positions, observed_features,
                         future_positions, future_features,
                         occlusion_mask=None, return_trace=False):
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

        pred_traj = self._get_predicted_positions(observed_positions)

        for b in range(B):
            files = []
            for i in range(N):
                files.append({
                    'identity_key': observed_features[b, 0, i, :].copy(),
                    'predicted_positions': pred_traj[b, :, i, :],
                })

            final_fut_pos = [future_positions[b, -1, i, :] for i in range(N)]
            final_fut_feat = [future_features[b, -1, i, :] for i in range(N)] if future_features is not None else None

            scores = np.full((N, N), -float('inf'))
            for i in range(N):
                for j in range(N):
                    fs = 0.0
                    if final_fut_feat is not None and final_fut_feat[i] is not None:
                        fkey = files[j]['identity_key']
                        nf = np.linalg.norm(final_fut_feat[i])
                        nk = np.linalg.norm(fkey)
                        if nf > 1e-8 and nk > 1e-8:
                            fs = np.dot(final_fut_feat[i], fkey) / (nf * nk)

                    pred_pos = files[j]['predicted_positions'][-1]
                    dist = np.linalg.norm(final_fut_pos[i] - pred_pos)
                    ts = -dist

                    scores[i, j] = self.feature_weight * fs + self.traj_weight * ts

            used = set()
            for i in range(N):
                valid = scores[i].copy()
                for j in used:
                    valid[j] = -float('inf')
                best = np.argmax(valid)
                results[b, i] = best
                used.add(best)

        if single:
            results = results[0]
        return results
