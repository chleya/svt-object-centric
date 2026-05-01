"""
SVT-v19b: Proximity-Enhanced Scorer (Keep GRU, Add Interaction to Scoring)

v19 finding: Complex interaction-aware trajectory encoder causes feature-hijack.
  - Trajectory scorer accuracy = 1.1% (completely failed)
  - Feature scorer dominated (100% clean, 0% conflict)
  - Same pattern as v18c Transformer encoder

Root cause: More complex trajectory encoder -> more parameters -> feature
scorer benefits more from the increased capacity, overwhelming trajectory scorer.

v19b solution: Keep GRU encoder (proven to work) but add proximity information
at the SCORING level, not the encoding level:
  1. Proximity-Enhanced Pairwise Scorer: Takes inter-object distance as
     additional input, allowing the scorer to be more careful when objects
     are close
  2. Distance-weighted scoring: When objects are close, the scorer relies
     more on trajectory-specific features; when far, it can use simpler cues
  3. No change to trajectory encoder (GRU stays simple)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.dual_pathway_object_file import DualPathwayObjectFile
from diagnostics.subspace_intervention import SubspaceInterventionTester


class ProximityEnhancedDualPathway(DualPathwayObjectFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        slot_dim = kwargs.get('slot_dim', 64)
        hidden_dim = kwargs.get('hidden_dim', 128)

        self.prox_enhanced_traj_scorer = nn.Sequential(
            nn.Linear(slot_dim * 2 + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1))

        self.prox_gate = nn.Sequential(
            nn.Linear(1, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1), nn.Sigmoid())

    def _compute_min_distances(self, positions):
        if isinstance(positions, np.ndarray):
            positions = torch.FloatTensor(positions)
        B = positions.shape[0]
        N = positions.shape[2]
        min_dists = torch.full((B, N), 1e6, device=positions.device)
        for j in range(N):
            for k in range(N):
                if j == k:
                    continue
                dist = torch.sqrt(
                    ((positions[:, :, j, :] - positions[:, :, k, :]) ** 2).sum(dim=-1) + 1e-8)
                min_dists[:, j] = torch.min(min_dists[:, j], dist.min(dim=1)[0])
        return min_dists

    def _compute_dual_scores(self, z_obs_feat, z_obs_traj, z_fut_feat, z_fut_traj,
                             obs_min_dists=None, fut_min_dists=None):
        feat_scores = self.feature_scorer(z_fut_feat, z_obs_feat)

        B = z_fut_traj.shape[0]
        N = z_fut_traj.shape[1]

        if obs_min_dists is not None and fut_min_dists is not None:
            traj_scores = torch.zeros(B, N, N, device=z_fut_traj.device)
            for i in range(N):
                for j in range(N):
                    pair = torch.cat([z_fut_traj[:, i, :], z_obs_traj[:, j, :]], dim=-1)
                    avg_dist = (obs_min_dists[:, j] + fut_min_dists[:, i]) / 2.0
                    norm_dist = avg_dist.unsqueeze(-1) / 64.0
                    prox_pair = torch.cat([pair, norm_dist], dim=-1)
                    base_score = self.trajectory_scorer.net(
                        torch.cat([z_fut_traj[:, i, :], z_obs_traj[:, j, :]], dim=-1))

                    enhanced_score = self.prox_enhanced_traj_scorer(prox_pair)

                    gate = self.prox_gate(norm_dist)
                    traj_scores[:, i, j] = (gate * enhanced_score + (1 - gate) * base_score).squeeze(-1)
        else:
            traj_scores = self.trajectory_scorer(z_fut_traj, z_obs_traj)

        return feat_scores, traj_scores

    def compute_loss(self, observed_positions, future_positions, identity_labels,
                     observed_features=None, future_features=None, is_swap=None,
                     p_conflict=0.0):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        if isinstance(future_positions, np.ndarray):
            future_positions = torch.FloatTensor(future_positions)
        if isinstance(identity_labels, np.ndarray):
            identity_labels = torch.LongTensor(identity_labels)
        elif isinstance(identity_labels, torch.Tensor) and identity_labels.dtype != torch.long:
            identity_labels = identity_labels.long()
        if observed_features is not None and isinstance(observed_features, np.ndarray):
            observed_features = torch.FloatTensor(observed_features)
        if future_features is not None and isinstance(future_features, np.ndarray):
            future_features = torch.FloatTensor(future_features)

        B = observed_positions.shape[0]
        N = self.num_objects

        aug_fut_feat = future_features
        traj_identity = identity_labels
        feat_identity = identity_labels
        conflict_labels = torch.zeros(B, device=observed_positions.device)

        if p_conflict > 0 and future_features is not None and N >= 2:
            aug_fut_feat = future_features.clone()
            feat_identity = identity_labels.clone()
            traj_identity = identity_labels.clone()
            for b in range(B):
                if torch.rand(1).item() < p_conflict:
                    if aug_fut_feat.dim() == 4:
                        aug_fut_feat[b, :, 0, :], aug_fut_feat[b, :, 1, :] = \
                            future_features[b, :, 1, :].clone(), future_features[b, :, 0, :].clone()
                    elif aug_fut_feat.dim() == 3:
                        aug_fut_feat[b, 0, :], aug_fut_feat[b, 1, :] = \
                            future_features[b, 1, :].clone(), future_features[b, 0, :].clone()
                    feat_identity[b, 0], feat_identity[b, 1] = \
                        identity_labels[b, 1].clone(), identity_labels[b, 0].clone()
                    conflict_labels[b] = 1.0

        z_obs, z_obs_feat, z_obs_traj, z_fut_feat, z_fut_traj = \
            self._encode(observed_positions, observed_features,
                         future_positions, aug_fut_feat)

        obs_min_dists = self._compute_min_distances(observed_positions)
        fut_min_dists = self._compute_min_distances(future_positions)

        feat_scores, traj_scores = self._compute_dual_scores(
            z_obs_feat, z_obs_traj, z_fut_feat, z_fut_traj,
            obs_min_dists, fut_min_dists)

        feat_loss = F.cross_entropy(feat_scores.reshape(-1, N), feat_identity.reshape(-1))
        traj_loss = F.cross_entropy(traj_scores.reshape(-1, N), traj_identity.reshape(-1))

        combined, agree, feat_assign, traj_assign = self._adaptive_combine(
            feat_scores, traj_scores)

        combined_identity = torch.where(
            agree.unsqueeze(-1).bool().expand_as(identity_labels),
            feat_identity, traj_identity)

        combined_loss = F.cross_entropy(combined.reshape(-1, N), combined_identity.reshape(-1))

        identity_loss = 0.3 * feat_loss + 0.3 * traj_loss + 0.4 * combined_loss

        smh_logits = torch.zeros(B, N, N, device=z_obs.device)
        for j in range(N):
            smh_logits[:, j, :] = self.smh(z_obs[:, j, :])
        smh_loss = F.cross_entropy(smh_logits.reshape(-1, N), identity_labels.reshape(-1))

        traj_preds = []
        for j in range(N):
            traj_j = self.traj_decoder(z_obs[:, j, :])
            traj_preds.append(traj_j.reshape(B, self.t_pred, self.dim))
        pred_traj = torch.stack(traj_preds, dim=2)
        traj_loss_val = F.mse_loss(pred_traj, future_positions)

        total_loss = (self.identity_weight * identity_loss +
                      self.smh_weight * smh_loss +
                      self.traj_weight * traj_loss_val)

        return total_loss, identity_loss, smh_loss, torch.tensor(0.0)

    def predict_identity(self, observed_positions, observed_features=None,
                         test_future=None, future_features=None,
                         future_positions=None, method="combined"):
        self.eval()
        with torch.no_grad():
            if future_positions is None and test_future is not None:
                future_positions = test_future
            if future_positions is None:
                future_positions = torch.zeros(
                    observed_positions.shape[0] if isinstance(observed_positions, torch.Tensor) else len(observed_positions),
                    self.t_pred, self.num_objects, self.dim)

            z_obs, z_obs_feat, z_obs_traj, z_fut_feat, z_fut_traj = \
                self._encode(observed_positions, observed_features,
                             future_positions, future_features)

            obs_min_dists = self._compute_min_distances(observed_positions)
            fut_min_dists = self._compute_min_distances(future_positions)

            feat_scores, traj_scores = self._compute_dual_scores(
                z_obs_feat, z_obs_traj, z_fut_feat, z_fut_traj,
                obs_min_dists, fut_min_dists)

            if method == "feature_only":
                pred = feat_scores.argmax(dim=-1)
            elif method == "trajectory_only":
                pred = traj_scores.argmax(dim=-1)
            else:
                combined, agree, feat_assign, traj_assign = self._adaptive_combine(
                    feat_scores, traj_scores)
                pred = combined.argmax(dim=-1)

        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        return pred

    def get_dual_scores(self, observed_positions, observed_features=None,
                        future_positions=None, future_features=None):
        self.eval()
        with torch.no_grad():
            if isinstance(observed_positions, np.ndarray):
                observed_positions = torch.FloatTensor(observed_positions)
            if observed_features is not None and isinstance(observed_features, np.ndarray):
                observed_features = torch.FloatTensor(observed_features)
            if future_positions is not None and isinstance(future_positions, np.ndarray):
                future_positions = torch.FloatTensor(future_positions)
            if future_features is not None and isinstance(future_features, np.ndarray):
                future_features = torch.FloatTensor(future_features)

            z_obs, z_obs_feat, z_obs_traj, z_fut_feat, z_fut_traj = \
                self._encode(observed_positions, observed_features,
                             future_positions, future_features)

            obs_min_dists = self._compute_min_distances(observed_positions)
            fut_min_dists = self._compute_min_distances(future_positions)

            feat_scores, traj_scores = self._compute_dual_scores(
                z_obs_feat, z_obs_traj, z_fut_feat, z_fut_traj,
                obs_min_dists, fut_min_dists)

            combined, agree, feat_assign, traj_assign = self._adaptive_combine(
                feat_scores, traj_scores)

        return {
            "feat_scores": feat_scores.cpu().numpy(),
            "traj_scores": traj_scores.cpu().numpy(),
            "combined_scores": combined.cpu().numpy(),
            "agreement": agree.cpu().numpy(),
            "feat_assignment": feat_assign.cpu().numpy(),
            "traj_assignment": traj_assign.cpu().numpy(),
            "min_distances": obs_min_dists.cpu().numpy(),
        }


def gen_train(n=1000, nobj=2, fdim=2, seed=0):
    rng = np.random.RandomState(seed)
    eps = []
    for _ in range(n):
        ep = _generate_single_episode(t_obs=10, t_pred=20, num_objects=nobj, arena_size=64.0,
            feature_mode="feature_bearing", feature_dim=fdim, randomize_object_order=True,
            identity_test=True, swap_probability=0.5, force_type="attractor",
            field_strength=0.5, damping=0.95, noise_std=0.1, rng=rng)
        eps.append(ep)
    return _stack_episodes(eps, "feature_bearing")


def train_model(model, train_data, epochs=80, batch_size=64, lr=1e-3, p_conflict=0.4):
    obs_pos = torch.FloatTensor(train_data["observed_positions"])
    fut_pos = torch.FloatTensor(train_data["future_positions"])
    ids = torch.LongTensor(train_data["identity_labels"])
    obs_feat = torch.FloatTensor(train_data["object_features_obs"]) if "object_features_obs" in train_data else None
    fut_feat = torch.FloatTensor(train_data["object_features_fut"]) if "object_features_fut" in train_data else None

    feat_params = list(model.feature_scorer.parameters()) + list(model.obs_feat_encoder.parameters()) + list(model.fut_feat_encoder.parameters())
    traj_params = list(model.trajectory_scorer.parameters()) + list(model.obs_traj_encoder.parameters()) + list(model.fut_traj_encoder.parameters())
    if hasattr(model, 'prox_enhanced_traj_scorer'):
        traj_params += list(model.prox_enhanced_traj_scorer.parameters()) + list(model.prox_gate.parameters())
    other_params = [p for p in model.parameters() if p not in set(feat_params) and p not in set(traj_params)]

    optimizer = torch.optim.Adam([
        {"params": feat_params, "lr": lr * 0.5},
        {"params": traj_params, "lr": lr * 2.0},
        {"params": other_params, "lr": lr},
    ], weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n_batches = len(obs_pos) // batch_size

    for epoch in range(epochs):
        if epoch < 15:
            for p in feat_params:
                p.requires_grad = False
        else:
            for p in feat_params:
                p.requires_grad = True

        indices = np.random.permutation(len(obs_pos))
        total_loss = 0
        for bi in range(n_batches):
            idx = indices[bi * batch_size:(bi + 1) * batch_size]
            loss, _, _, _ = model.compute_loss(
                obs_pos[idx], fut_pos[idx], ids[idx],
                observed_features=obs_feat[idx] if obs_feat is not None else None,
                future_features=fut_feat[idx] if fut_feat is not None else None,
                p_conflict=p_conflict)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")


def swap_features(fut_feat):
    if fut_feat is None:
        return None
    swapped = fut_feat.copy()
    if swapped.ndim == 4:
        swapped[:, :, 0, :], swapped[:, :, 1, :] = fut_feat[:, :, 1, :].copy(), fut_feat[:, :, 0, :].copy()
    elif swapped.ndim == 3:
        swapped[:, 0, :], swapped[:, 1, :] = fut_feat[:, 1, :].copy(), fut_feat[:, 0, :].copy()
    return swapped


def main():
    seed = 42
    print("Generating datasets...")
    eval_ds = generate_v3_dataset(n_train=1000, n_test=200, num_objects=2, feature_dim=2,
        feature_mode="feature_bearing", force_train_type="attractor", force_test_type="vortex",
        randomize_object_order=True, disjoint_init_split=True, seed=seed)
    train_data = gen_train(n=1000, seed=seed)
    swap_test = eval_ds["identity_test_swap_only"]
    clean_test = eval_ds["clean_test_id"]

    tester = SubspaceInterventionTester(num_objects=2)

    configs = [
        {
            "name": "DualPath_v18_baseline",
            "model_cls": DualPathwayObjectFile,
            "model_kwargs": {"num_objects": 2, "feature_dim": 2, "hidden_dim": 128,
                             "slot_dim": 64, "t_obs": 10, "t_pred": 20, "conflict_switch_temp": 0.1},
        },
        {
            "name": "ProxEnhanced_v19b",
            "model_cls": ProximityEnhancedDualPathway,
            "model_kwargs": {"num_objects": 2, "feature_dim": 2, "hidden_dim": 128,
                             "slot_dim": 64, "t_obs": 10, "t_pred": 20, "conflict_switch_temp": 0.1},
        },
    ]

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {cfg['name']}")
        print(f"{'='*70}")

        model = cfg["model_cls"](**cfg["model_kwargs"])
        train_model(model, train_data, epochs=80, p_conflict=0.4)

        tester.full_diagnosis(model, cfg["name"], train_data, clean_test, swap_test)

        obs_pos_s = swap_test["observed_positions"]
        obs_feat_s = swap_test.get("object_features_obs")
        fut_pos_s = swap_test["future_positions"]
        fut_feat_s = swap_test.get("object_features_fut")
        true_id_s = swap_test["identity_labels"]

        print("\n  --- Pathway Analysis ---")
        for method_name, method in [("combined", "combined"), ("feature_only", "feature_only"), ("trajectory_only", "trajectory_only")]:
            pred = model.predict_identity(obs_pos_s, obs_feat_s, future_positions=fut_pos_s,
                                          future_features=fut_feat_s, method=method)
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            bd = compute_identity_breakdown(pred, true_id_s)
            print(f"  swap_only ({method_name}): {bd['identity_swap_only']:.4f}")

        fut_feat_conflict = swap_features(fut_feat_s)
        for method_name, method in [("combined", "combined"), ("feature_only", "feature_only"), ("trajectory_only", "trajectory_only")]:
            pred_c = model.predict_identity(obs_pos_s, obs_feat_s, future_positions=fut_pos_s,
                                            future_features=fut_feat_conflict, method=method)
            if isinstance(pred_c, torch.Tensor):
                pred_c = pred_c.cpu().numpy()
            bd_c = compute_identity_breakdown(pred_c, true_id_s)
            print(f"  conflict ({method_name}): {bd_c['identity_swap_only']:.4f}")

        print("\n  --- Dual Score Analysis ---")
        ds = model.get_dual_scores(
            clean_test["observed_positions"], clean_test.get("object_features_obs"),
            clean_test["future_positions"], clean_test.get("object_features_fut"))
        print(f"  Agreement rate (clean): {ds['agreement'].mean():.4f}")

        ds_c = model.get_dual_scores(
            obs_pos_s, obs_feat_s, fut_pos_s, fut_feat_conflict)
        print(f"  Agreement rate (conflict): {ds_c['agreement'].mean():.4f}")

        traj_pred = ds['traj_assignment']
        true_id_clean = clean_test["identity_labels"]
        traj_acc = (traj_pred == true_id_clean).all(axis=1).mean()
        print(f"  Trajectory scorer acc (clean): {traj_acc:.4f}")

        if 'min_distances' in ds:
            min_dists = ds['min_distances']
            traj_correct = (traj_pred == true_id_clean).all(axis=1)
            print("\n  --- Proximity-Stratified Trajectory Accuracy ---")
            dist_bins = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100)]
            for lo, hi in dist_bins:
                mask = (min_dists.min(axis=1) >= lo) & (min_dists.min(axis=1) < hi)
                if mask.sum() > 0:
                    acc = traj_correct[mask].mean()
                    print(f"    min_dist [{lo:3d}, {hi:3d}): n={mask.sum():4d}, traj_acc={acc:.4f}")

    tester.print_fingerprint_map()

    print("\n" + "="*70)
    print("v19b ANALYSIS: Does proximity-enhanced scoring improve trajectory accuracy?")
    print("="*70)
    for name, r in tester.results.items():
        print(f"  {name}: State={r['state']}, Read={r['readability']:.3f}, "
              f"Caus={r['causality']:.3f}, Swap={r['swap_accuracy']:.3f}")


if __name__ == "__main__":
    main()
