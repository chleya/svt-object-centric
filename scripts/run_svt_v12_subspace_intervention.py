"""
SVT-v12: Subspace Intervention Diagnosis

Directly adapted from Relation-Internalization's Neural Probe methodology.

Core question: Is identity information CAUSALLY USED by the model,
or is it merely BYSTANDER information that's readable but not used?

Method:
  1. Train each model normally
  2. Train a linear probe to read identity from model's hidden representation
  3. Extract identity subspace via SVD of probe weights
  4. Remove that subspace → measure performance drop
  5. Classify each model into Structure States A/B/C/D

Expected findings:
  - TrajectoryOnly: State A or C (identity not in representation, or bystander)
  - FeatureOnly: State C (identity readable from features but not causally used for binding)
  - LearnedObjectFile: State C (degenerates to feature-only, identity is bystander)
  - ConflictAugmented: State B or C (forced to use trajectory, but may not internalize)
  - ProbabilisticStructure: hopefully State D (identity is causally used)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.object_file_models import TrajectoryOnlyAssignment, ConflictFirstObjectFile
from models.learned_object_file import LearnedObjectFile
from models.conflict_augmented_object_file import ConflictAugmentedLearnedObjectFile
from models.probabilistic_structure_object_file import ProbabilisticStructureObjectFile
from diagnostics.subspace_intervention import SubspaceInterventionTester, IdentityProbe
from utils.torch_training import train_model


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


class FeatureOnlyAssignment(nn.Module):
    def __init__(self, num_objects=2, feature_dim=2, hidden_dim=128, slot_dim=64):
        super().__init__()
        self.num_objects = num_objects
        self.obs_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        self.fut_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        self.logit_net = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observed_positions, observed_features=None,
                future_positions=None, future_features=None):
        if isinstance(observed_features, np.ndarray):
            observed_features = torch.FloatTensor(observed_features)
        if isinstance(future_features, np.ndarray):
            future_features = torch.FloatTensor(future_features)

        B = observed_features.shape[0]
        N = self.num_objects

        obs_pooled = observed_features[:, 0, :, :] if observed_features.dim() == 4 else observed_features
        fut_pooled = future_features[:, 0, :, :] if future_features.dim() == 4 else future_features

        z_obs = self.obs_encoder(obs_pooled)
        z_fut = self.fut_encoder(fut_pooled)

        logits = torch.zeros(B, N, N, device=observed_features.device)
        for i in range(N):
            for j in range(N):
                pair = torch.cat([z_fut[:, i, :], z_obs[:, j, :]], dim=-1)
                logits[:, i, j] = self.logit_net(pair).squeeze(-1)

        return logits

    def compute_loss(self, observed_positions, future_positions, identity_labels,
                     observed_features=None, future_features=None, is_swap=None):
        if isinstance(observed_features, np.ndarray):
            observed_features = torch.FloatTensor(observed_features)
        if isinstance(future_features, np.ndarray):
            future_features = torch.FloatTensor(future_features)
        if isinstance(identity_labels, np.ndarray):
            identity_labels = torch.LongTensor(identity_labels)
        elif isinstance(identity_labels, torch.Tensor) and identity_labels.dtype != torch.long:
            identity_labels = identity_labels.long()

        logits = self.forward(observed_positions, observed_features,
                              future_positions, future_features)
        N = self.num_objects
        loss = F.cross_entropy(logits.reshape(-1, N), identity_labels.reshape(-1))
        return loss, loss, torch.tensor(0.0), torch.tensor(0.0)

    def predict_identity(self, observed_positions, observed_features=None,
                         test_future=None, future_features=None,
                         future_positions=None, method="combined"):
        self.eval()
        with torch.no_grad():
            if isinstance(observed_features, np.ndarray):
                observed_features = torch.FloatTensor(observed_features)
            if isinstance(future_features, np.ndarray):
                future_features = torch.FloatTensor(future_features)

            logits = self.forward(observed_positions, observed_features,
                                  future_positions, future_features)
            pred = logits.argmax(dim=-1)
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        return pred

    def predict_future(self, observed_positions, observed_features=None):
        return None


def train_neural_model(model, train_data, epochs=30, batch_size=64, lr=1e-3,
                       uses_features=True, uses_future_features=True):
    obs_pos = torch.FloatTensor(train_data["observed_positions"])
    fut_pos = torch.FloatTensor(train_data["future_positions"])
    ids = torch.LongTensor(train_data["identity_labels"])

    obs_feat = None
    fut_feat = None
    if uses_features and "object_features_obs" in train_data:
        obs_feat = torch.FloatTensor(train_data["object_features_obs"])
    if uses_future_features and "object_features_fut" in train_data:
        fut_feat = torch.FloatTensor(train_data["object_features_fut"])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches = len(obs_pos) // batch_size

    for epoch in range(epochs):
        indices = np.random.permutation(len(obs_pos))
        total_loss = 0
        for bi in range(n_batches):
            idx = indices[bi * batch_size:(bi + 1) * batch_size]
            loss, _, _, _ = model.compute_loss(
                obs_pos[idx], fut_pos[idx], ids[idx],
                observed_features=obs_feat[idx] if obs_feat is not None else None,
                future_features=fut_feat[idx] if fut_feat is not None else None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")


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

    print("\n" + "="*70)
    print("MODEL 1: TrajectoryOnlyAssignment")
    print("="*70)
    traj_model = TrajectoryOnlyAssignment(num_objects=2, hidden_dim=256, num_layers=3)
    train_model(traj_model, train_data, val_data=clean_test, epochs=30, batch_size=64,
                lr=1e-3, device="cpu", uses_features=False, uses_future_features=False, verbose=False)
    tester.full_diagnosis(
        traj_model, "TrajectoryOnly",
        train_data, clean_test, swap_test,
        method="combined")

    print("\n" + "="*70)
    print("MODEL 2: FeatureOnlyAssignment")
    print("="*70)
    feat_model = FeatureOnlyAssignment(num_objects=2, feature_dim=2, hidden_dim=128, slot_dim=64)
    train_neural_model(feat_model, train_data, epochs=30, batch_size=64, lr=1e-3,
                       uses_features=True, uses_future_features=True)
    tester.full_diagnosis(
        feat_model, "FeatureOnly",
        train_data, clean_test, swap_test,
        method="combined")

    print("\n" + "="*70)
    print("MODEL 3: LearnedObjectFile")
    print("="*70)
    lof_model = LearnedObjectFile(num_objects=2, feature_dim=2, hidden_dim=128, slot_dim=64,
                                   t_obs=10, t_pred=20, identity_weight=1.0,
                                   conflict_weight=0.5, channel_aux_weight=0.3)
    train_neural_model(lof_model, train_data, epochs=30, batch_size=64, lr=1e-3,
                       uses_features=True, uses_future_features=True)
    tester.full_diagnosis(
        lof_model, "LearnedObjectFile",
        train_data, clean_test, swap_test,
        method="combined")

    print("\n" + "="*70)
    print("MODEL 4: ConflictAugmented (p_conflict=0.3)")
    print("="*70)
    ca_model = ConflictAugmentedLearnedObjectFile(
        num_objects=2, feature_dim=2, hidden_dim=128, slot_dim=64,
        t_obs=10, t_pred=20, identity_weight=1.0, conflict_weight=0.5,
        channel_aux_weight=0.3, p_conflict=0.3, p_feature_drop=0.1)
    train_neural_model(ca_model, train_data, epochs=30, batch_size=64, lr=1e-3,
                       uses_features=True, uses_future_features=True)
    tester.full_diagnosis(
        ca_model, "ConflictAugmented_p03",
        train_data, clean_test, swap_test,
        method="combined")

    print("\n" + "="*70)
    print("MODEL 5: ProbabilisticStructure (p_conflict=0.3)")
    print("="*70)
    ps_model = ProbabilisticStructureObjectFile(
        num_objects=2, feature_dim=2, hidden_dim=128, slot_dim=64,
        t_obs=10, t_pred=20, identity_weight=1.0, structure_weight=0.5,
        p_conflict=0.3, p_feature_drop=0.05)
    train_neural_model(ps_model, train_data, epochs=30, batch_size=64, lr=1e-3,
                       uses_features=True, uses_future_features=True)
    tester.full_diagnosis(
        ps_model, "ProbStruct_p03",
        train_data, clean_test, swap_test,
        method="combined")

    print("\n" + "="*70)
    print("MODEL 6: ProbabilisticStructure (p_conflict=0.5)")
    print("="*70)
    ps_model_5 = ProbabilisticStructureObjectFile(
        num_objects=2, feature_dim=2, hidden_dim=128, slot_dim=64,
        t_obs=10, t_pred=20, identity_weight=1.0, structure_weight=0.5,
        p_conflict=0.5, p_feature_drop=0.05)
    train_neural_model(ps_model_5, train_data, epochs=30, batch_size=64, lr=1e-3,
                       uses_features=True, uses_future_features=True)
    tester.full_diagnosis(
        ps_model_5, "ProbStruct_p05",
        train_data, clean_test, swap_test,
        method="combined")

    print("\n" + "="*70)
    print("MODEL 7: ConflictFirstObjectFile (rule-based)")
    print("="*70)
    traj_for_cf = TrajectoryOnlyAssignment(num_objects=2, hidden_dim=256, num_layers=3)
    train_model(traj_for_cf, train_data, val_data=clean_test, epochs=30, batch_size=64,
                lr=1e-3, device="cpu", uses_features=False, uses_future_features=False, verbose=False)
    cf_model = ConflictFirstObjectFile(traj_model=traj_for_cf, strategy="margin_gated",
                                        num_objects=2, feature_dim=2)

    obs_pos_swap = swap_test["observed_positions"]
    obs_feat_swap = swap_test.get("object_features_obs")
    fut_feat_swap = swap_test.get("object_features_fut")
    fut_pos_swap = swap_test["future_positions"]
    ids_swap = swap_test["identity_labels"]

    cf_result = cf_model.predict_identity(obs_pos_swap, obs_feat_swap, fut_pos_swap, fut_feat_swap,
                                           return_conflict_info=True)
    cf_pred = cf_result[0]
    cf_swap_acc = float((cf_pred == ids_swap).all(axis=1).mean())

    print(f"  ConflictFirstObjectFile swap accuracy: {cf_swap_acc:.4f}")
    print(f"  (Rule-based model: no hidden representation to probe)")

    tester.print_fingerprint_map()

    print("\n" + "="*70)
    print("KEY ANALYSIS")
    print("="*70)
    for name, r in tester.results.items():
        if 'C' in r['state']:
            print(f"\n  {name}: STATE C (Bystander)")
            print(f"    → Identity is readable but NOT causally used")
            print(f"    → Model achieves swap_acc={r['swap_accuracy']:.3f} without relying on identity subspace")
            print(f"    → This means: high swap accuracy ≠ genuine identity understanding")
        elif 'D' in r['state']:
            print(f"\n  {name}: STATE D (Causal)")
            print(f"    → Identity IS causally used by the model")
            print(f"    → Removing identity subspace drops performance by {r['causality']:.3f}")
            print(f"    → This means: model genuinely depends on identity for task")
        elif 'B' in r['state']:
            print(f"\n  {name}: STATE B (Unstable)")
            print(f"    → Identity is readable but not stable under perturbation")
            print(f"    → Likely surface correlation, not genuine structure")
        elif 'A' in r['state']:
            print(f"\n  {name}: STATE A (Not Formed)")
            print(f"    → Identity is not encoded in representation")
            print(f"    → Model doesn't even represent identity information")

    print("\n" + "="*70)
    print("IMPLICATION FOR OBJECTFILE DESIGN")
    print("="*70)
    print("""
If most models are in State C:
  → The feature-trajectory trade-off is not about which channel to trust
  → It's about whether identity is CAUSALLY USED at all
  → Solution: not better gating, but forcing causal dependency on identity

If ProbStruct reaches State D:
  → Probabilistic structure selection genuinely uses identity
  → The approach is on the right track
  → Next step: scale up and test on published models

If no model reaches State D:
  → Current architectures fundamentally cannot create causal identity dependency
  → Need structural change (e.g., S4 differentiable graph from R4 spec)
  → Or counterfactual training from Neural Stage
""")


if __name__ == "__main__":
    main()
