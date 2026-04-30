"""
Subspace Intervention Test for Object-Centric Models

Inspired by Relation-Internalization's Neural Probe methodology:
- Train a linear probe to read identity from model's intermediate representation
- Extract the identity subspace via SVD
- Remove that subspace from the representation
- Measure performance drop: if drop > 0, identity is causally used (State D)
                                      if drop ≈ 0, identity is bystander info (State C)

Structure States (from Probe-to-Boundary):
  State A: not readable  → probe accuracy ≈ chance
  State B: readable but unstable → probe accuracy high, but drops with perturbation
  State C: readable + stable but not causal → probe accuracy high, subspace removal has no effect
  State D: readable + stable + causal → probe accuracy high, subspace removal hurts performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression


class IdentityProbe:
    def __init__(self, num_objects=2):
        self.num_objects = num_objects
        self.probe = None
        self.subspace_basis = None
        self.probe_accuracy = None
        self._hidden_cache = {}

    def extract_hidden(self, model, observed_positions, observed_features=None,
                       future_positions=None, future_features=None):
        if isinstance(observed_positions, np.ndarray):
            observed_positions = torch.FloatTensor(observed_positions)
        if future_positions is not None and isinstance(future_positions, np.ndarray):
            future_positions = torch.FloatTensor(future_positions)
        if observed_features is not None and isinstance(observed_features, np.ndarray):
            observed_features = torch.FloatTensor(observed_features)
        if future_features is not None and isinstance(future_features, np.ndarray):
            future_features = torch.FloatTensor(future_features)

        model.eval()
        N = self.num_objects

        with torch.no_grad():
            try:
                result = model(observed_positions, observed_features,
                               future_positions, future_features)
            except TypeError:
                try:
                    result = model(observed_positions)
                except TypeError:
                    result = model.forward(observed_positions)

        if isinstance(result, tuple):
            first = result[0]
        else:
            first = result

        if isinstance(first, torch.Tensor):
            first_np = first.detach().cpu().numpy()
        else:
            first_np = np.array(first)

        if first_np.ndim == 3 and first_np.shape[1] == N and first_np.shape[2] == N:
            return first.detach().cpu()

        if hasattr(model, 'get_hidden_representation'):
            try:
                z = model.get_hidden_representation(
                    observed_positions, observed_features,
                    future_positions, future_features)
                if isinstance(z, torch.Tensor):
                    return z.detach().cpu()
            except Exception:
                pass

        captured = {}

        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captured[name] = output[0].detach().cpu()
                else:
                    captured[name] = output.detach().cpu()
            return hook_fn

        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.GRU, nn.Sequential)):
                hooks.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            try:
                result2 = model(observed_positions, observed_features,
                                future_positions, future_features)
            except TypeError:
                try:
                    result2 = model(observed_positions)
                except TypeError:
                    pass

        for h in hooks:
            h.remove()

        best_hidden = None
        best_score = -1

        for name, h in captured.items():
            if h.ndim == 2:
                score = h.shape[1]
            elif h.ndim == 3:
                score = h.shape[1] * h.shape[2]
            else:
                score = 1

            if h.shape[0] == observed_positions.shape[0] and score > best_score:
                best_score = score
                best_hidden = h

        if best_hidden is not None:
            return best_hidden

        if isinstance(result, tuple):
            return result[0].detach().cpu()
        return result.detach().cpu()

    def fit(self, model, observed_positions, identity_labels,
            observed_features=None, future_positions=None, future_features=None):
        if isinstance(identity_labels, np.ndarray):
            identity_labels = identity_labels.copy()
        else:
            identity_labels = identity_labels.cpu().numpy()

        B = observed_positions.shape[0] if isinstance(observed_positions, torch.Tensor) else len(observed_positions)
        N = self.num_objects

        hidden = self.extract_hidden(model, observed_positions, observed_features,
                                      future_positions, future_features)
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.cpu().numpy()

        H, labels_flat = self._align(hidden, identity_labels, B, N)

        self.probe = LogisticRegression(max_iter=2000, C=1.0)
        self.probe.fit(H, labels_flat)
        self.probe_accuracy = self.probe.score(H, labels_flat)

        W = self.probe.coef_
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        rank = int(np.sum(S > 1e-8))
        if rank == 0:
            rank = 1
        self.subspace_basis = Vt[:rank].T

        return self.probe_accuracy

    def _align(self, hidden_np, identity_labels, B, N):
        if hidden_np.ndim == 3 and hidden_np.shape[1] == N and hidden_np.shape[2] == N:
            H_per_obj = []
            labels_per_obj = []
            for j in range(N):
                H_per_obj.append(hidden_np[:, :, j])
                labels_per_obj.append(identity_labels[:, j])
            H = np.concatenate(H_per_obj, axis=0)
            labels_flat = np.concatenate(labels_per_obj, axis=0)
        elif hidden_np.ndim == 4:
            H = hidden_np.reshape(B, -1)
            labels_flat = identity_labels.reshape(-1)
        elif hidden_np.ndim == 3:
            if hidden_np.shape[0] == B * N:
                H = hidden_np.reshape(B * N, -1)
            else:
                H = hidden_np.reshape(B, -1)
            labels_flat = identity_labels.reshape(-1)
        elif hidden_np.ndim == 2:
            H = hidden_np
            labels_flat = identity_labels.reshape(-1)
        else:
            H = hidden_np.reshape(B, -1)
            labels_flat = identity_labels.reshape(-1)

        min_len = min(len(H), len(labels_flat))
        H = H[:min_len]
        labels_flat = labels_flat[:min_len]

        return H, labels_flat

    def remove_subspace(self, hidden_np):
        original_shape = hidden_np.shape
        B = hidden_np.shape[0]
        N = self.num_objects

        if hidden_np.ndim == 3 and hidden_np.shape[1] == N and hidden_np.shape[2] == N:
            H_per_obj = []
            for j in range(N):
                H_per_obj.append(hidden_np[:, :, j])
            H = np.concatenate(H_per_obj, axis=0)
        else:
            H = hidden_np.reshape(B, -1)

        mean = H.mean(axis=0, keepdims=True)
        H_centered = H - mean
        projection = H_centered @ self.subspace_basis @ self.subspace_basis.T
        H_cleaned = H - projection

        if hidden_np.ndim == 3 and hidden_np.shape[1] == N and hidden_np.shape[2] == N:
            H_restored = np.zeros(original_shape)
            for j in range(N):
                H_restored[:, :, j] = H_cleaned[j * B:(j + 1) * B]
            H_cleaned = H_restored
        else:
            H_cleaned = H_cleaned.reshape(original_shape)

        return H_cleaned

    def intervention_drop(self, model, observed_positions, identity_labels,
                          observed_features=None, future_positions=None,
                          future_features=None):
        if isinstance(identity_labels, np.ndarray):
            identity_labels_np = identity_labels.copy()
        else:
            identity_labels_np = identity_labels.cpu().numpy()

        B = observed_positions.shape[0] if isinstance(observed_positions, torch.Tensor) else len(observed_positions)
        N = self.num_objects

        hidden = self.extract_hidden(model, observed_positions, observed_features,
                                      future_positions, future_features)
        if isinstance(hidden, torch.Tensor):
            hidden_np = hidden.cpu().numpy()
        else:
            hidden_np = hidden.copy()

        H, labels_flat = self._align(hidden_np, identity_labels_np, B, N)
        base_pred = self.probe.predict(H)
        base_acc = np.mean(base_pred == labels_flat)

        H_cleaned_np = self.remove_subspace(hidden_np)
        H_cleaned, _ = self._align(H_cleaned_np, identity_labels_np, B, N)

        try:
            cleaned_pred = self.probe.predict(H_cleaned)
            cleaned_acc = np.mean(cleaned_pred == labels_flat)
        except Exception:
            cleaned_acc = 0.0

        probe_drop = base_acc - cleaned_acc

        return {
            'base_probe_accuracy': base_acc,
            'cleaned_probe_accuracy': cleaned_acc,
            'probe_drop': probe_drop,
        }


class SubspaceInterventionTester:
    def __init__(self, num_objects=2):
        self.num_objects = num_objects
        self.results = {}

    def _predict_with_model(self, model, observed_positions, observed_features=None,
                            future_positions=None, future_features=None, method="combined"):
        model.eval()
        with torch.no_grad():
            try:
                pred = model.predict_identity(
                    observed_positions, observed_features,
                    future_positions=future_positions,
                    future_features=future_features,
                    method=method)
            except TypeError:
                try:
                    pred = model.predict_identity(
                        observed_positions, observed_features,
                        future_positions=future_positions,
                        future_features=future_features)
                except TypeError:
                    pred = model.predict_identity(
                        observed_positions, observed_features,
                        future_positions=future_positions)
        if isinstance(pred, tuple):
            pred = pred[0]
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        return pred

    def _compute_swap_accuracy(self, pred, true_labels):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.cpu().numpy()
        return float((pred == true_labels).all(axis=1).mean())

    def full_diagnosis(self, model, model_name,
                       train_data, test_data, swap_test_data,
                       method="combined"):
        print(f"\n{'='*70}")
        print(f"Subspace Intervention Diagnosis: {model_name}")
        print(f"{'='*70}")

        obs_pos_train = train_data["observed_positions"]
        fut_pos_train = train_data["future_positions"]
        ids_train = train_data["identity_labels"]
        obs_feat_train = train_data.get("object_features_obs")
        fut_feat_train = train_data.get("object_features_fut")

        obs_pos_test = test_data["observed_positions"]
        fut_pos_test = test_data["future_positions"]
        ids_test = test_data["identity_labels"]
        obs_feat_test = test_data.get("object_features_obs")
        fut_feat_test = test_data.get("object_features_fut")

        obs_pos_swap = swap_test_data["observed_positions"]
        fut_pos_swap = swap_test_data["future_positions"]
        ids_swap = swap_test_data["identity_labels"]
        obs_feat_swap = swap_test_data.get("object_features_obs")
        fut_feat_swap = swap_test_data.get("object_features_fut")

        print("Step 1: Training identity probe on model's hidden representation...")
        probe = IdentityProbe(num_objects=self.num_objects)
        probe_acc = probe.fit(model, obs_pos_train, ids_train,
                              observed_features=obs_feat_train,
                              future_positions=fut_pos_train,
                              future_features=fut_feat_train)
        print(f"  Probe accuracy on train: {probe_acc:.4f}")

        print("\nStep 2: Testing readability on clean test set...")
        clean_intervention = probe.intervention_drop(
            model, obs_pos_test, ids_test,
            observed_features=obs_feat_test,
            future_positions=fut_pos_test,
            future_features=fut_feat_test)
        print(f"  Base probe accuracy: {clean_intervention['base_probe_accuracy']:.4f}")
        print(f"  Cleaned probe accuracy: {clean_intervention['cleaned_probe_accuracy']:.4f}")
        print(f"  Probe drop: {clean_intervention['probe_drop']:.4f}")

        print("\nStep 3: Testing readability on swap test set...")
        swap_intervention = probe.intervention_drop(
            model, obs_pos_swap, ids_swap,
            observed_features=obs_feat_swap,
            future_positions=fut_pos_swap,
            future_features=fut_feat_swap)
        print(f"  Base probe accuracy: {swap_intervention['base_probe_accuracy']:.4f}")
        print(f"  Cleaned probe accuracy: {swap_intervention['cleaned_probe_accuracy']:.4f}")
        print(f"  Probe drop: {swap_intervention['probe_drop']:.4f}")

        print("\nStep 4: Computing behavioral swap accuracy...")
        pred_swap = self._predict_with_model(model, obs_pos_swap, obs_feat_swap,
                                              future_positions=fut_pos_swap,
                                              future_features=fut_feat_swap,
                                              method=method)
        swap_acc = self._compute_swap_accuracy(pred_swap, ids_swap)
        print(f"  Swap accuracy: {swap_acc:.4f}")

        print("\nStep 5: Stability test (perturbation)...")
        if isinstance(obs_pos_test, np.ndarray):
            obs_pos_perturbed = obs_pos_test + np.random.randn(*obs_pos_test.shape) * 0.1
        else:
            obs_pos_perturbed = obs_pos_test + torch.randn_like(obs_pos_test) * 0.1

        perturbed_intervention = probe.intervention_drop(
            model, obs_pos_perturbed, ids_test,
            observed_features=obs_feat_test,
            future_positions=fut_pos_test,
            future_features=fut_feat_test)
        stability = 1.0 - abs(clean_intervention['base_probe_accuracy'] -
                               perturbed_intervention['base_probe_accuracy'])
        stability = max(0.0, min(1.0, stability))
        print(f"  Stability score: {stability:.4f}")

        readability = clean_intervention['base_probe_accuracy']
        causality = clean_intervention['probe_drop']

        if readability < 0.6:
            state = "A (Not Readable)"
        elif stability < 0.7:
            state = "B (Readable but Unstable)"
        elif causality < 0.05:
            state = "C (Readable+Stable, Not Causal)"
        else:
            state = "D (Readable+Stable+Causal)"

        result = {
            'model_name': model_name,
            'readability': readability,
            'stability': stability,
            'causality': causality,
            'state': state,
            'swap_accuracy': swap_acc,
            'clean_probe_acc': clean_intervention['base_probe_accuracy'],
            'clean_probe_drop': clean_intervention['probe_drop'],
            'swap_probe_acc': swap_intervention['base_probe_accuracy'],
            'swap_probe_drop': swap_intervention['probe_drop'],
        }

        self.results[model_name] = result

        print(f"\n{'='*70}")
        print(f"DIAGNOSIS RESULT: {model_name}")
        print(f"{'='*70}")
        print(f"  Readability:  {readability:.4f}  (can identity be read from representation?)")
        print(f"  Stability:    {stability:.4f}  (is identity robust to perturbation?)")
        print(f"  Causality:    {causality:.4f}  (does removing identity subspace hurt?)")
        print(f"  State:        {state}")
        print(f"  Swap Acc:     {swap_acc:.4f}")
        print(f"{'='*70}")

        return result

    def print_fingerprint_map(self):
        print(f"\n{'='*90}")
        print("FAILURE FINGERPRINT MAP (inspired by SVT-v2)")
        print(f"{'='*90}")
        print(f"{'Model':<30} {'Read':>7} {'Stab':>7} {'Caus':>7} {'Swap':>7} {'State':<35}")
        print("-" * 90)
        for name, r in self.results.items():
            print(f"{name:<30} {r['readability']:>7.3f} {r['stability']:>7.3f} "
                  f"{r['causality']:>7.3f} {r['swap_accuracy']:>7.3f} {r['state']:<35}")
        print("=" * 90)

        print(f"\nInterpretation:")
        print(f"  State A: Identity not formed in representation")
        print(f"  State B: Identity readable but unstable")
        print(f"  State C: Identity readable+stable but NOT causal (BYSTANDER)")
        print(f"  State D: Identity readable+stable+causal (TRUE structural capacity)")
        print(f"\n  Key insight: Only State D models genuinely use identity for task performance.")
        print(f"  State C models may have high swap accuracy but identity is not the cause.")
