"""
Base adapter interface for external object-centric models.

All external models must implement this interface to run through
the SVT stress test pipeline.
"""

import numpy as np
import torch
import torch.nn.functional as F


class ObjectCentricAdapter:
    def encode(self, observations):
        raise NotImplementedError

    def predict_identity(self, observed_positions, observed_features,
                         future_positions, future_features):
        raise NotImplementedError

    def predict_confidence(self, observed_positions, observed_features,
                           future_positions, future_features):
        raise NotImplementedError


class SlotAttentionAdapter(ObjectCentricAdapter):
    def __init__(self, model):
        self.model = model

    def encode(self, observed_positions, observed_features=None):
        self.model.eval()
        with torch.no_grad():
            if isinstance(observed_positions, np.ndarray):
                observed_positions = torch.FloatTensor(observed_positions)
            if observed_features is not None and isinstance(observed_features, np.ndarray):
                observed_features = torch.FloatTensor(observed_features)
            obj_reps = self.model._encode_inputs(observed_positions, observed_features)
            slots = self.model.slot_attention(obj_reps)
        return slots

    def predict_identity(self, observed_positions, observed_features,
                         future_positions, future_features):
        self.model.eval()
        with torch.no_grad():
            pred = self.model.predict_identity(
                observed_positions, observed_features,
                future_features=future_features)
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        return pred

    def predict_confidence(self, observed_positions, observed_features,
                           future_positions, future_features):
        self.model.eval()
        with torch.no_grad():
            if isinstance(observed_positions, np.ndarray):
                observed_positions = torch.FloatTensor(observed_positions)
            if observed_features is not None and isinstance(observed_features, np.ndarray):
                observed_features = torch.FloatTensor(observed_features)
            if future_features is not None and isinstance(future_features, np.ndarray):
                future_features = torch.FloatTensor(future_features)

            _, assignment_logits = self.model(
                observed_positions, observed_features, future_features)

            if assignment_logits is not None:
                probs = F.softmax(assignment_logits, dim=-1)
                max_probs = probs.max(dim=-1)[0]
                return max_probs.cpu().numpy()
        return None
