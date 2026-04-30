"""
Base adapter interface for external object-centric models.

All external models must implement this interface to run through
the SVT stress test pipeline.
"""

import numpy as np
import torch


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

    def encode(self, observations):
        self.model.eval()
        with torch.no_grad():
            if isinstance(observations, np.ndarray):
                observations = torch.FloatTensor(observations)
            slots = self.model.encode(observations)
        return slots

    def predict_identity(self, observed_positions, observed_features,
                         future_positions, future_features):
        self.model.eval()
        with torch.no_grad():
            pred = self.model.predict_identity(
                observed_positions, observed_features,
                future_positions, future_features)
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        return pred

    def predict_confidence(self, observed_positions, observed_features,
                           future_positions, future_features):
        if not hasattr(self.model, 'predict_confidence'):
            return None
        self.model.eval()
        with torch.no_grad():
            conf = self.model.predict_confidence(
                observed_positions, observed_features,
                future_positions, future_features)
        if isinstance(conf, torch.Tensor):
            conf = conf.cpu().numpy()
        return conf
