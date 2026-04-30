"""
FeatureAwareIdentityBaseline for SVT-v2.2
"""

import numpy as np


class FeatureAwareIdentityBaseline:
    def predict_identity(self, observed_features=None, future_features=None):
        if observed_features is None:
            return None

        if observed_features.ndim == 3:
            observed_features = observed_features[np.newaxis]
            if future_features is not None:
                future_features = future_features[np.newaxis]

        B, T, N, F = observed_features.shape
        ids = np.tile(np.arange(N), (B, 1))

        if N != 2:
            return ids

        obs_feat_ref = observed_features[:, 0]

        if future_features is not None:
            fut_feat_ref = future_features[:, 0]
        else:
            return ids

        for i in range(B):
            all_same = np.allclose(obs_feat_ref[i, 0], obs_feat_ref[i, 1])
            if all_same:
                continue

            sim_no_swap = (
                np.dot(obs_feat_ref[i, 0], fut_feat_ref[i, 0]) +
                np.dot(obs_feat_ref[i, 1], fut_feat_ref[i, 1])
            )
            sim_swap = (
                np.dot(obs_feat_ref[i, 0], fut_feat_ref[i, 1]) +
                np.dot(obs_feat_ref[i, 1], fut_feat_ref[i, 0])
            )

            if sim_swap > sim_no_swap:
                ids[i] = np.array([1, 0])

        return ids

    def score(self, observed_features=None, future_features=None, identity_labels=None):
        pred_ids = self.predict_identity(observed_features, future_features)
        if pred_ids is None or identity_labels is None:
            return 0.5
        return float(np.mean(pred_ids == identity_labels))
