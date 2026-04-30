import numpy as np
from typing import Optional
from sklearn.neighbors import NearestNeighbors


class BaseKNNV2:
    def __init__(self, k: int = 5, weighting: str = "inverse_distance"):
        self.k = k
        self.weighting = weighting
        self.train_obs = None
        self.train_future = None
        self.train_identity = None
        self.nn = None

    def fit(self, train_observed, train_future, train_identity=None):
        raise NotImplementedError

    def predict_future(self, test_observed):
        raise NotImplementedError

    def predict_identity(self, test_observed, test_future=None):
        raise NotImplementedError

    def _get_weights(self, distances):
        if self.weighting == "uniform":
            return np.ones_like(distances)
        elif self.weighting == "inverse_distance":
            weights = 1.0 / (distances + 1e-8)
            return weights / weights.sum(axis=1, keepdims=True)
        else:
            return np.ones_like(distances)

    def _identity_by_trajectory_match(self, pred_future, test_future):
        B, T, N, D = pred_future.shape
        pred_ids = np.tile(np.arange(N), (B, 1))

        if N != 2 or test_future is None:
            return pred_ids

        for i in range(B):
            mse_no_swap = np.mean((pred_future[i] - test_future[i]) ** 2)
            swapped_pred = pred_future[i].copy()
            swapped_pred[:, [0, 1]] = swapped_pred[:, [1, 0]]
            mse_swap = np.mean((swapped_pred - test_future[i]) ** 2)

            if mse_swap < mse_no_swap:
                pred_ids[i] = np.array([1, 0])

        return pred_ids


class RawDeltaKNN(BaseKNNV2):
    def fit(self, train_observed, train_future, train_identity=None):
        self.train_obs = train_observed.reshape(train_observed.shape[0], -1)
        self.train_future = train_future
        self.train_identity = train_identity
        self.train_obs_last = train_observed[:, -1].copy()
        self.train_future_delta = train_future - train_observed[:, -1][:, None, :, :]
        self.nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        self.nn.fit(self.train_obs)

    def predict_future(self, test_observed):
        test_flat = test_observed.reshape(test_observed.shape[0], -1)
        distances, indices = self.nn.kneighbors(test_flat)
        weights = self._get_weights(distances)

        pred_delta = np.zeros((test_observed.shape[0],) + self.train_future_delta.shape[1:])
        for i in range(test_observed.shape[0]):
            for j, idx in enumerate(indices[i]):
                pred_delta[i] += weights[i, j] * self.train_future_delta[idx]

        test_last = test_observed[:, -1]
        pred = test_last[:, None, :, :] + pred_delta
        return pred

    def predict_identity(self, test_observed, test_future=None):
        pred_future = self.predict_future(test_observed)
        return self._identity_by_trajectory_match(pred_future, test_future)


class TranslationNormalizedDeltaKNN(BaseKNNV2):
    def fit(self, train_observed, train_future, train_identity=None):
        self.train_obs_raw = train_observed.copy()
        self.train_future = train_future.copy()
        self.train_identity = train_identity

        self.centers = []
        train_norm = train_observed.copy()
        for i in range(train_norm.shape[0]):
            center = train_norm[i].mean(axis=(0, 1), keepdims=True)
            train_norm[i] = train_norm[i] - center
            self.centers.append(center)

        self.train_future_delta = train_future - train_observed[:, -1][:, None, :, :]
        flat = train_norm.reshape(train_norm.shape[0], -1)
        self.nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        self.nn.fit(flat)

    def predict_future(self, test_observed):
        test_norm = test_observed.copy()
        test_centers = []
        for i in range(test_norm.shape[0]):
            center = test_norm[i].mean(axis=(0, 1), keepdims=True)
            test_norm[i] = test_norm[i] - center
            test_centers.append(center)

        test_flat = test_norm.reshape(test_norm.shape[0], -1)
        distances, indices = self.nn.kneighbors(test_flat)
        weights = self._get_weights(distances)

        pred_delta = np.zeros((test_observed.shape[0],) + self.train_future_delta.shape[1:])
        for i in range(test_observed.shape[0]):
            for j, idx in enumerate(indices[i]):
                pred_delta[i] += weights[i, j] * self.train_future_delta[idx]

        test_last = test_observed[:, -1]
        pred = test_last[:, None, :, :] + pred_delta
        return pred

    def predict_identity(self, test_observed, test_future=None):
        pred_future = self.predict_future(test_observed)
        return self._identity_by_trajectory_match(pred_future, test_future)


class VelocityDeltaKNN(BaseKNNV2):
    def fit(self, train_observed, train_future, train_identity=None):
        self.train_vel = np.diff(train_observed, axis=1)
        self.train_future = train_future
        self.train_identity = train_identity
        self.train_obs_last = train_observed[:, -1].copy()
        self.train_future_delta = train_future - train_observed[:, -1][:, None, :, :]
        flat = self.train_vel.reshape(self.train_vel.shape[0], -1)
        self.nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        self.nn.fit(flat)

    def predict_future(self, test_observed):
        test_vel = np.diff(test_observed, axis=1)
        test_flat = test_vel.reshape(test_vel.shape[0], -1)
        distances, indices = self.nn.kneighbors(test_flat)
        weights = self._get_weights(distances)

        pred_delta = np.zeros((test_observed.shape[0],) + self.train_future_delta.shape[1:])
        for i in range(test_observed.shape[0]):
            for j, idx in enumerate(indices[i]):
                pred_delta[i] += weights[i, j] * self.train_future_delta[idx]

        test_last = test_observed[:, -1]
        pred = test_last[:, None, :, :] + pred_delta
        return pred

    def predict_identity(self, test_observed, test_future=None):
        pred_future = self.predict_future(test_observed)
        return self._identity_by_trajectory_match(pred_future, test_future)


class PermutationConsistentDeltaKNN(BaseKNNV2):
    def fit(self, train_observed, train_future, train_identity=None):
        self.train_obs = train_observed.copy()
        self.train_future = train_future.copy()
        self.train_identity = train_identity
        self.train_obs_last = train_observed[:, -1].copy()
        self.train_future_delta = train_future - train_observed[:, -1][:, None, :, :]
        flat = self.train_obs.reshape(self.train_obs.shape[0], -1)
        self.nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        self.nn.fit(flat)

    def predict_future(self, test_observed):
        test_flat = test_observed.reshape(test_observed.shape[0], -1)
        distances, indices = self.nn.kneighbors(test_flat)
        weights = self._get_weights(distances)

        pred_delta = np.zeros((test_observed.shape[0],) + self.train_future_delta.shape[1:])
        for i in range(test_observed.shape[0]):
            for j, idx in enumerate(indices[i]):
                pred_delta[i] += weights[i, j] * self.train_future_delta[idx]

        test_last = test_observed[:, -1]
        pred = test_last[:, None, :, :] + pred_delta
        return pred

    def predict_identity(self, test_observed, test_future=None):
        pred_future = self.predict_future(test_observed)
        return self._identity_by_trajectory_match(pred_future, test_future)


class LastVelocityBaseline:
    def __init__(self):
        self.t_pred = None

    def fit(self, train_observed, train_future, train_identity=None):
        self.t_pred = train_future.shape[1]

    def predict_future(self, test_observed):
        T_obs, N, D = test_observed.shape[1:]
        t_pred = self.t_pred if self.t_pred is not None else 10
        last_pos = test_observed[:, -1]
        last_vel = test_observed[:, -1] - test_observed[:, -2]

        pred = np.zeros((test_observed.shape[0], t_pred, N, D))
        pos = last_pos.copy()
        for t in range(t_pred):
            pos = pos + last_vel
            pred[:, t] = pos
        return pred

    def predict_identity(self, test_observed, test_future=None):
        pred_future = self.predict_future(test_observed)
        B, T, N, D = pred_future.shape
        pred_ids = np.tile(np.arange(N), (B, 1))

        if N != 2 or test_future is None:
            return pred_ids

        for i in range(B):
            mse_no_swap = np.mean((pred_future[i] - test_future[i]) ** 2)
            swapped_pred = pred_future[i].copy()
            swapped_pred[:, [0, 1]] = swapped_pred[:, [1, 0]]
            mse_swap = np.mean((swapped_pred - test_future[i]) ** 2)

            if mse_swap < mse_no_swap:
                pred_ids[i] = np.array([1, 0])

        return pred_ids


class KNNLastVelocityBlend:
    def __init__(self, knn_model, alpha: float = 0.5):
        self.knn = knn_model
        self.baseline = LastVelocityBaseline()
        self.alpha = alpha

    def fit(self, train_observed, train_future, train_identity=None):
        self.knn.fit(train_observed, train_future, train_identity)
        self.baseline.fit(train_observed, train_future, train_identity)

    def predict_future(self, test_observed):
        knn_pred = self.knn.predict_future(test_observed)
        base_pred = self.baseline.predict_future(test_observed)
        return self.alpha * knn_pred + (1 - self.alpha) * base_pred

    def predict_identity(self, test_observed, test_future=None):
        pred_future = self.predict_future(test_observed)
        B, T, N, D = pred_future.shape
        pred_ids = np.tile(np.arange(N), (B, 1))

        if N != 2 or test_future is None:
            return pred_ids

        for i in range(B):
            mse_no_swap = np.mean((pred_future[i] - test_future[i]) ** 2)
            swapped_pred = pred_future[i].copy()
            swapped_pred[:, [0, 1]] = swapped_pred[:, [1, 0]]
            mse_swap = np.mean((swapped_pred - test_future[i]) ** 2)

            if mse_swap < mse_no_swap:
                pred_ids[i] = np.array([1, 0])

        return pred_ids


KNN_V2_REGISTRY = {
    "RawDeltaKNN": RawDeltaKNN,
    "TranslationNormalizedDeltaKNN": TranslationNormalizedDeltaKNN,
    "VelocityDeltaKNN": VelocityDeltaKNN,
    "PermutationConsistentDeltaKNN": PermutationConsistentDeltaKNN,
    "LastVelocityBaseline": LastVelocityBaseline,
}
