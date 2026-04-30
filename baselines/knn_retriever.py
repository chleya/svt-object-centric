import numpy as np
from typing import Optional
from sklearn.neighbors import NearestNeighbors


class BaseKNN:
    def __init__(self, k: int = 5, weighting: str = "uniform"):
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
        """Predict identity by comparing predicted future under swap vs no-swap.
        
        For 2-object episodes: compute MSE between predicted future and actual future
        under both identity assignments [0,1] and [1,0]. Pick the one with lower MSE.
        """
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


class RawTrajectoryKNN(BaseKNN):
    def fit(self, train_observed, train_future, train_identity=None):
        self.train_obs = train_observed.reshape(train_observed.shape[0], -1)
        self.train_future = train_future
        self.train_identity = train_identity
        self.nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        self.nn.fit(self.train_obs)

    def predict_future(self, test_observed):
        test_flat = test_observed.reshape(test_observed.shape[0], -1)
        distances, indices = self.nn.kneighbors(test_flat)
        weights = self._get_weights(distances)

        pred = np.zeros((test_observed.shape[0],) + self.train_future.shape[1:])
        for i in range(test_observed.shape[0]):
            for j, idx in enumerate(indices[i]):
                pred[i] += weights[i, j] * self.train_future[idx]
        return pred

    def predict_identity(self, test_observed, test_future=None):
        pred_future = self.predict_future(test_observed)
        return self._identity_by_trajectory_match(pred_future, test_future)


class TranslationNormalizedKNN(BaseKNN):
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

        pred = np.zeros((test_observed.shape[0],) + self.train_future.shape[1:])
        for i in range(test_observed.shape[0]):
            for j, idx in enumerate(indices[i]):
                pred[i] += weights[i, j] * (self.train_future[idx] - self.centers[idx] + test_centers[i])
        return pred

    def predict_identity(self, test_observed, test_future=None):
        pred_future = self.predict_future(test_observed)
        return self._identity_by_trajectory_match(pred_future, test_future)


class VelocityOnlyKNN(BaseKNN):
    def fit(self, train_observed, train_future, train_identity=None):
        self.train_vel = np.diff(train_observed, axis=1)
        self.train_future = train_future
        self.train_identity = train_identity
        flat = self.train_vel.reshape(self.train_vel.shape[0], -1)
        self.nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        self.nn.fit(flat)

    def predict_future(self, test_observed):
        test_vel = np.diff(test_observed, axis=1)
        test_flat = test_vel.reshape(test_vel.shape[0], -1)
        distances, indices = self.nn.kneighbors(test_flat)
        weights = self._get_weights(distances)

        pred = np.zeros((test_observed.shape[0],) + self.train_future.shape[1:])
        for i in range(test_observed.shape[0]):
            for j, idx in enumerate(indices[i]):
                pred[i] += weights[i, j] * self.train_future[idx]
        return pred

    def predict_identity(self, test_observed, test_future=None):
        pred_future = self.predict_future(test_observed)
        return self._identity_by_trajectory_match(pred_future, test_future)


class PermutationMinKNN(BaseKNN):
    def fit(self, train_observed, train_future, train_identity=None):
        self.train_obs = train_observed.copy()
        self.train_future = train_future.copy()
        self.train_identity = train_identity
        flat = self.train_obs.reshape(self.train_obs.shape[0], -1)
        self.nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        self.nn.fit(flat)

    def predict_future(self, test_observed):
        test_flat = test_observed.reshape(test_observed.shape[0], -1)
        distances, indices = self.nn.kneighbors(test_flat)
        weights = self._get_weights(distances)

        pred = np.zeros((test_observed.shape[0],) + self.train_future.shape[1:])
        for i in range(test_observed.shape[0]):
            for j, idx in enumerate(indices[i]):
                pred[i] += weights[i, j] * self.train_future[idx]
        return pred

    def predict_identity(self, test_observed, test_future=None):
        pred_future = self.predict_future(test_observed)
        return self._identity_by_trajectory_match(pred_future, test_future)


KNN_REGISTRY = {
    "RawTrajectoryKNN": RawTrajectoryKNN,
    "TranslationNormalizedKNN": TranslationNormalizedKNN,
    "VelocityOnlyKNN": VelocityOnlyKNN,
    "PermutationMinKNN": PermutationMinKNN,
}
