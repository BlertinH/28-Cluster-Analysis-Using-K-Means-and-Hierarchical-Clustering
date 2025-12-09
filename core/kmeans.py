import numpy as np
from scipy.spatial.distance import cdist

VALID_METRICS = ["euclidean", "cityblock"]


class KMeans:
    def __init__(
        self,
        k=3,
        distance_metric="euclidean",
        max_iters=300,
        random_state=None,
        batch_size=100,
        tol=1e-4,
    ):
        self.k = k
        self.max_iters = max_iters
        self.distance_metric = distance_metric
        self.random_state = random_state
        self.batch_size = batch_size
        self.tol = tol

        if distance_metric not in VALID_METRICS:
            raise ValueError(f"Invalid distance metric: {distance_metric}")

        self.rng = np.random.default_rng(random_state)

        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.algorithm_used = None

    def _validate_input(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array.")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Input data contains NaN or Inf.")
        return X

    def _compute_inertia(self, X, centroids):
        distances = cdist(X, centroids, metric=self.distance_metric)
        min_dist = np.min(distances, axis=1)

        if self.distance_metric == "euclidean":
            return np.sum(min_dist ** 2)
        else:
            return np.sum(min_dist)

    def _compute_centroid(self, pts):
        if len(pts) == 0:
            return None

        if self.distance_metric == "cityblock":
            return np.median(pts, axis=0)
        return np.mean(pts, axis=0)

    def _init_kmeanspp(self, X):
        n = len(X)
        centroids = [X[self.rng.integers(n)]]

        for _ in range(1, self.k):
            dist_matrix = cdist(X, np.array(centroids), metric=self.distance_metric)
            min_dist = np.min(dist_matrix, axis=1)
            min_dist_sq = np.maximum(min_dist ** 2, 1e-12)

            probs = min_dist_sq / min_dist_sq.sum()
            idx = self.rng.choice(n, p=probs)
            centroids.append(X[idx])

        return np.array(centroids)

    def _lloyd(self, X, init_centroids):
        centroids = init_centroids.copy()
        k_local = centroids.shape[0]
        prev_inertia = None

        for _ in range(self.max_iters):
            distances = cdist(X, centroids, metric=self.distance_metric)
            labels = np.argmin(distances, axis=1)

            new_centroids = []
            for j in range(k_local):
                pts = X[labels == j]
                c = self._compute_centroid(pts)
                if c is None:
                    far_idx = np.argmax(
                        np.min(cdist(X, centroids, metric=self.distance_metric), axis=1)
                    )
                    c = X[far_idx]
                new_centroids.append(c)

            new_centroids = np.array(new_centroids)
            inertia = self._compute_inertia(X, new_centroids)

            if prev_inertia is not None and abs(prev_inertia - inertia) < self.tol:
                break

            prev_inertia = inertia
            centroids = new_centroids

        final_dist = cdist(X, centroids, metric=self.distance_metric)
        labels = np.argmin(final_dist, axis=1)
        inertia = self._compute_inertia(X, centroids)

        return centroids, labels, inertia

    def fit(self, X):
        X = self._validate_input(X)
        n = len(X)

        if self.k > n:
            raise ValueError("k cannot exceed number of samples")

        self.algorithm_used = "kmeans++"
        init = self._init_kmeanspp(X)
        centroids, labels, inertia = self._lloyd(X, init)

        self.centroids = centroids
        self.labels = labels
        self.inertia_ = inertia
        return labels