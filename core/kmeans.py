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
            tol=1e-4,
    ):
        self.k = k
        self.max_iters = max_iters
        self.distance_metric = distance_metric
        self.random_state = random_state
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
        d = cdist(X, centroids, metric=self.distance_metric)
        min_dist = np.min(d, axis=1)
        return np.sum(min_dist ** 2) if self.distance_metric == "euclidean" else np.sum(min_dist)

    def _compute_centroid(self, pts):
        if len(pts) == 0:
            return None
        return np.median(pts, axis=0) if self.distance_metric == "cityblock" else np.mean(pts, axis=0)

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
        prev_inertia = None

        for _ in range(self.max_iters):
            distances = cdist(X, centroids, metric=self.distance_metric)
            labels = np.argmin(distances, axis=1)

            new_centroids = []
            for j in range(len(centroids)):
                pts = X[labels == j]
                c = self._compute_centroid(pts)

                if c is None:  # empty cluster fix
                    far_idx = np.argmax(np.min(cdist(X, centroids), axis=1))
                    c = X[far_idx]

                new_centroids.append(c)

            new_centroids = np.array(new_centroids)
            inertia = self._compute_inertia(X, new_centroids)

            if prev_inertia is not None and abs(prev_inertia - inertia) < self.tol:
                break

            prev_inertia = inertia
            centroids = new_centroids

        distances = cdist(X, centroids, metric=self.distance_metric)
        labels = np.argmin(distances, axis=1)
        inertia = self._compute_inertia(X, centroids)

        return centroids, labels, inertia

    def _two_means(self, X_sub):
        n = len(X_sub)

        c1 = X_sub[self.rng.integers(n)]
        dist = cdist(X_sub, c1[None, :], metric=self.distance_metric).ravel()

        if dist.sum() == 0:
            c2 = c1.copy()
        else:
            probs = dist / dist.sum()
            c2 = X_sub[self.rng.choice(n, p=probs)]

        init = np.vstack([c1, c2])
        return self._lloyd(X_sub, init)

    def _bisecting_kmeans(self, X):
        n = len(X)
        clusters = [np.arange(n)]

        while len(clusters) < self.k:
            worst_idx = max(
                range(len(clusters)),
                key=lambda idx: self._compute_inertia(X[clusters[idx]],
                                                      self._compute_centroid(X[clusters[idx]])[None, :])
            )

            idxs = clusters.pop(worst_idx)

            if len(idxs) <= 2:
                clusters.append(idxs)
                continue

            X_sub = X[idxs]
            _, labels2, _ = self._two_means(X_sub)

            left = idxs[labels2 == 0]
            right = idxs[labels2 == 1]

            if len(left) == 0 or len(right) == 0:
                clusters.append(idxs)
                continue

            clusters.append(left)
            clusters.append(right)

        centroids = []
        labels = np.empty(n, dtype=int)

        for cluster_label, idxs in enumerate(clusters):
            c = self._compute_centroid(X[idxs])
            centroids.append(c)
            labels[idxs] = cluster_label

        return np.array(centroids), labels, self._compute_inertia(X, np.array(centroids))

    def fit(self, X):
        X = self._validate_input(X)
        n = len(X)

        if self.k > n:
            raise ValueError("k cannot exceed number of samples")

        # Choose algorithm
        if self.k <= 10:
            self.algorithm_used = "kmeans++"
            init = self._init_kmeanspp(X)
            centroids, labels, inertia = self._lloyd(X, init)
        else:
            self.algorithm_used = "bisecting"
            centroids, labels, inertia = self._bisecting_kmeans(X)

        self.centroids = centroids
        self.labels = labels
        self.inertia_ = inertia
        return labels

    def predict(self, X_new):
        if self.centroids is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_new = self._validate_input(X_new)
        dist = cdist(X_new, self.centroids, metric=self.distance_metric)
        return np.argmin(dist, axis=1)
