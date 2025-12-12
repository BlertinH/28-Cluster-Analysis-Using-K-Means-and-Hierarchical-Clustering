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
        max_iteration=5
    ):
        if distance_metric not in VALID_METRICS:
            raise ValueError(f"Invalid distance metric: {distance_metric}")

        self.k = k
        self.distance_metric = distance_metric
        self.max_iters = max_iters
        self.random_state = random_state
        self.batch_size = batch_size
        self.tol = tol
        self.max_iteration = max_iteration

        self.rng = np.random.default_rng(random_state)

        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.algorithm_used = None

    def _validate_input(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Input must be 2D")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("NaN or Inf detected")
        return X

    def _compute_inertia(self, X, centroids):
        d = cdist(X, centroids, metric=self.distance_metric)
        min_dist = np.min(d, axis=1)
        return (
            np.sum(min_dist ** 2)
            if self.distance_metric == "euclidean"
            else np.sum(min_dist)
        )

    def _compute_centroid(self, pts):
        if len(pts) == 0:
            return None
        return (
            np.median(pts, axis=0)
            if self.distance_metric == "cityblock"
            else np.mean(pts, axis=0)
        )

    def _init_kmeanspp(self, X):
        n = len(X)
        centroids = [X[self.rng.integers(n)]]

        for _ in range(1, self.k):
            d = cdist(X, np.array(centroids), metric=self.distance_metric)
            min_dist = np.min(d, axis=1)
            probs = np.maximum(min_dist ** 2, 1e-12)
            probs /= probs.sum()
            centroids.append(X[self.rng.choice(n, p=probs)])

        return np.array(centroids)

    def _lloyd(self, X, init_centroids):
        centroids = init_centroids.copy()
        prev_inertia = None

        for _ in range(self.max_iters):
            d = cdist(X, centroids, metric=self.distance_metric)
            labels = np.argmin(d, axis=1)

            new_centroids = []
            for j in range(len(centroids)):
                pts = X[labels == j]
                c = self._compute_centroid(pts)

                if c is None:
                    far_idx = np.argmax(np.min(d, axis=1))
                    c = X[far_idx]

                new_centroids.append(c)

            centroids = np.array(new_centroids)
            inertia = self._compute_inertia(X, centroids)

            if prev_inertia is not None and abs(prev_inertia - inertia) < self.tol:
                break

            prev_inertia = inertia

        labels = np.argmin(cdist(X, centroids, metric=self.distance_metric), axis=1)
        inertia = self._compute_inertia(X, centroids)
        return centroids, labels, inertia

    def _minibatch(self, X, init_centroids):
        centroids = init_centroids.copy()

        for it in range(1, self.max_iters + 1):
            batch_idx = self.rng.choice(len(X), size=self.batch_size, replace=True)
            batch = X[batch_idx]

            d = cdist(batch, centroids, metric=self.distance_metric)
            labels = np.argmin(d, axis=1)

            lr = 1.0 / np.sqrt(it)

            for j in range(len(centroids)):
                pts = batch[labels == j]
                if len(pts) == 0:
                    continue

                update = self._compute_centroid(pts)
                centroids[j] = (1 - lr) * centroids[j] + lr * update

        labels = np.argmin(cdist(X, centroids, metric=self.distance_metric), axis=1)
        inertia = self._compute_inertia(X, centroids)
        return centroids, labels, inertia

    def _two_means(self, X_sub):
        n = len(X_sub)
        c1 = X_sub[self.rng.integers(n)]
        d = cdist(X_sub, c1[None, :], metric=self.distance_metric).ravel()
        probs = d / d.sum() if d.sum() > 0 else None
        c2 = X_sub[self.rng.choice(n, p=probs)]
        return self._lloyd(X_sub, np.vstack([c1, c2]))

    def _cluster_inertia(self, X, idxs):
        if len(idxs) == 0:
            return 0.0
        c = self._compute_centroid(X[idxs])
        return self._compute_inertia(X[idxs], c[None, :])

    def _bisecting_kmeans(self, X):
        n = len(X)
        clusters = [np.arange(n)]

        while len(clusters) < self.k:
            worst = max(
                range(len(clusters)),
                key=lambda i: self._cluster_inertia(X, clusters[i]),
            )

            idxs = clusters.pop(worst)
            if len(idxs) <= 2:
                clusters.append(idxs)
                continue

            _, labels2, _ = self._two_means(X[idxs])
            left = idxs[labels2 == 0]
            right = idxs[labels2 == 1]

            if len(left) == 0 or len(right) == 0:
                clusters.append(idxs)
                continue

            clusters.extend([left, right])

        centroids = []
        labels = np.empty(n, dtype=int)

        for i, idxs in enumerate(clusters):
            centroids.append(self._compute_centroid(X[idxs]))
            labels[idxs] = i

        centroids = np.array(centroids)
        inertia = self._compute_inertia(X, centroids)
        return centroids, labels, inertia

    def fit(self, X):
        X = self._validate_input(X)
        n = len(X)

        if self.k > n:
            raise ValueError("k cannot exceed number of samples")

        if n >= 5000 and self.k <= 10:
            self.algorithm_used = "minibatch"
            init = self._init_kmeanspp(X)
            centroids, labels, inertia = self._minibatch(X, init)

        elif self.k >= 10:
            self.algorithm_used = "bisecting"
            centroids, labels, inertia = self._bisecting_kmeans(X)

        else:
            self.algorithm_used = "kmeans++"
            init = self._init_kmeanspp(X)
            centroids, labels, inertia = self._lloyd(X, init)

        self.centroids = centroids
        self.labels = labels
        self.inertia_ = inertia
        return labels

    def predict(self, X_new):
        if self.centroids is None:
            raise RuntimeError("Model not fitted")
        X_new = self._validate_input(X_new)
        return np.argmin(
            cdist(X_new, self.centroids, metric=self.distance_metric), axis=1
        )
