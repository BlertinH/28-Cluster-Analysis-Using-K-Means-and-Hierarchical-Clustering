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
        max_iteration = 5
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
        self.max_iteration = max_iteration
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

        pts_trimmed = self._trim_cluster_points(pts, keep_ratio=0.90)

        if self.distance_metric == "cityblock":
            return np.median(pts_trimmed, axis=0)

        return np.mean(pts_trimmed, axis=0)

    def _trim_cluster_points(self, pts, keep_ratio=0.90):
        if len(pts) <= 2:
            return pts

        centroid = np.mean(pts, axis=0)
        dist = np.linalg.norm(pts - centroid, axis=1)

        keep_n = max(1, int(len(pts) * keep_ratio))
        keep_idx = np.argsort(dist)[:keep_n]

        return pts[keep_idx]

    def _init_kmeanspp(self, X):
        n = len(X)

        centroids = [X[self.rng.integers(n)]]

        if self.k == 1:
            return np.array(centroids)

        for _ in range(1, self.k):
            dist_matrix = cdist(X, np.array(centroids), metric=self.distance_metric)
            min_dist = np.min(dist_matrix, axis=1)

            cutoff = np.percentile(min_dist, 90)
            mask = min_dist < cutoff

            if mask.sum() == 0:
                probs = np.ones_like(min_dist) / len(min_dist)
            else:
                safe_dist = min_dist[mask]
                safe_dist_sq = np.maximum(safe_dist ** 2, 1e-12)
                probs = np.zeros_like(min_dist)
                probs[mask] = safe_dist_sq / safe_dist_sq.sum()

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
                    d_global = distances.min(axis=1)
                    far_idx = np.argmax(d_global)
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

    def _minibatch(self, X, init_centroids):
        centroids = init_centroids.copy()
        k_local = centroids.shape[0]

        for it in range(1, self.max_iters + 1):
            batch_idx = self.rng.choice(len(X), size=self.batch_size, replace=True)
            batch = X[batch_idx]

            distances = cdist(batch, centroids, metric=self.distance_metric)
            labels = np.argmin(distances, axis=1)

            lr = 1.0 / np.sqrt(it)

            for j in range(k_local):
                pts = batch[labels == j]
                if len(pts) == 0:
                    continue

                pts_trimmed = self._trim_cluster_points(pts, keep_ratio=0.90)

                if self.distance_metric == "cityblock":
                    update = np.median(pts_trimmed, axis=0)
                else:
                    update = np.mean(pts_trimmed, axis=0)

                if np.any(np.isnan(update)) or np.any(np.isinf(update)):
                    continue

                centroids[j] = centroids[j] * (1 - lr) + update * lr

        full_dist = cdist(X, centroids, metric=self.distance_metric)
        final_labels = np.argmin(full_dist, axis=1)
        inertia = self._compute_inertia(X, centroids)

        return centroids, final_labels, inertia

    def _two_means(self, X_sub):
        n_sub = len(X_sub)

        c1 = X_sub[self.rng.integers(n_sub)]
        dist = cdist(X_sub, c1[None, :], metric=self.distance_metric).ravel()
        if dist.sum() == 0:
            c2 = X_sub[self.rng.integers(n_sub)]
        else:
            probs = dist / dist.sum()
            c2 = X_sub[self.rng.choice(n_sub, p=probs)]

        init = np.array([c1, c2])
        centroids, labels, inertia = self._lloyd(X_sub, init)
        return centroids, labels, inertia

    def _cluster_inertia(self, X, idxs):
        if len(idxs) == 0:
            return 0.0
        pts = X[idxs]
        c = self._compute_centroid(pts)
        return self._compute_inertia(pts, c[None, :])

    def _bisecting_kmeans(self, X):
        n = len(X)
        clusters = [np.arange(n)]
        last_split = None

        while len(clusters) < self.k:
            candidates = [i for i in range(len(clusters))
                          if clusters[i].tobytes() != (last_split.tobytes() if last_split is not None else None)]

            if not candidates:
                candidates = list(range(len(clusters)))

            worst_idx = max(
                candidates,
                key=lambda idx: self._cluster_inertia(X, clusters[idx]),
            )

            idxs = clusters.pop(worst_idx)
            last_split = idxs.copy()

            if len(idxs) <= 2:
                clusters.append(idxs)
                if all(len(c) <= 2 for c in clusters):
                    break
                continue

            X_sub = X[idxs]

            attempts = 0
            while attempts < self.max_iteration:
                c2, labels2, _ = self._two_means(X_sub)
                left = idxs[labels2 == 0]
                right = idxs[labels2 == 1]

                if len(left) > 0 and len(right) > 0:
                    break

                attempts += 1

            if len(left) == 0 or len(right) == 0:
                clusters.append(idxs)
                continue

            clusters.append(left)
            clusters.append(right)

            if len(clusters) > self.k:
                break

        centroids = []
        labels = np.empty(n, dtype=int)

        for cluster_label, idxs in enumerate(clusters):
            c = self._compute_centroid(X[idxs])
            centroids.append(c)
            labels[idxs] = cluster_label

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

        elif self.k >= 20 or (self.k >= 10 and n >= 2000):
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