import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def _closest_centroid(self, X):
        labels = []
        for p in X:
            dists = np.linalg.norm(p - self.centroids, axis=1)
            labels.append(np.argmin(dists))
        return np.array(labels)

    def fit(self, X):
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]

        for _ in range(self.max_iters):
            labels = self._closest_centroid(X)

            new_centroids = []
            for i in range(self.k):
                pts = X[labels == i]
                new_centroids.append(pts.mean(axis=0) if len(pts) > 0 else self.centroids[i])

            new_centroids = np.array(new_centroids)

            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids

        return labels


