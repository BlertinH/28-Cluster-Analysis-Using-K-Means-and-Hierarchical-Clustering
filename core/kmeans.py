import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]

        for _ in range(self.max_iters):
            labels = []
            for p in X:
                d = np.linalg.norm(p - self.centroids, axis=1)
                labels.append(np.argmin(d))
            labels = np.array(labels)

            new_centroids = []
            for i in range(self.k):
                pts = X[labels == i]
                if len(pts) > 0:
                    new_centroids.append(pts.mean(axis=0))
                else:
                    new_centroids.append(self.centroids[i])

            new_centroids = np.array(new_centroids)
            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids

        return labels
