import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

VALID_METRICS = ["euclidean", "cityblock"]
MAX_HIER_SAMPLES = 3000


def _compute_prototypes(X, labels, metric):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique = np.unique(labels)

    prototypes = []
    for lab in unique:
        pts = X[labels == lab]
        if pts.size == 0:
            continue

        if metric == "cityblock":
            prototypes.append(np.median(pts, axis=0))
            prototypes.append(np.mean(pts, axis=0))

    return np.array(prototypes), unique


def hierarchical_clustering(data, metric, num_clusters):
    X = np.asarray(data, dtype=float)

    # Basic validations
    if X.ndim != 2:
        raise ValueError(f"Data must be 2D array, got shape {X.shape}")
    n = len(X)
    if n < 2:
        raise ValueError("Need at least 2 points for hierarchical clustering")
    if not (1 <= num_clusters <= n):
        raise ValueError(f"num_clusters must be between 1 and {n}")

    # Metric + method selection
    if metric not in VALID_METRICS:
        metric = "euclidean"
    method = "ward" if metric == "euclidean" else "complete"
    metric_for_linkage = "euclidean" if metric == "euclidean" else metric

    try:
        if n <= MAX_HIER_SAMPLES:
            Z = linkage(X, method=method, metric=metric_for_linkage)

            fig = plt.Figure(figsize=(7, 4))
            ax = fig.add_subplot(111)
            dendrogram(Z, ax=ax)

            ax.set_title(f"Dendrogram ({method}, metric={metric_for_linkage})")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Distance")
            ax.grid(True)

            labels = fcluster(Z, num_clusters, criterion="maxclust") - 1
            return labels, fig

        rng = np.random.default_rng(42)
        sample_size = min(n, MAX_HIER_SAMPLES)
        idx_sample = rng.choice(n, size=sample_size, replace=False)
        X_sample = X[idx_sample]

        Z_sample = linkage(X_sample, method=method, metric=metric_for_linkage)
        sample_labels = fcluster(Z_sample, num_clusters, criterion="maxclust") - 1

        protos, _ = _compute_prototypes(X_sample, sample_labels, metric_for_linkage)
        dists = cdist(X, protos, metric=metric_for_linkage)
        labels_full = np.argmin(dists, axis=1).astype(int)

        fig = plt.Figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        dendrogram(Z_sample, ax=ax)
        ax.set_title(
            f"Dendrogram (sampled: {sample_size}/{n}, metric={metric_for_linkage})"
        )
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Distance")
        ax.grid(True)

        return labels_full, fig

    except Exception as e:
        print("Hierarchical error:", e)
        return None, None