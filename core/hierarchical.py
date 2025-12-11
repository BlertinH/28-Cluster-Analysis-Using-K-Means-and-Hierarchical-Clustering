import numpy as np
import warnings
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    import fastcluster
    FASTCLUSTER_AVAILABLE = True
except ImportError:
    FASTCLUSTER_AVAILABLE = False


def normalize_preserving_order(raw_labels):
    seen = []
    for lab in raw_labels:
        if lab not in seen:
            seen.append(lab)
    mapping = {lab: i for i, lab in enumerate(seen)}
    return np.array([mapping[l] for l in raw_labels]), seen


def hierarchical_clustering(data, metric="euclidean", k=3, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError("Data must be 2D")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Invalid values")

    n, d = X.shape
    if not (1 <= k <= n):
        raise ValueError("Invalid cluster count")

    if metric not in ["euclidean", "cityblock"]:
        warnings.warn("")
        metric = "euclidean"

    Xs = StandardScaler().fit_transform(X)
    dendro_fig = None

    if n <= 5000:
        method = "ward" if metric == "euclidean" else "complete"
        Z = linkage(Xs, method=method, metric=metric)
        fig = plt.figure(figsize=(10, 6))
        dendrogram(Z)
        raw = fcluster(Z, k, criterion="maxclust")
        labels, _ = normalize_preserving_order(raw)
        plt.close(fig)
        return labels, fig

    if n <= 50000:
        if metric == "cityblock":
            model = AgglomerativeClustering(n_clusters=k, metric="manhattan", linkage="complete")
        else:
            model = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
        labels = model.fit_predict(Xs)
        return labels, None

    if FASTCLUSTER_AVAILABLE:
        if metric == "euclidean":
            Z = fastcluster.linkage_vector(Xs, method="ward")
        else:
            Z = fastcluster.linkage(Xs, method="complete", metric="cityblock")
        raw = fcluster(Z, k, criterion="maxclust")
        labels, _ = normalize_preserving_order(raw)
        return labels, None

    pca = PCA(n_components=min(50, d, n - 1))
    reduced = pca.fit_transform(Xs)

    if metric == "cityblock":
        model = AgglomerativeClustering(n_clusters=k, metric="manhattan", linkage="complete")
    else:
        model = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")

    labels = model.fit_predict(reduced)
    return labels, None