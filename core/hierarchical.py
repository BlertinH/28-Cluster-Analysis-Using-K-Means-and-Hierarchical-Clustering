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
    mapped = np.array([mapping[l] for l in raw_labels])
    return mapped, seen


def hierarchical_clustering(data, metric="euclidean", k=3, random_state=None):

    if random_state is not None:
        np.random.seed(random_state)

    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or Inf")

    n, d = data.shape

    if not (1 <= k <= n):
        raise ValueError("Invalid number of clusters")

    if metric not in ["euclidean", "cityblock"]:
        warnings.warn(f"Invalid metric '{metric}', switching to euclidean.")
        metric = "euclidean"

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    if n <= 5000:
        method = "ward" if metric == "euclidean" else "complete"
        Z = linkage(data_scaled, method=method, metric=metric)

        dendro_fig = plt.figure(figsize=(10, 6))
        dendrogram(Z)
        plt.title(f"Dendrogram ({method}, metric={metric})")
        plt.xlabel("Samples")
        plt.ylabel("Distance")

        raw = fcluster(Z, k, criterion="maxclust")
        labels, _ = normalize_preserving_order(raw)

        plt.close(dendro_fig)
        return labels, dendro_fig

    if n <= 50000:
        if metric == "cityblock":
            model = AgglomerativeClustering(
                n_clusters=k,
                metric="manhattan",
                linkage="complete"
            )
        else:
            model = AgglomerativeClustering(
                n_clusters=k,
                metric="euclidean",
                linkage="ward"
            )

        labels = model.fit_predict(data_scaled)
        return labels, None

    if FASTCLUSTER_AVAILABLE:

        print(f"[INFO] Using fastcluster for large dataset (n={n})")

        if metric == "euclidean":
            Z = fastcluster.linkage_vector(data_scaled, method="ward")
        else:
            Z = fastcluster.linkage(data_scaled, method="complete", metric="cityblock")

        raw = fcluster(Z, k, criterion="maxclust")
        labels, _ = normalize_preserving_order(raw)
        return labels, None

    print(f"[INFO] Using PCA reduction for large dataset (n={n})")

    pca = PCA(n_components=min(50, d, n - 1))
    reduced = pca.fit_transform(data_scaled)

    if metric == "cityblock":
        model = AgglomerativeClustering(
            n_clusters=k, metric="manhattan", linkage="complete"
        )
    else:
        model = AgglomerativeClustering(
            n_clusters=k, metric="euclidean", linkage="ward"
        )

    labels = model.fit_predict(reduced)
    return labels, None
