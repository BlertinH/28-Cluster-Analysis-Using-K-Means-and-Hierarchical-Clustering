import numpy as np
import warnings
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.preprocessing import StandardScaler

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


def hierarchical_clustering(
    data,
    metric="euclidean",
    k=3,
    random_state=None,
    birch_threshold=0.5
):

    if random_state is not None:
        np.random.seed(random_state)

    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError("Data must be 2D.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or Inf.")

    n, d = data.shape

    if not (1 <= k <= n):
        raise ValueError("Invalid k")

    if metric not in ["euclidean", "cityblock"]:
        warnings.warn("Invalid metric → switching to euclidean")
        metric = "euclidean"

    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    if n > 20000:
        print("[INFO] Using BIRCH (hierarchical, scalable)")

        birch = Birch(
            n_clusters=k,
            threshold=birch_threshold
        )

        labels = birch.fit_predict(X)
        return labels, None

    if FASTCLUSTER_AVAILABLE:
        print("[INFO] Using fastcluster")

        if metric == "euclidean":
            Z = fastcluster.linkage_vector(X, method="ward")
        else:
            Z = fastcluster.linkage(X, method="complete", metric="cityblock")

        labels_raw = fcluster(Z, k, criterion="maxclust")
        labels, _ = normalize_preserving_order(labels_raw)

        fig = plt.figure(figsize=(10, 6))
        dendrogram(Z)
        plt.title("Dendrogram (fastcluster)")
        plt.close(fig)

        return labels, fig

    if n <= 8000:
        print("[INFO] Using scipy linkage (Ward / Complete)")

        method = "ward" if metric == "euclidean" else "complete"
        Z = linkage(X, method=method, metric=metric)

        labels_raw = fcluster(Z, k, criterion="maxclust")
        labels, _ = normalize_preserving_order(labels_raw)

        fig = plt.figure(figsize=(10, 6))
        dendrogram(Z)
        plt.title(f"Dendrogram ({method})")
        plt.close(fig)

        return labels, fig

    print("[INFO] Using AgglomerativeClustering")

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

    labels = model.fit_predict(X)
    return labels, None