import matplotlib.pyplot as plt
import numpy as np

def plot_clusters(data, labels, centroids=None, title="Clusters"):
    fig = plt.Figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    unique_labels = np.unique(labels)

    for lbl in unique_labels:
        pts = data[labels == lbl]
        ax.scatter(pts[:, 0], pts[:, 1], s=40, label=f"Cluster {lbl}")

    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], c="black", s=120, marker="X", label="Centroids")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)

    return fig
