import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import numpy as np

def plot_voronoi_mid(data, centroids, labels=None, title="Voronoi Regions"):
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    if labels is not None:
        unique_clusters = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
    else:
        colors = None

    try:
        vor = Voronoi(centroids)
    except Exception as e:
        print("Voronoi error:", e)
        return

    for ridge in vor.ridge_vertices:
        if -1 not in ridge:     
            v0, v1 = vor.vertices[ridge]
            ax.plot([v0[0], v1[0]], [v0[1], v1[1]], "k--", alpha=0.6)

    ax.scatter(
        data[:, 0], data[:, 1],
        c="gray", s=35, alpha=0.6, label="Data"
    )

    ax.scatter(
        centroids[:, 0], centroids[:, 1],
        c="red", s=150, marker="X",
        edgecolor="black", linewidth=1.2,
        label="Centroids"
    )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()
