import matplotlib.pyplot as plt
import numpy as np

def plot_clusters(data, labels, centroids=None, title="Cluster Visualization"):
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = plt.cm.Set2(np.linspace(0, 1, len(np.unique(labels))))

    for i, lbl in enumerate(np.unique(labels)):
        pts = data[labels == lbl]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=70, alpha=0.9,
            label=f"Cluster {lbl}",
            color=colors[i]
        )

    if centroids is not None:
        ax.scatter(
            centroids[:, 0], centroids[:, 1],
            c="black", s=220, marker="X",
            edgecolors="white", linewidths=2,
            label="Centroids"
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    return fig
