import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, QhullError
import numpy as np

def plot_voronoi(data, centroids, labels, title="Voronoi Decision Boundary"):
    fig = plt.Figure(figsize=(8, 7))
    ax = fig.add_subplot(111)

    colors = plt.cm.Set2(np.linspace(0, 1, len(np.unique(labels))))

    all_points = np.vstack([data, centroids])
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)

    range_x = x_max - x_min
    range_y = y_max - y_min

    min_pad = 0.5

    padding = max(range_x, range_y) * 0.35
    padding = max(padding, min_pad)

    xmin = x_min - padding
    xmax = x_max + padding
    ymin = y_min - padding
    ymax = y_max + padding

    boundary_points = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymin],
        [xmax, ymax],
    ])

    centroids_extended = np.vstack([centroids, boundary_points])

    try:
        vor = Voronoi(centroids_extended)

        for i in range(len(centroids)):
            region_index = vor.point_region[i]
            region = vor.regions[region_index]

            if -1 not in region and region:
                polygon = [vor.vertices[v] for v in region]
                ax.fill(*zip(*polygon), alpha=0.20, color=colors[i])

    except QhullError:
        print("[WARN] Voronoi failed to generate (collinear centroids).")

    ax.scatter(data[:, 0], data[:, 1], s=40, alpha=0.45, color="gray", label="Points")

    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="red",
        s=220,
        marker="X",
        edgecolors="black",
        linewidths=2,
        label="Centroids"
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    return fig