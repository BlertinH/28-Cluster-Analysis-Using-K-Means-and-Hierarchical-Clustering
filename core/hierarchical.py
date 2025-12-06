from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

def hierarchical_clustering(data, num_clusters):
    Z = linkage(data, method="ward", metric="euclidean")

    fig = plt.Figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    dendrogram(Z, ax=ax)

    labels = fcluster(Z, num_clusters, criterion="maxclust")
    return labels, fig
