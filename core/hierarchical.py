from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

def hierarchical_clustering(data,metric,num_clusters):
    if metric =="cityblock":
        method ="complete"
        metric_for_linkage="cityblock"
    else:
        method = "ward"
        metric_for_linkage = "euclidean"

    Z = linkage(data, method=method, metric=metric_for_linkage)

    fig = plt.Figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    dendrogram(Z, ax=ax)

    ax.set_title("Hierarchical Dendogram(Ward)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Distance")

    labels = fcluster(Z, num_clusters, criterion="maxclust")
    return labels, fig
