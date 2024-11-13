import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import KDTree, NearestNeighbors

from measures import base_measure


class DCV_Index(base_measure.BaseMeasure):
    """
    Xie, Jiang, et al.
    "A new internal index based on density core for clustering validation."
    Information Sciences 506 (2020): 346-365.
    https://doi.org/10.1016/j.ins.2019.08.029
    """

    def __init__(self):
        super().__init__()
        self.name = "DCVI"
        self.worst_value = np.inf
        self.best_value = 0
        self.normalization_params = (1.680808, 1.374437)
        self.function = dcvi
        self.kwargs = {}
        self.needs_quadratic = True
        self.less_is_better = True
        self.function_clusters = dcvi_clusters


def dcvi_clusters(dists, labels):
    clusters = [x for x in np.unique(labels) if x != -1]
    ret = {}
    for cluster in clusters:
        # compactness
        points = dists[labels == cluster, :][:, labels == cluster]
        mst = minimum_spanning_tree(points).toarray()
        com = np.max(mst)
        # separation
        points = dists[labels == cluster, :][:, labels != cluster]
        sep = np.min(points)
        ret[cluster] = com / sep
    return ret


def dcvi(dists, labels):
    clusters = [x for x in np.unique(labels) if x != -1]
    dcvi = 0
    for cluster in clusters:
        # compactness
        points = dists[labels == cluster, :][:, labels == cluster]
        mst = minimum_spanning_tree(points).toarray()
        com = np.max(mst)
        # separation
        points = dists[labels == cluster, :][:, labels != cluster]
        sep = np.min(points)
        if sep != 0:
            dcvi += com / sep
    return dcvi / len(clusters)
