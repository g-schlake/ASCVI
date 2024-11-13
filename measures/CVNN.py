import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from auxiliaries.decorators import cache_score
from measures import base_measure


class CVNN(base_measure.BaseMeasure):
    """
    Liu, Yanchi, et al.
    "Understanding and enhancement of internal clustering validation measures."
    IEEE transactions on cybernetics 43.3 (2013): 982-994.
    https://doi.org/10.1109/TSMCB.2012.2220543
    """

    def __init__(self, k=None):
        super().__init__()
        self.name = "CVNN"
        self.worst_value = np.inf
        self.best_value = 0
        self.normalization_params = (75.397617, 165.993193)
        self.function = cvnn
        self.function_norm = ValueError
        self.kwargs = {"k": k}
        self.needs_quadratic = False
        self.less_is_better = True

    def score_distance_function(self, data, labels, k=7, **kwargs):
        data, labels, share = self.clean_outliers(data, labels)
        if not self.check_valid(labels):
            return self.worst_value
        # start=time.time()
        # print(f"Start {self.name}")
        res = cvnn_dist(data, labels, k)
        # print(f"Finished {self.name} in {time.time()-start:.2f}")
        ret = res * share
        ret = self.ensure_finite(ret)
        return ret


class CVNN_halkidi(base_measure.BaseMeasure):
    """
    Correction from
    Halkidi, Maria, Michalis Vazirgiannis, and Christian Hennig.
    "Method-independent indices for cluster validation and estimating the number of clusters."
    Handbook of cluster analysis. Chapman and Hall/CRC, 2015. 616-639.
    eBook ISBN 9780429185472
    """

    def __init__(self) -> object:
        super().__init__()
        self.name = "hal_CVNN"
        self.worst_value = np.inf
        self.best_value = 0
        self.normalization_params = (7.907539, 17.557362)
        self.function = cvnn_halkidi
        self.function_norm = ValueError
        self.kwargs = {"k": None}
        self.needs_quadratic = False
        self.less_is_better = True

    @cache_score
    def score_distance_function(self, data, labels, k=None, **kwargs):
        if not self.check_valid(labels):
            return self.worst_value
        data, labels, share = self.clean_outliers(data, labels)
        if not self.check_valid(labels):
            return self.worst_value
        # start=time.time()
        # print(f"Start {self.name}")
        res = cvnn_halkidi_dist(data, labels, k)
        # print(f"Finished {self.name} in {time.time()-start:.2f}")
        ret = res * share
        ret = self.ensure_finite(ret)
        return ret


def cvnn(X, labels, k: int = None):
    num_labels = len(np.unique(labels))
    num_points = X.shape[0]
    if not k:
        k = max(min(10, num_points - 1), min(100, int(num_points / (num_labels * 100))))
    k = min(k, (num_points - 1))
    return sep(X, labels, k) + com(X,
                                   labels)


def sep(X, labels, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    clusters = [x for x in np.unique(labels) if x != -1]
    sep = -1 * np.inf
    for cluster in clusters:
        neighbor_indices = nbrs.kneighbors(X[labels == cluster], return_distance=False)
        sep_cl = (1 / np.count_nonzero(labels == cluster)) * (
                    (labels[neighbor_indices] != cluster).astype(int).sum() / k)
        if sep_cl > sep:
            sep = sep_cl
    return sep


def com(X, labels):
    clusters = [x for x in np.unique(labels) if x != -1]
    com = 0
    for cluster in clusters:
        elems = np.count_nonzero(labels == cluster)
        if elems > 1:
            com += (2 / (elems * (elems - 1))) * pairwise_distances(X[labels == cluster]).sum()
            pass
    return com


def cvnn_dist(dists, labels, k: int = None):
    num_labels = len(np.unique(labels))
    num_points = dists.shape[0]
    if not k:
        k = max(min(10, num_points - 1), min(100, int(num_points / (num_labels * 100))))
    k = min(k, (num_points - 1))
    return sep_dist(dists, labels, k) + com_dist(dists,
                                                 labels)


def sep_dist(dists, labels, k):
    nbrs = np.argsort(dists, axis=1)
    clusters = [x for x in np.unique(labels) if x != -1]
    sep = -1 * np.inf
    for cluster in clusters:
        neighbor_indices = nbrs[labels == cluster, :k + 1]
        sep_cl = (1 / np.count_nonzero(labels == cluster)) * (
                    ((labels[neighbor_indices] != cluster).astype(int)).sum() / k)
        if sep_cl > sep:
            sep = sep_cl
    return sep


def com_dist(dists, labels):
    clusters = [x for x in np.unique(labels) if x != -1]
    com = 0
    for cluster in clusters:
        elems = np.count_nonzero(labels == cluster)
        if elems > 1:
            com += (2 / (elems * (elems - 1))) * dists[labels == cluster, :][:, labels == cluster].sum()
    return com


def cvnn_halkidi(X, labels, k=None, dists=None):
    num_labels = len(np.unique(labels))
    num_points = X.shape[0]
    if not k:
        k = max(min(10, num_points - 1), min(100, int(num_points / (num_labels * 100))))
    k = min(k, (num_points - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    neighbor_indices = nbrs.kneighbors(X, return_distance=False)
    neighbor_indices = neighbor_indices[:, 1:]
    X = pairwise_distances(X)
    comp = nji = 0
    sepj = []
    for i in np.unique(labels):
        nj = np.sum(labels == i)
        compj = np.sum(X[labels == i, :][:, labels == i])
        nji = nji + nj * (nj - 1)
        comp += compj
        share = np.sum(labels[neighbor_indices[labels == i, :]] != i) / (nj * k)
        sepj.append(share)
    sep = np.max(sepj)
    comp = comp / nji
    return sep + comp


def cvnn_halkidi_dist(dists, labels, k=None):
    num_labels = len(np.unique(labels))
    num_points = dists.shape[0]
    if not k:
        k = max(min(10, num_points - 1), min(100, int(num_points / (num_labels * 100))))
    k = min(k, (num_points - 1))
    neighbor_indices = np.argsort(dists, axis=1)[:, 1:k + 1]
    comp = nji = 0
    sepj = []
    for i in np.unique(labels):
        nj = np.sum(labels == i)
        compj = np.sum(dists[labels == i, :][:, labels == i])
        nji = nji + nj * (nj - 1)
        comp += compj
        share = np.sum(labels[neighbor_indices[labels == i, :]] != i) / (nj * k)
        sepj.append(share)
    sep = np.max(sepj)
    comp = comp / nji
    return sep + comp
