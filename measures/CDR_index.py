import warnings
from copy import deepcopy

import numpy as np
from sklearn.metrics import pairwise_distances

from measures import base_measure


class CDR_Index(base_measure.BaseMeasure):
    """
    This implementation corrects a mistake in eq.4, where the numerator has to be divided by the cluster size before being
    devided by the divider
    Rojas‐Thomas, Juan Carlos, and Matilde Santos.
    "New internal clustering validation measure for contiguous arbitrary‐shape clusters." _
    International Journal of Intelligent Systems_ 36.10 (2021): 5506-5529.
    https://doi.org/10.1002/int.22521
    """

    def __init__(self):
        super().__init__()
        self.name = "CDR"
        self.worst_value = np.inf
        self.best_value = 0
        self.normalization_params = (0.553459, 0.126152)
        self.function = cdr
        self.function_norm = ValueError
        self.kwargs = {"avg": True}
        self.needs_quadratic = False
        self.function_clusters = cdr_clusters
        self.less_is_better = True

    def score_distance_function(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        data = data.copy()
        data, labels, share = self.clean_outliers(data, labels)
        if not self.check_valid(labels):
            return self.worst_value
        kwargs_out = deepcopy(self.kwargs)
        for kw in kwargs:
            kwargs_out[kw] = kwargs[kw]
        kwargs_out["X"] = data
        kwargs_out["labels"] = labels
        kwargs_out["distance"] = "precomputed"
        # start=time.time()
        # print(f"Start {self.name}")
        res = self.function(**kwargs_out)
        # print(f"Finished {self.name} in {time.time()-start:.2f}")
        ret = res * share
        ret = self.ensure_finite(ret)
        return ret


class CDR_Index_not_averaged(base_measure.BaseMeasure):
    """
    This implementation is true to the paper and does not correct the mistake
    Rojas‐Thomas, Juan Carlos, and Matilde Santos.
    "New internal clustering validation measure for contiguous arbitrary‐shape clusters." _
    International Journal of Intelligent Systems_ 36.10 (2021): 5506-5529.
    https://doi.org/10.1002/int.22521
    """

    def __init__(self):
        super().__init__()
        self.name = "CDR (old)"
        self.worst_value = np.inf
        self.function = cdr
        self.kwargs = {"avg": False}
        self.needs_quadratic = False
        self.less_is_better = True
        self.function_clusters = cdr_clusters

    def score_distance_function(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        data, labels, share = self.clean_outliers(data, labels)
        if not self.check_valid(labels):
            return self.worst_value
        kwargs_out = deepcopy(self.kwargs)
        for kw in kwargs:
            kwargs_out[kw] = kwargs[kw]
        kwargs_out["X"] = data
        kwargs_out["labels"] = labels
        kwargs_out["distance"] = "precomputed"
        # start=time.time()
        # print(f"Start {self.name}")
        res = self.function(**kwargs_out)
        # print(f"Finished {self.name} in {time.time()-start:.2f}")
        ret = res * share
        ret = self.ensure_finite(ret)
        return ret


def cdr(X, labels, distance="euclidean", avg=True):
    unique_labels = np.unique(labels)
    if distance != "precomputed":
        distances = pairwise_distances(X, n_jobs=1, metric=distance)
    else:
        distances = X.copy()
    np.fill_diagonal(distances, np.inf)
    CDR = 0
    for cluster in unique_labels:
        if cluster == -1:
            continue
        cluster_points = np.where(labels == cluster)[0]
        cluster_size = len(cluster_points)
        uniformity = 0
        if cluster_size > 1:
            inner_cluster_distances = distances[cluster_points, :][:, cluster_points]
            local_densities = np.min(inner_cluster_distances, axis=0)
            avg_den = np.sum(local_densities) / cluster_size
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                numerator = np.sum(np.abs((local_densities - avg_den)))
                if avg:
                    numerator /= cluster_size
                uniformity = numerator / avg_den
        CDR += cluster_size * uniformity
    if len(labels) == 0:
        return np.inf
    return CDR / len(labels)


def cdr_clusters(X, labels, distance="precomputed", avg=True):
    unique_labels = np.unique(labels)
    if distance != "precomputed":
        distances = pairwise_distances(X, n_jobs=1, metric=distance)
    else:
        distances = X.copy()
    np.fill_diagonal(distances, np.inf)
    ret = {}
    for cluster in unique_labels:
        if cluster == -1:
            continue
        cluster_points = np.where(labels == cluster)[0]
        cluster_size = len(cluster_points)
        if cluster_size > 1:
            inner_cluster_distances = distances[cluster_points, :][:, cluster_points]
            local_densities = np.min(inner_cluster_distances, axis=0)
            avg_den = np.sum(local_densities) / cluster_size
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                numerator = np.sum(np.abs((local_densities - avg_den)))
                if avg:
                    numerator /= cluster_size
                ret[cluster] = numerator / avg_den
    return ret
