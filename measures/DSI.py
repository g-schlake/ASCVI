# Distance-based Separability Index
from copy import deepcopy

import numpy as np
from scipy.spatial import distance
from scipy.stats import ks_2samp
from sklearn.metrics import pairwise_distances

from measures import base_measure


# Adapted from original authors implementation at https://github.com/ShuyueG/CVI_using_DSI
class DSI(base_measure.BaseMeasure):
    """
    Guan, Shuyue, and Murray Loew.
    A Distance-based Separability Measure for Internal Cluster Validation.
    INTERNATIONAL JOURNAL ON ARTIFICIAL INTELLIGENCE TOOLS_ 31.07 (2022): 2260005.
    https://doi.org/10.1142/S0218213022600053
    """

    def __init__(self):
        super().__init__()
        self.name = "DSI"
        self.worst_value = 0
        self.best_value = 1
        self.function = dsi_function
        self.function_norm = ValueError
        self.function_clusters = dsi_clusters
        self.kwargs = {}
        self.needs_quadratic = False

    def score_distance_function(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        data = data.copy()
        data, labels, share = self.clean_outliers(data, labels)
        if not self.check_valid(labels):
            return self.worst_value
        kwargs_out = deepcopy(self.kwargs)
        for kw in kwargs:
            kwargs_out[kw] = kwargs[kw]
        kwargs_out["dists"] = data
        kwargs_out["labels"] = labels
        # start=time.time()
        # print(f"Start {self.name}")
        res = dsi_dist(**kwargs_out)
        # print(f"Finished {self.name} in {time.time()-start:.2f}")
        ret = res * share
        ret = self.ensure_finite(ret)
        return ret


def dsi_function(X, labels):
    return dsi_dist(pairwise_distances(X), labels)


def dsi_clusters(X, labels):  # KS test on ICD and BCD
    dists = pairwise_distances(X)
    classes = np.unique(labels)
    ret = {}
    for c in classes:
        dist_pos = dists[labels == c, :][:, labels == c]
        dist_pos = dist_pos[np.triu_indices_from(dist_pos, 1)]
        distbtw = np.reshape(dists[labels == c, :][:, labels != c], (-1))
        if dist_pos.size == 0:
            continue
        D, _ = ks_2samp(dist_pos, distbtw)  # KS test
        ret[c] = D
    return ret


def dsi_dist(dists, labels):
    classes = np.unique(labels)
    SUM = 0
    for c in classes:
        dist_pos = dists[labels == c, :][:, labels == c]
        dist_pos = dist_pos[np.triu_indices_from(dist_pos, 1)]
        distbtw = np.reshape(dists[labels == c, :][:, labels != c], (-1))
        if dist_pos.size == 0:
            continue
        D, _ = ks_2samp(dist_pos, distbtw)  # KS test
        SUM += D
    SUM = SUM / classes.shape[0]  # normed: b/c ks_2samp ranges [0,1]
    return SUM
