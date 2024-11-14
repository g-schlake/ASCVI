import numpy as np

from measures import base_measure
from measures.auxiliaries.dbcv import dbcv, dbcv_dist_matrix, dbcv_clusterwise


class DBCV(base_measure.BaseMeasure):
    """
    Moulavi, Davoud, et al.
    "Density-based clustering validation."
    Proceedings of the 2014 SIAM international conference on data mining. Society for Industrial and Applied Mathematics
    2014.
    """
    def __init__(self):
        super().__init__()
        self.name = "DBCV"
        self.worst_value = -1
        self.best_value = 1
        self.kwargs = {}
        self.needs_quadratic = False

    def score(self, data, labels):
        return self.ensure_finite(dbcv(data, labels))

    def score_norm(self, data, labels):
        return self.score(data, labels)

    def score_distance_function(self, dists: np.ndarray, labels: np.ndarray, dim: int = 2) -> float:
        return self.ensure_finite(dbcv_dist_matrix(dists, labels, dim))

    def score_clusters(self, dists: np.ndarray, labels: np.ndarray, dim: int = 2):
        ret = dbcv_clusterwise(dists, labels, dim)
        for res in ret:
            ret[res] = self.ensure_finite(ret[res])
        return ret
