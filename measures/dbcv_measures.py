import numpy as np

from auxiliaries.decorators import cache_score, timed_named
from measures import base_measure
from measures.auxiliaries.dbcv import dbcv, dbcv_dist_matrix, clusterwise


class DBCV(base_measure.BaseMeasure):
    def __init__(self):
        super().__init__()
        self.name = "DBCV"
        self.worst_value = -1
        self.best_value = 1
        self.kwargs = {}
        self.needs_quadratic = False

    def score(self, data, labels):
        return self.ensure_finite(rebuild_dbcv(data, labels))

    def score_norm(self, data, labels):
        return self.score(data, labels)

    def score_distance_function(self, dists: np.ndarray, labels: np.ndarray, dim: int = 2) -> float:
        return self.ensure_finite(rebuild_dbcv_dist_matrix(dists, labels, dim))

    def score_clusters(self, dists: np.ndarray, labels: np.ndarray, dim: int = 2):
        ret = rebuild_dbcv_clusterwise(dists, labels, dim)
        for res in ret:
            ret[res] = self.ensure_finite(ret[res])
        return ret
