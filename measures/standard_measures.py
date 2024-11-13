from copy import deepcopy

import numpy as np
from scipy.spatial import QhullError
from s_dbw import S_Dbw
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_mutual_info_score, silhouette_samples
from auxiliaries.dataset_fetcher import clean_labels
from measures import base_measure
from measures.auxiliaries.CDBW import cdbw


def function_clusters_swc(data: np.ndarray, labels: np.ndarray, **kwargs):
    sample_sils = silhouette_samples(data, labels, **kwargs)
    ret = {}
    for cluster in np.unique(labels):
        ret[cluster] = np.mean(sample_sils[labels == cluster])
    return ret


class Silhouette_Coefficient(base_measure.BaseMeasure):
    def __init__(self):
        super().__init__()
        self.name = "SWC"
        self.worst_value = -1
        self.best_value = 1
        self.function = silhouette_score
        self.function_norm = silhouette_score
        self.kwargs = {"metric": "precomputed"}
        self.needs_quadratic = True
        self.function_clusters = function_clusters_swc


class VRC(base_measure.BaseMeasure):
    def __init__(self):
        super().__init__()
        self.name = "VRC"
        self.worst_value = -1 * np.inf
        self.best_value = 0
        self.normalization_params = (1490.943623, 1646.226119)
        self.function = calinski_harabasz_score
        self.function_norm = ValueError
        self.function_clusters = ValueError
        self.kwargs = {}
        self.needs_quadratic = False


class CDBW(base_measure.BaseMeasure):
    def __init__(self):
        super().__init__()
        self.name = "CDbw"
        self.worst_value = -1 * np.inf
        self.best_value = 0
        self.function_norm = ValueError
        self.function_clusters = ValueError
        self.function = cdbw
        self.kwargs = {}
        self.needs_quadratic = False

    def score(self, data, labels):
        if not self.check_valid(labels):
            return self.worst_value
        if self.needs_quadratic:
            data = self.ensure_distance_matrix(data)
        data, labels, share = self.clean_outliers(data, labels)
        kwargs = deepcopy(self.kwargs)
        kwargs["X"] = data
        kwargs["labels"] = labels
        try:
            res = self.function(**kwargs) * share
        except QhullError:
            res = self.worst_value
        return res


class SDBW(base_measure.BaseMeasure):
    def __init__(self):
        super().__init__()
        self.name = "S-Dbw"
        self.worst_value = np.inf
        self.best_value = 0
        self.normalization_params = (0.380854, 0.180388)
        self.function_norm = ValueError
        self.function = S_Dbw
        self.kwargs = {}
        self.needs_quadratic = False
        self.less_is_better = True
