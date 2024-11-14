
import numpy as np
from s_dbw import S_Dbw
from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples
from measures import base_measure


def function_clusters_swc(data: np.ndarray, labels: np.ndarray, **kwargs):
    sample_sils = silhouette_samples(data, labels, **kwargs)
    ret = {}
    for cluster in np.unique(labels):
        ret[cluster] = np.mean(sample_sils[labels == cluster])
    return ret


class Silhouette_Coefficient(base_measure.BaseMeasure):
    """
    Rousseeuw, Peter J.
    "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis."
    Journal of computational and applied mathematics 20 (1987): 53-65.
    """
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
    """
    Caliński, Tadeusz, and Jerzy Harabasz.
    "A dendrite method for cluster analysis."
     Communications in Statistics-theory and Methods 3.1 (1974): 1-27.
    """
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


class SDBW(base_measure.BaseMeasure):
    """
    Halkidi, Maria, and Michalis Vazirgiannis.
    "Clustering validity assessment: Finding the optimal partitioning of a data set."
    Proceedings 2001 IEEE international conference on data mining. IEEE, 2001
    """
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
