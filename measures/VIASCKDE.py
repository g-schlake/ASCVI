import os.path

import numpy as np
from scipy.spatial import KDTree
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KernelDensity
from measures import base_measure


# Adaption of original authors code at https://github.com/senolali/VIASCKDE
class VIASCKDE(base_measure.BaseMeasure):
    """
     Ali Şenol
     VIASCKDE Index: A Novel Internal Cluster Validity Index for Arbitrary-Shaped Clusters Based on the Kernel Density Estimation
     Computational Intelligence and Neuroscience, vol. 2022, Article ID 4059302, 20 pages, 2022.
     https://doi.org/10.1155/2022/4059302"
    """

    def __init__(self):
        super().__init__()
        self.name = "VIASCKDE"
        self.worst_value = -1
        self.best_value = 1
        self.function = viasckde
        self.function_norm = viasckde
        self.kwargs = {}
        self.needs_quadratic = False


class VIASCKDE_cw(base_measure.BaseMeasure):
    """
    This implementation fits a KDE for each cluster instead of one KDE for the complete dataset.
     Ali Şenol
     VIASCKDE Index: A Novel Internal Cluster Validity Index for Arbitrary-Shaped Clusters Based on the Kernel Density Estimation
     Computational Intelligence and Neuroscience, vol. 2022, Article ID 4059302, 20 pages, 2022.
     https://doi.org/10.1155/2022/4059302"
    """

    def __init__(self):
        super().__init__()
        self.name = "VIASCKDE"
        self.worst_value = -1
        self.function = viasckde_cw
        self.kwargs = {}
        self.needs_quadratic = False


def closest_node(n, v):
    kdtree = KDTree(v)
    d, i = kdtree.query(n)
    return d


def fit_kernel(X, krnl, b_width):
    if not b_width:
        grid = RandomizedSearchCV(KernelDensity(),
                                  {'bandwidth': np.linspace(0.1, 100.0, 30)},
                                  cv=4)
        grid.fit(X)
        b_width = grid.best_params_["bandwidth"]
    kde = KernelDensity(kernel=krnl, bandwidth=b_width).fit(X)
    return kde.score_samples(X)


def viasckde(X, labels, krnl='gaussian', b_width=None):
    num_k = np.unique(labels)
    iso = fit_kernel(X, krnl, b_width,
                     dir=os.path.join("cache", "KDE", (hash_fit("KDE", X, kernel=krnl, b_width=b_width))))
    ASC = np.array([])
    numC = np.array([])
    CoSeD = np.array([])
    viasc = 0
    if len(num_k) > 1:
        for i in num_k:
            data_of_cluster = X[labels == i]
            data_of_not_its = X[labels != i]
            isos = iso[labels == i]
            if max(isos) != min(isos):
                isos = (isos - min(isos)) / (max(isos) - min(isos))
            else:
                isos = np.ones_like(isos) / len(isos)
            for j in range(len(data_of_cluster)):  # for each data of cluster j
                row = np.delete(data_of_cluster, j, 0)  # exclude the data j
                XX = data_of_cluster[j]
                a = closest_node(XX, row)
                b = closest_node(XX, data_of_not_its)
                if b == a:
                    ASC = np.hstack((ASC, 0 * isos[j]))
                else:
                    ASC = np.hstack((ASC, ((b - a) / max(a, b)) * isos[j]))
            numC = np.hstack((numC, ASC.size))
            CoSeD = np.hstack((CoSeD, ASC.mean()))
        for k in range(len(numC)):
            viasc += numC[k] * CoSeD[k]
        viasc = viasc / sum(numC)
    else:
        viasc = float("nan")
    return viasc


def viasckde_cw(X, labels, krnl='gaussian', b_width=0.05):
    if not b_width:
        grid = RandomizedSearchCV(KernelDensity(),
                                  {'bandwidth': np.linspace(0.1, 100.0, 30)},
                                  cv=4)
        grid.fit(X)
        b_width = grid.best_params_["bandwidth"]
    num_k = np.unique(labels)
    ASC = np.array([])
    numC = np.array([])
    CoSeD = np.array([])
    viasc = 0
    if len(num_k) > 1:
        for i in num_k:
            data_of_cluster = X[labels == i]
            data_of_not_its = X[labels != i]
            if len(labels == i) > 1:
                kde = KernelDensity(kernel=krnl, bandwidth=b_width).fit(data_of_cluster)
                isos = kde.score_samples(data_of_cluster)
                if max(isos) != min(isos):
                    isos = (isos - min(isos)) / (max(isos) - min(isos))
                else:
                    isos = np.ones_like(isos) / len(isos)
            else:
                isos = np.array([1])
            for j in range(len(data_of_cluster)):  # for each data of cluster j
                row = np.delete(data_of_cluster, j, 0)  # exclude the data j
                XX = data_of_cluster[j]
                a = closest_node(XX, row)
                b = closest_node(XX, data_of_not_its)
                if b == a:
                    ASC = np.hstack((ASC, 0 * isos[j]))
                else:
                    ASC = np.hstack((ASC, ((b - a) / max(a, b)) * isos[j]))
            numC = np.hstack((numC, ASC.size))
            CoSeD = np.hstack((CoSeD, ASC.mean()))
        for k in range(len(numC)):
            viasc += numC[k] * CoSeD[k]
        viasc = viasc / sum(numC)
    else:
        viasc = float("nan")
    return viasc
