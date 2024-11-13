import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, floyd_warshall
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from measures import base_measure


class CVDD(base_measure.BaseMeasure):
    """
    Hu, Lianyu, and Caiming Zhong.
    An internal validity index based on density-involved distance.
    IEEE Access 7 (2019): 40038-40051.
    https://doi.org/10.1109/ACCESS.2019.2906949
    """

    def __init__(self):
        super().__init__()
        self.name = "CVDD"
        self.worst_value = 0
        self.best_value = np.inf
        self.normalization_params = (0.134506, 0.261888)
        self.function = cvdd
        self.function_norm = ValueError
        self.kwargs = {"k": 7}
        self.needs_quadratic = False
        self.function_clusters = function_cluster

    def score_distance_function(self, data, labels, k=7, **kwargs):
        def cvdd(distances, labels, k=7):
            unique_labels = np.unique(labels)
            # CVDD = Null
            # den = Eqn1w
            nn_distances = np.sort(distances, axis=1)[:, :k]
            dens = np.fromiter((np.mean(x) for x in nn_distances), dtype=float)
            # fden = Eqn2
            if 0 in dens:
                min_dens = np.min(dens[dens != 0]) / 5
                dens = np.where(dens == 0.0, min_dens, dens)
            max_den = np.max(dens)
            fden = np.fromiter((den / max_den for den in dens), dtype=float)
            # frel = Eqn5
            rels = np.outer(1 / dens, dens)
            fRels = 1 - np.exp(-(rels + np.transpose(rels) - 2))
            # DD = eq9
            drDs = distances + (fRels * (dens[:, np.newaxis] + dens))
            pDs = compute_path_distances(distances)
            conDs = compute_path_distances(drDs)
            DDs = conDs * np.sqrt(fden[:, np.newaxis] * fden)
            seps = {}
            coms = {}
            for i in unique_labels:
                # sep[i] = Ci Eqn11
                indices_i = np.where(labels == i)
                min_sep = np.inf
                DDs_i = DDs[indices_i, :][0, :, :]
                for j in unique_labels:
                    if i == j: continue
                    indices_j = np.where(labels == j)
                    sep = np.min(DDs_i[:, indices_j][:, 0, :])
                    if sep < min_sep:
                        min_sep = sep
                seps[i] = min_sep
                # com[i] = Ci Eqn 12
                pDs_i = pDs[indices_i, :][0, :, :][:, indices_i][:, 0, :]
                std = np.std(pDs_i)
                mean = np.mean(pDs_i)
                coms[i] = (1 / len(indices_i)) * std * mean
            # CVDD = Eqn 13
            sep_sum = np.sum([x for x in seps.values()])
            com_sum = np.sum([x for x in coms.values()])
            cvdd = sep_sum / com_sum
            return cvdd

        data, labels, share = self.clean_outliers(data, labels)
        if not self.check_valid(labels):
            return self.worst_value
        res = cvdd(data, labels, k)
        ret = res * share
        ret = self.ensure_finite(ret)
        return ret


def function_cluster(distances, labels, k=7):
    unique_labels = np.unique(labels)
    # CVDD = Null
    # den = Eqn1w
    nn_distances = np.sort(distances, axis=1)[:, :k]
    dens = np.fromiter((np.mean(x) for x in nn_distances), dtype=float)
    # fden = Eqn2
    if 0 in dens:
        min_dens = np.min(dens[dens != 0]) / 5
        dens = np.where(dens == 0.0, min_dens, dens)
    max_den = np.max(dens)
    fden = np.fromiter((den / max_den for den in dens), dtype=float)
    # frel = Eqn5
    rels = np.outer(1 / dens, dens)
    fRels = 1 - np.exp(-(rels + np.transpose(rels) - 2))
    # DD = eq9
    drDs = distances + (fRels * (dens[:, np.newaxis] + dens))
    pDs = compute_path_distances(distances)
    conDs = compute_path_distances(drDs)
    DDs = conDs * np.sqrt(fden[:, np.newaxis] * fden)
    seps = {}
    coms = {}
    ret = {}
    for i in unique_labels:
        # sep[i] = Ci Eqn11
        indices_i = np.where(labels == i)
        min_sep = np.inf
        DDs_i = DDs[indices_i, :][0, :, :]
        for j in unique_labels:
            if i == j: continue
            indices_j = np.where(labels == j)
            sep = np.min(DDs_i[:, indices_j][:, 0, :])
            if sep < min_sep:
                min_sep = sep
        seps[i] = min_sep
        # com[i] = Ci Eqn 12
        pDs_i = pDs[indices_i, :][0, :, :][:, indices_i][:, 0, :]
        std = np.std(pDs_i)
        mean = np.mean(pDs_i)
        coms[i] = (1 / len(indices_i)) * std * mean
        ret[i] = seps[i] / coms[i]
    return ret


def cvdd(X, labels, k=7):
    unique_labels = np.unique(labels)
    # CVDD = Null
    # den = Eqn1w
    distances = pairwise_distances(X, n_jobs=1)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    nn_distances, indices = nbrs.kneighbors(X)
    dens = np.fromiter((np.mean(x) for x in nn_distances), dtype=float)
    # fden = Eqn2
    if 0 in dens:
        min_dens = np.min(dens[dens != 0]) / 5
        dens = np.where(dens == 0.0, min_dens, dens)
    max_den = np.max(dens)
    fden = np.fromiter((den / max_den for den in dens), dtype=float)
    # frel = Eqn5
    rels = np.outer(1 / dens, dens)
    fRels = 1 - np.exp(-(rels + np.transpose(rels) - 2))
    # DD = eq9
    drDs = distances + (fRels * (dens[:, np.newaxis] + dens))
    pDs = compute_path_distances(distances)
    conDs = compute_path_distances(drDs)
    DDs = conDs * np.sqrt(fden[:, np.newaxis] * fden)
    seps = {}
    coms = {}
    for i in unique_labels:
        # sep[i] = Ci Eqn11
        indices_i = np.where(labels == i)
        min_sep = np.inf
        DDs_i = DDs[indices_i, :][0, :, :]
        for j in unique_labels:
            if i == j: continue
            indices_j = np.where(labels == j)
            sep = np.min(DDs_i[:, indices_j][:, 0, :])
            if sep < min_sep:
                min_sep = sep
        seps[i] = min_sep
        # com[i] = Ci Eqn 12
        pDs_i = pDs[indices_i, :][0, :, :][:, indices_i][:, 0, :]
        std = np.std(pDs_i)
        mean = np.mean(pDs_i)
        coms[i] = (1 / len(indices_i)) * std * mean
    # CVDD = Eqn 13
    sep_sum = np.sum([x for x in seps.values()])
    com_sum = np.sum([x for x in coms.values()])
    return sep_sum / com_sum


def compute_path_distances(matrix):
    mst = minimum_spanning_tree(matrix)
    res_a = floyd_warshall(mst, directed=False)
    return res_a
