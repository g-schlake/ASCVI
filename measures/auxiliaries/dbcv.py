import warnings
from copy import copy

import numpy as np
from sklearn.metrics import pairwise_distances


def dbcv(data, partition, outlier_cluster=-1):
    partition = copy(partition)
    clusters = np.unique(partition)
    dist = np.power(pairwise_distances(data), 2)
    for cluster in clusters:
        if np.count_nonzero(partition == cluster) == 1:
            partition[partition == cluster] = outlier_cluster
            clusters[clusters == cluster] = outlier_cluster
    clusters = clusters[clusters != outlier_cluster]
    if len(clusters) <= 1:
        return 0
    data = data[partition != outlier_cluster, :]
    dist = dist[partition != outlier_cluster, :][:, partition != outlier_cluster]
    poriginal = partition
    partition = partition[partition != outlier_cluster]
    nclusters = len(clusters)
    nobjects, nfeatures = np.shape(data)
    d_ucore_cl = np.zeros((nobjects))
    compcl = np.zeros((nclusters))
    int_edges = [[]] * nclusters
    int_node_data = [[]] * nclusters
    for i in range(len(clusters)):
        objcl = np.where(partition == clusters[i])[0]
        nuobjcl = len(objcl)
        d_ucore_cl[objcl], mr = matrix_mutual_reachability_distance(nuobjcl, dist[objcl, :][:, objcl], nfeatures)
        G = {"no_vertices": nuobjcl, "MST_edges": np.zeros((nuobjcl - 1, 3)),
             "MST_degrees": np.zeros((nuobjcl), dtype=int), "MST_parent": np.zeros((nuobjcl), dtype=int)}
        Edges, Degrees = MST_Edges(G, 0, mr)
        int_node = np.where(Degrees != 1)[0]
        int_edg1 = np.where(np.in1d(Edges[:, 0], int_node))[0]
        int_edg2 = np.where(np.in1d(Edges[:, 1], int_node))[0]
        int_edges[i] = np.intersect1d(int_edg1, int_edg2)
        if len(int_edges[i]) > 0:
            compcl[i] = np.max(Edges[int_edges[i], 2])
        else:
            compcl[i] = np.max(Edges[:, 2])
        int_node_data[i] = objcl[int_node]
        if len(int_node_data[i]) == 0:
            int_node_data[i] = objcl
    core_dists = np.repeat(np.array(d_ucore_cl).reshape(-1, 1), nobjects, axis=1)
    core_dists_i = core_dists[:, :, np.newaxis]
    core_dists_j = core_dists.T[:, :, np.newaxis]
    pairwise_distance = dist[:, :, np.newaxis]
    mutual_reachability_distance_matrix = np.concatenate([core_dists_i, core_dists_j, pairwise_distance], axis=-1)
    sep_point = np.max(mutual_reachability_distance_matrix, axis=-1)
    valid = 0
    sepcl = np.full((nclusters), np.inf)
    for i in range(nclusters):
        sep = np.full((nclusters), np.inf)
        for j in range(nclusters):
            if i == j:
                continue
            sep[j] = np.min(sep_point[int_node_data[i], :][:, int_node_data[j]])
        sepcl[i] = np.min(sep)
        dbcvcl = (sepcl[i] - compcl[i]) / np.max([compcl[i], sepcl[i]])
        valid = valid + (dbcvcl * np.sum(partition == clusters[i]))
    valid = valid / len(poriginal)
    return valid


def matrix_mutual_reachability_distance(MinPts, G_edges_weights, d):
    G_edges_weights = G_edges_weights.copy()
    No = G_edges_weights.shape[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        K_NN_Dist = np.power(G_edges_weights, -1 * d)
    K_NN_Dist[
        K_NN_Dist == np.inf] = 0
    d_ucore = sum(K_NN_Dist)
    d_ucore = d_ucore / (No - 1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        d_ucore = np.power((1 / d_ucore), 1 / d)
    d_ucore[d_ucore == np.inf] = 0
    for i in range(No):
        for j in range(MinPts):
            G_edges_weights[i, j] = np.max([d_ucore[i], d_ucore[j], G_edges_weights[i, j]])
            G_edges_weights[j, i] = G_edges_weights[i, j]
    return d_ucore, G_edges_weights


def MST_Edges(G, start, G_edges_weights):
    intree = np.zeros((G["no_vertices"]), dtype=int)
    d = np.ndarray((G["no_vertices"]))
    for i in range(G["no_vertices"]):
        d[i] = np.inf
        G["MST_parent"][i] = i
    d[start] = 0
    v = start
    counter = 0
    while counter < G["no_vertices"] - 1:
        intree[v] = 1
        dist = np.inf
        for w in range(G["no_vertices"]):
            if w != v and intree[w] == 0:
                weight = G_edges_weights[v, w]
                if (d[w] > weight):
                    d[w] = weight
                    G["MST_parent"][w] = v
                if dist > d[w]:
                    dist = d[w]
                    next_v = w
        G["MST_edges"][counter, 0] = G["MST_parent"][next_v]
        G["MST_edges"][counter, 1] = next_v
        G["MST_edges"][counter, 2] = G_edges_weights[G["MST_parent"][next_v], next_v]
        G["MST_degrees"][G["MST_parent"][next_v]] = G["MST_degrees"][G["MST_parent"][next_v]] + 1
        G["MST_degrees"][next_v] += 1
        v = next_v
        counter += 1
    return G["MST_edges"], G["MST_degrees"]


def dbcv_dist_matrix(dist, partition, n_features,
                     outlier_cluster=-1, own_mst_implementation=True):
    clusters = np.unique(partition)
    # This information is rather hidden in the Paper
    for cluster in clusters:
        if np.count_nonzero(partition == cluster) == 1:
            partition[partition == cluster] = outlier_cluster
            clusters[clusters == cluster] = outlier_cluster
    clusters = clusters[clusters != outlier_cluster]
    if len(clusters) <= 1:
        return 0
    # Here, some equal points have a nonzero distance in the MatLab/Octave implementation
    dist = dist[partition != outlier_cluster, :][:, partition != outlier_cluster]
    poriginal = partition
    partition = partition[partition != outlier_cluster]
    nclusters = len(clusters)
    nobjects, nfeatures = dist.shape[0], n_features
    d_ucore_cl = np.zeros((nobjects))
    compcl = np.zeros((nclusters))
    int_edges = [[]] * nclusters
    int_node_data = [[]] * nclusters
    for i in range(len(clusters)):
        objcl = np.where(partition == clusters[i])[0]
        nuobjcl = len(objcl)
        d_ucore_cl[objcl], mr = matrix_mutual_reachability_distance(nuobjcl, dist[objcl, :][:, objcl], nfeatures)
        G = {"no_vertices": nuobjcl, "MST_edges": np.zeros((nuobjcl - 1, 3)),
             "MST_degrees": np.zeros((nuobjcl), dtype=int), "MST_parent": np.zeros((nuobjcl), dtype=int)}
        Edges, Degrees = MST_Edges(G, 0, mr)
        int_node = np.where(Degrees != 1)[0]
        int_edg1 = np.where(np.in1d(Edges[:, 0], int_node))[0]
        int_edg2 = np.where(np.in1d(Edges[:, 1], int_node))[0]
        int_edges[i] = np.intersect1d(int_edg1, int_edg2)
        if len(int_edges[i]) > 0:
            compcl[i] = np.max(Edges[int_edges[i], 2])
        else:
            compcl[i] = np.max(Edges[:, 2])
        int_node_data[i] = objcl[int_node]
        if len(int_node_data[i]) == 0:
            int_node_data[i] = objcl
    core_dists = np.repeat(np.array(d_ucore_cl).reshape(-1, 1), nobjects, axis=1)
    core_dists_i = core_dists[:, :, np.newaxis]
    core_dists_j = core_dists.T[:, :, np.newaxis]
    pairwise_distance = dist[:, :, np.newaxis]
    mutual_reachability_distance_matrix = np.concatenate([core_dists_i, core_dists_j, pairwise_distance], axis=-1)
    sep_point = np.max(mutual_reachability_distance_matrix, axis=-1)
    valid = 0
    sepcl = np.full((nclusters), np.Inf)
    for i in range(nclusters):
        sep = np.full((nclusters), np.Inf)
        for j in range(nclusters):
            if i == j:
                continue
            sep[j] = np.min(sep_point[int_node_data[i], :][:, int_node_data[j]])
        sepcl[i] = np.min(sep)
        dbcvcl = (sepcl[i] - compcl[i]) / np.max([compcl[i], sepcl[i]])
        valid = valid + (dbcvcl * np.sum(partition == clusters[i]))
    valid = valid / len(poriginal)
    return valid


def dbcv_clusterwise(dist, partition, n_features, outlier_cluster=-1):
    clusters = np.unique(partition)
    for cluster in clusters:
        if np.count_nonzero(partition == cluster) == 1:
            partition[partition == cluster] = outlier_cluster
            clusters[clusters == cluster] = outlier_cluster
    clusters = clusters[clusters != outlier_cluster]
    if len(clusters) <= 1:
        return 0
    dist = dist[partition != outlier_cluster, :][:, partition != outlier_cluster]
    partition = partition[partition != outlier_cluster]
    nclusters = len(clusters)
    nobjects, nfeatures = dist.shape[0], n_features
    d_ucore_cl = np.zeros((nobjects))
    compcl = {}
    int_edges = {}
    int_node_data = {}
    for cluster in clusters:
        objcl = np.where(partition == cluster)[0]
        nuobjcl = len(objcl)
        d_ucore_cl[objcl], mr = matrix_mutual_reachability_distance(nuobjcl, dist[objcl, :][:, objcl], nfeatures)
        G = {"no_vertices": nuobjcl, "MST_edges": np.zeros((nuobjcl - 1, 3)),
             "MST_degrees": np.zeros((nuobjcl), dtype=int), "MST_parent": np.zeros((nuobjcl), dtype=int)}
        Edges, Degrees = MST_Edges(G, 0, mr)
        int_node = np.where(Degrees != 1)[0]
        int_edg1 = np.where(np.in1d(Edges[:, 0], int_node))[0]
        int_edg2 = np.where(np.in1d(Edges[:, 1], int_node))[0]
        int_edges[cluster] = np.intersect1d(int_edg1, int_edg2)
        if len(int_edges[cluster]) > 0:
            compcl[cluster] = np.max(Edges[int_edges[cluster], 2])
        else:
            compcl[cluster] = np.max(Edges[:, 2])
        int_node_data[cluster] = objcl[int_node]
        if len(int_node_data[cluster]) == 0:
            int_node_data[cluster] = objcl
    core_dists = np.repeat(np.array(d_ucore_cl).reshape(-1, 1), nobjects, axis=1)
    core_dists_i = core_dists[:, :, np.newaxis]
    core_dists_j = core_dists.T[:, :, np.newaxis]
    pairwise_distance = dist[:, :, np.newaxis]
    mutual_reachability_distance_matrix = np.concatenate([core_dists_i, core_dists_j, pairwise_distance], axis=-1)
    sep_point = np.max(mutual_reachability_distance_matrix, axis=-1)
    sepcl = {}
    ret = {}
    for cluster_i in clusters:
        sep = {}
        for cluster_j in clusters:
            if cluster_i == cluster_j:
                continue
            sep[cluster_j] = np.min(sep_point[int_node_data[cluster_i], :][:, int_node_data[cluster_j]])
        sepcl[cluster_i] = np.min(list(sep.values()))
        ret[cluster_i] = (sepcl[cluster_i] - compcl[cluster_i]) / np.max([compcl[cluster_i], sepcl[cluster_i]])
    return ret
