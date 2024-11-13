import time

import numpy as np
import scipy.spatial
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree


# @timed
def cdbw(X, labels):
    # Determine a good number of representatives (remember curse of dimensionality)
    dim = X.shape[1]
    num_representatives = 3 * dim  # Five was the minimum proposed by Halkidi et. al for two-dimensional data, ten the optimal value
    # Determine representatives per cluster
    cluster_reps = {}
    stdevs = {}
    trees = {}
    cluster_sizes = {}
    cluster_centers = {}
    clusters = [x for x in np.unique(labels) if x != -1]
    cluster_for_start = time.time()
    for cluster in clusters:
        metadata_start = time.time()
        points_in_cluster = X[labels == cluster]
        trees[cluster] = KDTree(points_in_cluster)
        cluster_centers[cluster] = np.mean(points_in_cluster, axis=0)
        cluster_sizes[cluster] = len(points_in_cluster)
        stdevs[cluster] = np.std(points_in_cluster)
        if cluster_sizes[cluster] <= num_representatives:
            cluster_reps[cluster] = X
            continue
        cluster_reps[cluster] = determine_reps(points_in_cluster, num_representatives)
    # Compute CD_bw
    inter_dens = 0
    sep = 0
    for cluster_i in clusters:
        densities = []
        dists = []
        for cluster_j in clusters:
            if cluster_i == cluster_j:
                continue
            RCRs = find_rcrs(cluster_reps[cluster_i], cluster_reps[cluster_j])
            stdev = (stdevs[cluster_i] + stdevs[cluster_j]) / 2
            densities.append(density(RCRs, stdev, trees[cluster_i], trees[cluster_j], cluster_sizes[cluster_i],
                                     cluster_sizes[cluster_j]))
            dists.append(dist(RCRs))
            # print(f"Ended Combination of Clusters {cluster_i} and {cluster_j} in {time.time()-cluster_j_start:.2f}")
        inter_dens += np.max(densities, initial=0)
        sep += np.min(dists, initial=0)
        # print(f"Ended Computation of Cluster {cluster_i} in {time.time()-cluster_i_start:.2f}")
    # print(f"Ended the loop with both clusters in {time.time()-cluster_i_for_start:.2f}")
    inter_dens /= len(clusters)
    sep = (1 / len(clusters) * sep) / 1 + inter_dens
    intra_dens = [
        dens_cl(clusters, trees, cluster_reps, s, cluster_centers, cluster_sizes, stdevs) / len(clusters) * np.std(X)
        for s in np.arange(0.1, 0.8, step=0.1)]
    compactness = np.sum(intra_dens) / len(intra_dens)
    intra_change = 0
    for i in range(1, len(intra_dens)):
        intra_change += abs(intra_dens[i - 1] - intra_dens[i])
    intra_change /= (len(intra_dens) - 1)
    cohesion = compactness / (1 + intra_change)
    SC = sep * compactness
    return cohesion * SC


def dens_cl(clusters, trees, cluster_reps, s, centers, sizes, stdevs):
    dens_cl = 0
    for cluster in clusters:
        for rep in cluster_reps[cluster]:
            rep_s = (rep + s * (centers[cluster] - rep)).reshape(1, -1)
            dens_cl += trees[cluster].query_radius(rep_s, r=stdevs[cluster], count_only=True)[0] / sizes[cluster]
    return dens_cl


def determine_reps(X, num_representatives):
    dists = pairwise_distances(X)
    num_points_in_cluster = len(X)
    cluster_mean = np.mean(X, axis=0)
    reps_of_cluster = []
    distance_to_mean = np.linalg.norm(X - cluster_mean, axis=1)
    max_distance_idx = np.argmax(distance_to_mean)
    reps_of_cluster.append(X[max_distance_idx])
    distance_to_reps = dists[:, max_distance_idx]
    while len(reps_of_cluster) < num_representatives:
        start_rep = time.time()
        new_rep = np.argmax(distance_to_reps)
        reps_of_cluster.append(X[new_rep])
        new_dists = dists[:, max_distance_idx]
        distance_to_reps = np.min(np.vstack((distance_to_reps, new_dists)), axis=0)
        pass
    return np.array(reps_of_cluster)


def find_rcrs(points_in_i, points_in_j):
    distances = pairwise_distances(points_in_i, points_in_j)
    nearest_i = distances.argmax(axis=1)
    nearest_j = distances.argmax(axis=0)
    rcrs = []
    for i in range(len(nearest_i)):
        j = nearest_i[i]
        if nearest_j[j] == i:
            rcrs.append((points_in_i[i], points_in_j[j]))
    return rcrs


def density(RCRs, stdev, tree_i, tree_j, cluster_size_i, cluster_size_j):
    Dens = 0
    for point_i, point_j in RCRs:
        u = ((point_i + point_j) / 2).reshape(1, -1)
        num_neighbors = tree_i.query_radius(u, r=stdev, count_only=True)[0] + \
                        tree_j.query_radius(u, r=stdev, count_only=True)[0]
        cardinality = num_neighbors / (cluster_size_i + cluster_size_j)
        dist = euclidean(point_i, point_j)
        Dens += (dist / (2 * stdev)) * cardinality
    return Dens / len(RCRs)


def dist(RCRs):
    Dist = 0
    for point_i, point_j in RCRs:
        Dist += euclidean(point_i, point_j)
    return Dist / len(RCRs)


