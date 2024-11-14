import multiprocessing
import warnings
from heapq import heappop, heappush
from itertools import count

import numpy as np
import scipy
from networkx import Graph
from scipy.sparse import csr_matrix

from measures import base_measure


"""
Parts of this code are derived from the NetworkX packaage unter the 3-clause BSD License:
Copyright (C) 2004-2024, NetworkX Developers
Aric Hagberg <hagberg@lanl.gov>
Dan Schult <dschult@colgate.edu>
Pieter Swart <swart@lanl.gov>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided
    with the distribution.

  * Neither the name of the NetworkX Developers nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

class IC_av(base_measure.BaseMeasure):
    """
    Computes the ICav according to the paper
    How many clusters: A validation index for arbitrary-shaped clusters.
    Baya, Ariel E., and Pablo M. Granitto.
     IEEE/ACM Transactions on Computational Biology and Bioinformatics 10.2 (2013): 401-414.
     https://doi.org/10.1109/TCBB.2013.32
    """

    def __init__(self):
        super().__init__()
        self.name = "IC-av"
        self.worst_value = np.inf
        self.best_value = 0
        self.normalization_params = (377.018371, 1818.392474)
        self.function = ic_av
        self.function_norm = ValueError
        self.kwargs = {}
        self.needs_quadratic = True
        self.less_is_better = True
        self.function_clusters = ic_av_cluster


def ic_av_cluster(dists, labels):
    """
    :param dists: Data
    :param labels: labels
    :return:
    """
    meds = MEDs(dists, labels)
    clusters = [x for x in np.unique(labels) if x != -1]
    ret = {}
    for cluster in clusters:
        cluster_med = meds[cluster]
        ret[cluster] = np.sum(np.square(np.tril(cluster_med, k=-1)))
    return ret


# @timed
def ic_av(dists, labels):
    meds = MEDs(dists, labels)
    clusters = [x for x in np.unique(labels) if x != -1]
    ic_av = 0
    for cluster in clusters:
        cluster_med = meds[cluster]
        cluster_sum = np.sum(np.square(np.tril(cluster_med, k=-1)))
        ic_av += ((1 / cluster_med.shape[0]) * cluster_sum)
    return ic_av


def MEDs(dists, labels):
    clusters = [x for x in np.unique(labels) if x != -1]
    cluster_meds = {}
    for cluster in clusters:
        points = dists[labels == cluster, :][:, labels == cluster]
        mst_scipy = scipy.sparse.csgraph.minimum_spanning_tree(points)
        mst = Graph(mst_scipy.toarray())
        if mst_scipy.shape[0] > 1000:
            num_workers = 64
        else:
            num_workers = 32
        with warnings.catch_warnings():
            jada = parallel_bottleneck_paths(mst, num_workers)
        cluster_med = np.zeros_like(points)
        for i, results in jada:
            keys = list(results.keys())
            values = list(results.values())
            arr = np.array(values)
            arr = arr[np.argsort(keys)]
            cluster_med[i, :] = arr
        cluster_meds[cluster] = cluster_med
    return cluster_meds


def worker(n, G, results):
    results.append((n, minimum_bottleneck_paths(G, {n})))


def parallel_bottleneck_paths(G, num_workers):
    manager = multiprocessing.Manager()
    results = manager.list()
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(worker, [(n, G, results) for n in G])
    return list(results)


def minimum_bottleneck_paths(G, sources):
    if not sources:
        raise ValueError("sources must not be empty")
    from networkx.algorithms.shortest_paths.weighted import _weight_function
    weight = _weight_function(G, "weight")
    paths = {source: [source] for source in sources}  # dictionary of paths
    dist = _dijkstra_multisource(
        G, sources, weight, paths=paths, cutoff=None, target=None
    )
    return dist


def _dijkstra_multisource(
        G, sources, weight, pred=None, paths=None, cutoff=None, target=None
):
    G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []
    for source in sources:
        seen[source] = 0
        push(fringe, (np.inf, next(c), source))
    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break
        for u, e in G_succ[v].items():
            cost = weight(v, u, e)
            if cost is None or cost == 0:
                continue
            vu_dist = min(dist[v], cost)
            if cutoff is not None:
                if vu_dist > cutoff:
                    continue
            if u in dist:
                u_dist = dist[u]
                if pred is not None and vu_dist == u_dist:
                    pred[u].append(v)
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
                if pred is not None:
                    pred[u] = [v]
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)

    # The optional predecessor and path dictionaries can be accessed
    # by the caller via the pred and paths objects passed as arguments.
    return dist
