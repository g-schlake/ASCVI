import random

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets

from auto_clustering import run
from evaluations.dataset_fetcher import fetch_datasets_cd_synthetic


# import tqdm


def distance(point_a, point_b):
    sum = 0
    for i in range(0, len(point_a)):
        sum += (point_a[i] - point_b[i]) ** 2
    return np.sqrt(sum)


if __name__ == '__main__':
    rng = np.random.default_rng()
    data = rng.normal((1, 1), 0.3, (10, 2))
    data = np.append(data, rng.normal((3, 4), 0.3, (10, 2)), axis=0)
    data = np.append(data, rng.normal((1, 4), 0.3, (10, 2)), axis=0)
    data = np.append(data, rng.normal((3, 1), 0.3, (10, 2)), axis=0)
    n_samples = 500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8, n_features=3)
    no_structure = np.random.rand(n_samples, 2), None
    syn_datasets = fetch_datasets_cd_synthetic()
    twenty = (syn_datasets["twenty"]["data"], syn_datasets["twenty"]["labels"].flatten())
    chainlink = (syn_datasets["chainlink"]["data"], syn_datasets["chainlink"]["labels"])
    fifteen = (syn_datasets["R15"]["data"], syn_datasets["R15"]["labels"])
    # plt.xkcd()
    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )
    linked_samples = [[1, 2, 7, 4], [13, 17, 11], [21, 27, 25], [33, 31]]
    unlinked_samples = [[1, 13, 21, 33]]
    # labeled_data= [noisy_moons, noisy_circles, blobs, varied, aniso]
    for data in [twenty, fifteen, chainlink, noisy_moons, noisy_circles, blobs, varied, aniso, no_structure]:
        # for data in [noisy_circles]:
        if data[1] is None:
            distance = "euclidean"
            data = data[0]
            sns.scatterplot(x=data[:, 0], y=data[:, 1])
            plt.show()
            run(data, [], [], distance, preprocessing=False)
            continue
        num_samples = data[0].shape[0]
        linked_samples = []
        num_clusters = np.max(data[1]) + 1
        num_linked_samples = int(num_samples / 50) * num_clusters
        linked_indices = random.choices(range(0, num_samples), k=num_linked_samples)
        linked_samples = [[] for _ in range(num_clusters)]
        plot_labels = [-1] * num_samples
        for sample in linked_indices:
            linked_samples[data[1][sample]].append(sample)
            plot_labels[sample] = data[1][sample]
        data = data[0]
        # sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=plot_labels)
        # plt.show()
        distance = "euclidean"
        run(data, linked_samples, unlinked_samples, distance, preprocessing=False)
