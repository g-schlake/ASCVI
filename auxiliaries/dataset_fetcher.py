import os

import numpy as np
import scipy.io
from sklearn import datasets
import arff


def fetch_datasets_sklearn(n_samples=500):
    dataset = {}
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    dataset["circles"] = {"data": noisy_circles[0], "labels": noisy_circles[1]}
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
    dataset["moons"] = {"data": noisy_moons[0], "labels": noisy_moons[1]}
    blobs = datasets.make_blobs(n_samples=n_samples, n_features=3)
    dataset["blobs"] = {"data": blobs[0], "labels": blobs[1]}
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    dataset["anisotropic"] = {"data": X_aniso, "labels": y}
    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )
    dataset["varied"] = {"data": varied[0], "labels": varied[1]}
    return dataset


def fetch_datasets_cd_synthetic():
    evaluations_directory = os.path.dirname(os.path.realpath(__file__))
    other_outlier = {"zelnik2": 2, "cure-t2-4k": 6}
    folder = os.path.join(evaluations_directory, "Clustering-Datasets", "02. Synthetic")
    datasets = {}
    for filename in os.listdir(folder):
        dataset_name = filename.split('.')[0]
        if dataset_name in datasets:
            continue
        if filename.endswith(".mat"):
            dataset = scipy.io.loadmat(os.path.join(folder, filename))
            data = dataset["data"]
            labels = dataset["label"]
        elif filename.endswith(".arff"):
            dataset = arff.load(os.path.join(folder, filename))
            data = []
            labels = []
            first = True
            fail = False
            try:
                for row in dataset:
                    if first:
                        try:
                            y = row["class"]
                        except KeyError:
                            try:
                                y = row["CLASS"]
                            except KeyError:
                                # print("Dataset "+dataset_name+" has no Attribute Class.")
                                fail = True
                                break
                        ls = list(row)
                        claspos = ls.index(y)
                        first = False
                    if not row:
                        continue
                    ls = list(row)
                    try:
                        labels.append(int(ls[claspos]))
                    except ValueError:
                        print("Dataset "+dataset_name+" has Labels not direchtly castable "+str(ls[claspos]))
                        fail = True
                        break
                    data.append(ls[:claspos] + ls[claspos + 1:])
            except ValueError as Err:
                pass
                print("Dataset " + dataset_name + " has bad Values.")
                print(Err)
            if fail: continue
        else:
            continue
        if len(labels)==0:
            continue
        dataset, labels = clean(data, labels)
        if dataset_name in other_outlier:
            if labels.dtype == np.dtype('uint8'):
                labels = labels.astype(np.dtype('int8'))
            labels[labels == other_outlier[dataset_name]] = -1
        datasets[dataset_name] = {"data": dataset, "labels": labels}
    return datasets


def fetch_datasets_cd_uci():
    evaluations_directory = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(evaluations_directory, "Clustering-Datasets", "01. UCI")
    datasets = {}
    for filename in os.listdir(folder):
        dataset_name = filename.split('.')[0]
        if dataset_name in datasets:
            continue
        if filename.endswith(".mat"):
            dataset = scipy.io.loadmat(os.path.join(folder, filename))
            data = dataset["data"]
            labels = dataset["label"]
        elif filename.endswith(".arff"):
            dataset = arff.load(os.path.join(folder, filename))
            data = []
            labels = []
            first = True
            fail = False
            try:
                for row in dataset:
                    if first:
                        try:
                            y = row["class"]
                        except KeyError:
                            try:
                                y = row["CLASS"]
                            except KeyError:
                                try:
                                    y = row["Class"]
                                except KeyError:
                                    print("Dataset "+dataset_name+" has no Attribute Class.")
                                    fail = True
                                    break
                        ls = list(row)
                        claspos = ls.index(y)
                        first = False
                    if not row:
                        continue
                    ls = list(row)
                    try:
                        labels.append(int(ls[claspos]))
                    except ValueError:
                        print("Dataset "+dataset_name+" has Labels not direchtly castable "+str(ls[claspos]))
                        fail = True
                        break
                    data.append(ls[:claspos] + ls[claspos + 1:])
            except ValueError as Err:
                print("Dataset " + dataset_name + " has bad Values.")
                print(Err)
            except IndexError as Err:
                print("Dataset " + dataset_name + " has bad Indizes.")
                print(Err)
            data = np.array(data)
            labels = np.array(labels)
            if fail: continue
        else:
            continue
        if len(data) == 0:
            continue
        try:
            if data.dtype.type in [np.bytes_, np.str_]:
                data = data.astype(int)
        except ValueError as Err:
            print("Dataset " + dataset_name + " has uncastable Values.")
            print(Err)
            continue
        if np.isnan(data).any():
            data = data[~np.isnan(data).any(axis=1)]
        dataset, labels = clean(data, labels)
        datasets[dataset_name] = {"data": dataset, "labels": labels}
    return datasets


def fetch_datasets():
    datasets = {}
    datasets["uci"] = fetch_datasets_cd_uci()
    datasets["syntetic"] = fetch_datasets_cd_synthetic()
    # datasets["ac"] = fetch_datasets_ac()
    return datasets


def fetch_dataset(name: str):
    dss = fetch_datasets()
    for datasets in dss.values():
        if name in datasets:
            return datasets[name]
    raise ValueError("No dataset with " + name)


def clean(dataset, labels):
    dataset = np.array(dataset)
    if type(labels[0]) == list:
        true_labels = np.array([x for y in labels for x in y])
    elif type(labels) == list:
        true_labels = np.array(labels)
    else:
        true_labels = labels
    if type(true_labels) == np.ndarray:
        if len(true_labels.shape) == 2:
            if true_labels.shape[1] == 1:
                true_labels = true_labels.reshape(true_labels.shape[0])
            elif true_labels.shape[0] == 1:
                true_labels = true_labels.reshape(true_labels.shape[1])
            else:
                raise RuntimeWarning("Invalid Labels detected")
    valid_indizes = np.where(~np.isnan(true_labels))[0]
    nan_values = len(valid_indizes) != len(true_labels)
    if nan_values:
        true_labels = np.array([true_labels[x] for x in valid_indizes])
    dataset = dataset[valid_indizes]
    return dataset, true_labels


def clean_labels(labels):
    if type(labels[0]) == list:
        true_labels = np.array([x for y in labels for x in y])
    elif type(labels) == list:
        true_labels = np.array(labels)
    else:
        true_labels = labels
    if type(true_labels) == np.ndarray:
        if len(true_labels.shape) == 2:
            if true_labels.shape[1] == 1:
                true_labels = true_labels.reshape(true_labels.shape[0])
            elif true_labels.shape[0] == 1:
                true_labels = true_labels.reshape(true_labels.shape[1])
            else:
                raise RuntimeWarning("Invalid Labels detected")
    valid_indizes = np.where(~np.isnan(true_labels))[0]
    if len(valid_indizes) != len(true_labels):
        true_labels = np.array([true_labels[x] for x in valid_indizes])
    return true_labels
