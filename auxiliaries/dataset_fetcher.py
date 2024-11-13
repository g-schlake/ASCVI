import csv
import os

import numpy as np
import scipy.io
from sklearn import datasets

def fetch_datasets_dbcv():
    datasets = {}
    base_directory = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(base_directory, "datasets", "iris.data")) as iris_file:
        iris_labels = []
        iris_points = []
        for row in csv.reader(iris_file):
            if len(row) == 0:
                continue
            iris_points.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
            if row[4] == "Iris-setosa":
                iris_labels.append(0)
            elif row[4] == "Iris-versicolor":
                iris_labels.append(1)
            else:
                iris_labels.append(2)
    datasets["iris"] = {"data": iris_points, "labels": iris_labels}
    with open(os.path.join(base_directory, "datasets", "wine.data")) as wine_file:
        wine_labels = []
        wine_points = []
        for row in csv.reader(wine_file):
            if not row: continue
            wine_labels.append(int(row[0]))
            a = list(map(float, row[1:13]))
            wine_points.append(a)
    datasets["wine"] = {"data": wine_points, "labels": wine_labels}
    with open(os.path.join(base_directory, "datasets", "glass.data")) as glass_file:
        glass_labels = []
        glass_points = []
        for row in csv.reader(glass_file):
            if not row: continue
            glass_labels.append(int(row[10]))
            a = list(map(float, row[2:9]))
            glass_points.append(a)
    datasets["glass"] = {"data": glass_points, "labels": glass_labels}
    with open(os.path.join(base_directory, "datasets", "synthetic_control.data")) as kdd_file:
        kdd_labels = []
        kdd_points = []
        idx = 0
        for row in csv.reader(kdd_file, delimiter=' '):
            kdd_labels.append(int(idx / 100))
            row = list(filter(lambda item: item != '', row))
            idx += 1
            if not row: continue
            a = list(map(float, row))
            kdd_points.append(a)
    datasets["kdd"] = {"data": kdd_points, "labels": kdd_labels}
    with open(os.path.join(base_directory, "datasets", "cell237.txt")) as cell237_file:
        cell237_labels = []
        cell237_points = []
        first = True
        for row in csv.reader(cell237_file, delimiter='\t'):
            if first:
                first = False
                continue
            if not row: continue
            cell237_labels.append(int(row[1]))
            a = list(map(float, row[2:18]))
            cell237_points.append(a)
    datasets["cell237"] = {"data": cell237_points, "labels": cell237_labels}
    with open(os.path.join(base_directory, "datasets", "cell384.txt")) as cell384_file:
        cell384_labels = []
        cell384_points = []
        first = True
        for row in csv.reader(cell384_file, delimiter='\t'):
            if first:
                first = False
                continue
            if not row: continue
            cell384_labels.append(int(row[1]))
            a = list(map(float, row[2:18]))
            cell384_points.append(a)
    datasets["cell384"] = {"data": cell384_points, "labels": cell384_labels}
    for name, dataset in datasets.items():
        data, labels = clean(dataset["data"], dataset["labels"])
        datasets[name] = {"data": data, "labels": labels}
    return datasets

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
    import arff
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
                        # print("Dataset "+dataset_name+" has Labels not direchtly castable "+str(ls[claspos]))
                        fail = True
                        break
                    data.append(ls[:claspos] + ls[claspos + 1:])
            except ValueError as Err:
                pass
                # print("Dataset " + dataset_name + " has bad Values.")
            if fail: continue
        else:
            continue
        dataset, labels = clean(data, labels)
        if dataset_name in other_outlier:
            if labels.dtype == np.dtype('uint8'):
                labels = labels.astype(np.dtype('int8'))
            labels[labels == other_outlier[dataset_name]] = -1
        datasets[dataset_name] = {"data": dataset, "labels": labels}
    return datasets


def fetch_datasets_cd_uci():
    import arff
    evaluations_directory = os.path.dirname(os.path.realpath(__file__))
    if not evaluations_directory.endswith("evaluations"):
        evaluations_directory = os.path.join(evaluations_directory, "evaluations")
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
                        # print("Dataset "+dataset_name+" has Labels not direchtly castable "+str(ls[claspos]))
                        fail = True
                        break
                    data.append(ls[:claspos] + ls[claspos + 1:])
            except ValueError as Err:
                pass
                # print("Dataset " + dataset_name + " has bad Values.")
            except IndexError as Err:
                pass
                # print("Dataset " + dataset_name + " has bad Indizes.")
            data = np.array(data)
            labels = np.array(labels)
            if fail: continue
        else:
            continue
        if len(data) == 0:
            continue
        if data.dtype.type in [np.string_, np.str_]:
            data = data.astype(int)
        if np.isnan(data).any():
            data = data[~np.isnan(data).any(axis=1)]
        dataset, labels = clean(data, labels)
        datasets[dataset_name] = {"data": dataset, "labels": labels}
    return datasets


def fetch_datasets():
    datasets = {"dbcv": fetch_datasets_dbcv()}
    datasets["uci"] = fetch_datasets_cd_uci()
    datasets["syntetic"] = fetch_datasets_cd_synthetic()
    # datasets["ac"] = fetch_datasets_ac()
    return datasets


def fetch_datasets_real_syn():
    return {"real": {**fetch_datasets_dbcv(), **fetch_datasets_cd_uci()}, "synthetic": fetch_datasets_cd_synthetic()}


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


