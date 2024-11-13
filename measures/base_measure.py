import abc
import inspect
from copy import deepcopy

import numpy as np
import sklearn


class BaseMeasure(abc.ABC):

    def __init__(self):
        self.worst_value: float = -1
        self.best_value: float = -1
        self.name: str = "Undefined"
        self.needs_quadratic: bool = False
        self.kwargs: dict = {}
        self.less_is_better = False
        self.decimals = 3
        self.function_clusters = NotImplementedError
        self.normalization_params = None
        self.function_norm = NotImplementedError

    def plot_name(self):
        if self.less_is_better:
            return self.name + "↓"
        else:
            return self.name + "↑"

    @staticmethod
    def clean_outliers(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if labels.dtype == np.uint8:
            labels = labels.astype(np.int16)
        quadratic = data.shape[0] == data.shape[1]
        clusters, quantities = np.unique(labels, return_counts=True)
        singleton_clusters = clusters[quantities == 1].tolist()
        for singleton in singleton_clusters:
            labels[labels == singleton] = -1
        non_outliers = np.where(labels != -1)[0]
        if len(non_outliers) != len(labels):
            no_labels = labels[non_outliers]
            if quadratic:
                no_data = data[non_outliers, :][:, non_outliers]
                np.fill_diagonal(no_data, 0)
            else:
                no_data = data[non_outliers, :]
            return no_data, no_labels, len(non_outliers) / len(labels)
        else:
            return data, labels, 1

    @staticmethod
    def check_valid(labels: np.ndarray) -> bool:
        non_outliers = np.where(labels != -1)[0]
        if len(non_outliers) != len(labels):
            if len(non_outliers) == 0:
                return False
            no_labels = labels[non_outliers]
            if len(np.unique(no_labels)) == 1:
                return False
        elif len(np.unique(labels)) == 1:
            return False
        elif len(np.unique(labels)) == len(labels):
            return False
        return True

    def score(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        if not self.check_valid(labels):
            return self.worst_value
        if self.needs_quadratic:
            data = self.ensure_distance_matrix(data)
        data, labels, share = self.clean_outliers(data, labels)
        if not self.check_valid(labels):
            return self.worst_value
        kwargs_out = deepcopy(self.kwargs)
        for kw in kwargs:
            kwargs_out[kw] = kwargs[kw]
        args = inspect.getfullargspec(self.function).args
        if "X" in args:
            kwargs_out["X"] = data
        elif "dists" in args:
            kwargs_out["dists"] = data
        kwargs_out["labels"] = labels
        # start=time.time()
        # print(f"Start {self.name}")
        res = self.function(**kwargs_out)
        # print(f"Finished {self.name} in {time.time()-start:.2f}")
        if not self.less_is_better:
            ret = res * share
        else:
            ret = res * (1 / share)
        ret = self.ensure_finite(ret)
        return ret

    def score_norm_(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        worst_value = self.worst_value
        """if worst_value == -np.inf:
            worst_value=-99999999999999999999
        if worst_value == np.inf:
            worst_value=99999999999999999999"""
        if not self.check_valid(labels):
            return worst_value
        if self.needs_quadratic:
            data = self.ensure_distance_matrix(data)
        data, labels, share = self.clean_outliers(data, labels)
        if not self.check_valid(labels):
            return worst_value
        kwargs_out = deepcopy(self.kwargs)
        for kw in kwargs:
            kwargs_out[kw] = kwargs[kw]
        args = inspect.getfullargspec(self.function_norm).args
        if "X" in args:
            kwargs_out["X"] = data
        elif "dists" in args:
            kwargs_out["dists"] = data
        kwargs_out["labels"] = labels
        # start=time.time()
        # print(f"Start {self.name}")
        res = self.function_norm(**kwargs_out)
        # print(f"Finished {self.name} in {time.time()-start:.2f}")
        if not self.less_is_better:
            ret = res * share
        else:
            ret = res * (1 / share)
        ret = self.ensure_finite(ret)
        return ret

    def score_norm(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        if not type(self.function_norm) == type:
            kwargs_out = deepcopy(self.kwargs)
            for kw in kwargs:
                kwargs_out[kw] = kwargs[kw]
            res = self.score_norm_(data, labels, **kwargs_out)
        else:
            res = self.score(data, labels, **kwargs)
            # print(f"Unnormal result for {self.name} is {res}")
            if not (np.isinf(self.worst_value) or np.isinf(self.best_value)):
                res = -1 + (((res - self.worst_value) * 2) / (self.best_value - self.worst_value))
            else:
                res = np.tanh((res - self.normalization_params[0]) / self.normalization_params[1])
            if self.worst_value > self.best_value:
                res *= -1
        return res

    def score_clusters(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> dict:
        if not self.check_valid(labels):
            return self.worst_value
        dists = self.ensure_distance_matrix(data)
        dists, labels, share = self.clean_outliers(dists, labels)
        if not self.check_valid(labels):
            return self.worst_value
        kwargs_out = deepcopy(self.kwargs)
        for kw in kwargs:
            kwargs_out[kw] = kwargs[kw]
        if "dim" in inspect.getfullargspec(self.function).args and data.shape[0] != data.shape[1]:
            kwargs_out["dim"] = data.shape[1]
        if self.function_clusters == NotImplementedError:
            raise NotImplementedError(f"No score per cluster implemented for {self.name}")
        if self.function_clusters == ValueError:
            raise ValueError(f"No score per cluster possible for {self.name}")
        ret = self.function_clusters(dists, labels, **kwargs_out)
        for res in ret:
            ret[res] = self.ensure_finite(ret[res])
        return ret

    @staticmethod
    def ensure_distance_matrix(data: np.ndarray) -> np.ndarray:
        if data.shape[0] != data.shape[1]:
            data = sklearn.metrics.pairwise_distances(data, n_jobs=-1)
        return data

    def score_max(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        orig_score = self.score(data, labels, **kwargs)
        # orig_score = self.ensure_finite(orig_score)
        if self.less_is_better:
            return -orig_score
        return orig_score

    def score_min(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        orig_score = self.score(data, labels, **kwargs)
        # orig_score = self.ensure_finite(orig_score)
        if not self.less_is_better:
            return -orig_score
        return orig_score

    def ensure_finite(self, score):
        score = round(score, self.decimals)
        if type(score) == int:
            ii64 = np.iinfo(np.int32)
        else:
            ii64 = np.finfo(np.float32)
        if score == np.nan:
            score = self.worst_value
        if score == -np.inf:
            score = ii64.min
        elif score == np.inf:
            score = ii64.max
        return score

    def worst_value_finite(self):
        return self.ensure_finite(self.worst_value)

    def function(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        raise NotImplementedError

    def score_distance_function(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        if not self.check_valid(labels):
            return self.worst_value
        if self.needs_quadratic:
            dists = self.ensure_distance_matrix(data)
            dists, labels, share = self.clean_outliers(dists, labels)
            if not self.check_valid(labels):
                return self.worst_value
            kwargs_out = deepcopy(self.kwargs)
            for kw in kwargs:
                kwargs_out[kw] = kwargs[kw]
            args = inspect.getfullargspec(self.function).args
            if "X" in args:
                kwargs_out["X"] = dists
            elif "dists" in args:
                kwargs_out["dists"] = dists
            kwargs_out["labels"] = labels
            # start=time.time()
            # print(f"Start {self.name}")
            res = self.function(**kwargs_out)
            # print(f"Finished {self.name} in {time.time()-start:.2f}")
            if not self.less_is_better:
                ret = res * share
            else:
                ret = res * (1 / share)
            ret = self.ensure_finite(ret)
            return ret
        else:
            raise NotImplementedError(f"No score based on a  distance function is implemented for {self.name}")

    def score_distance_function_max(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        orig_score = self.score_distance_function(data, labels, **kwargs)
        # orig_score = self.ensure_finite(orig_score)
        if self.less_is_better:
            return -orig_score
        return orig_score

    def score_distance_function_min(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        orig_score = self.score_distance_function(data, labels, **kwargs)
        # orig_score = self.ensure_finite(orig_score)
        if not self.less_is_better:
            return -orig_score
        return orig_score
