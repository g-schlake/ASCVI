from auxiliaries.dataset_fetcher import fetch_datasets_sklearn
from measures.registry import get_measures


if __name__ == '__main__':
    datasets = fetch_datasets_sklearn()
    for ds_name, ds_dict in datasets.items():
        data = ds_dict['data']
        labels = ds_dict['labels']
        for measure in get_measures():
            scorer = measure()
            print(f"Starting {ds_name}: {scorer.plot_name()}")
            print(f"Value for {ds_name}: {scorer.plot_name()} = {scorer.score(data, labels)}")