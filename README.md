# ASCVI
This GitHub Repository is a selection of Python implementations of **Clustering Validation Indices** (*CVIs*) for 
*Arbitrary Shaped Clustering*.
Every scorer can score datasets according to their labels using the `score` function. 
This does automatically add a penalty based on the number of outliers, if no implicit outlier handling is done.
It is possible, to ensure the scores to be maximized and minimized using the `score_max` and `score_min` function
(multiplying the actual scores by -1 to fit the wanted behaviour).
There is a normalized version (`score_norm`), but on most CVIs this is simply a z-normalization using the values
from our empirical study.
As not all datasets behave like the ones in our study and may not meet all precondition, this should be viewed as highly
experimental.
If you use our implementations in your work, please cite
> Schlake, Georg Stefan, and Christian Beecks. 
> "Validating Arbitrary Shaped Clusters-A Survey." 
> 2024 IEEE 11th International Conference on Data Science and Advanced Analytics (DSAA). IEEE, 2024.
> https://doi.org/10.1109/DSAA61799.2024.10722773

Citations for all used CVIs can be found in the docstrings their respective classes.
An example usage can be seen in `main.py`.
The `dataset_fetcher` in the `auxiliaries` folder retrieves our toy datasets from scikit-learn as well as the 
synthetic and real datasets from the [clustering datasets repository](https://github.com/milaan9/Clustering-Datasets).
We adopted some of these datasets slightly to make them compatible with our program. 
The Repo was tested with Python 3.10.
