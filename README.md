# ASCVI
This GitHub Repository is a selection of Python implementation of **Clustering Validation Indices** (*CVIs*) for 
*Arbitrary Shaped Clustering*.
Every scorer can score datasets according to their labels using the `score` function. 
This does automatically add a penalty based on the number of outliers, if no implicit  outlier handling is done.
It is possible, to ensure the scores to be maximized and minimized using the `score_max` and `score_min` function
(multiplying the acutual scores by -1 to fit the wanted behaviour).
There exists a normalized version (`score_norm`), but on most CVIs, this is simply a z-normalization using the values
from our empirical study.
As not all datasets behave like in our study and the precondition for this is not met, this should be viewed as highly
experimental.
If you use our implementations in your work, please cite
> Schlake, Georg Stefan, and Christian Beecks. 
> "Validating Arbitrary Shaped Clusters-A Survey." 
> 2024 IEEE 11th International Conference on Data Science and Advanced Analytics (DSAA). IEEE, 2024.

Citations for all used CVIs can be found in their respective classes.
An example of  the  usage can be seen in `main.py`.
The `dataset_fetcher` in  the `auxiliaries` folder can retrieve our toy datasets from scikit-learn as well as the 
synthetical and real datasets from the [clustering datasets repository](https://github.com/milaan9/Clustering-Datasets).
We adopted some of these datasets slightly to be compatible with this program. 