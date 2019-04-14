Our evaluation metric is the Silhouette Score (in addition to qualitative analysis in the final report/presentation).

As described in the scikit-learn documentation: "The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b)."

The score ranges from -1 to 1, with 1 being the best possible score and -1 being the worst possible score.

To run the scoring file with a simple-baseline file (call it results.csv) and display the score, run the following in a terminal:

python score.py --input results.csv