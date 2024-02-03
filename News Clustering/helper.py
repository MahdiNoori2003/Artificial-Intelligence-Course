import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import homogeneity_score, silhouette_score


def plot_cluster(X_test, y_pred, title, model=None):

    reduced_X = PCA(n_components=2).fit_transform(X_test)

    for label in set(y_pred):
        if (label == -1):
            plt.scatter(reduced_X[y_pred == label, 0],
                        reduced_X[y_pred == label, 1], label="Outlier")
        else:
            plt.scatter(reduced_X[y_pred == label, 0],
                        reduced_X[y_pred == label, 1], label=f"Cluster {label}")

    if (model):
        centroids = model.cluster_centers_
        reduced_centroids = PCA(n_components=2).fit_transform(centroids)
        plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], marker="x", s=130, linewidths=3,
                    color="hotpink", zorder=10, label="centroids")

    title = f"{title} clustering (PCA-reduced data)\n"
    if (model):
        title += "Centroids are marked with pink cross"

    plt.title(title)
    plt.legend()
    plt.show()


def dbscan_gridsearch(estimator, param_grid, X_train, X_test, y_test):
    eps_range = param_grid["eps"]
    min_samples_range = param_grid["min_samples"]
    max_score = -1
    res_grid = dict()

    for eps in eps_range:
        for min_samples in min_samples_range:
            estimator.eps = eps
            estimator.min_samples = min_samples
            estimator.fit(X_train)
            y_pred = estimator.fit_predict(X_test)
            if len(set(y_pred)) == 1:
                continue
            H = homogeneity_score(y_test, y_pred)
            S = silhouette_score(X_test, y_pred)

            if ((H+S)/2.0 > max_score and H > 0 and S > 0):
                res_grid["eps"] = eps
                res_grid["min_samples"] = min_samples
                max_score = (H+S)/2.0

    return res_grid
