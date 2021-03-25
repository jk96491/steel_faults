from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import numpy as np
from Utils import normalize
import random

random_Seed = random.randrange(0, 16546)
marker = ['o', 's', '^', 'x', '*', 's', 'X']
use_normalize = True
exclude_cols = [11, 12, 19, 20, 26]


def train():
    algorithm = "GMM"

    n_cluster = 7

    xy = np.loadtxt('faults.csv', delimiter=',') #1941
    x_data = xy[:, :-n_cluster]

    if use_normalize:
        normalize(x_data, exclude_cols)

    pandas_data = pd.DataFrame(data=x_data)

    if algorithm == "K-Means":
        result = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=300, random_state=random_Seed).fit(pandas_data).labels_
    elif algorithm == "GMM":
        gmm = GaussianMixture(n_components=7, random_state=random_Seed)
        gmm_label = gmm.fit(pandas_data).predict(pandas_data)
        result = gmm_label
    else:
        return

    pandas_data['cluster'] = result

    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(pandas_data)

    pandas_data['pca_x'] = pca_transformed[:, 0]
    pandas_data['pca_y'] = pca_transformed[:, 1]
    pandas_data.head(3)

    marker_ind = []

    for i in range(n_cluster):
        marker_ind.append(pandas_data[pandas_data['cluster'] == i].index)

    for i in range(n_cluster):
        plt.scatter(x=pandas_data.loc[marker_ind[i], 'pca_x'], y=pandas_data.loc[marker_ind[i], 'pca_y'], marker=marker[i])

    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('7 Clusters Visualization by 2 PCA Components({0})'.format(algorithm))

    plt.show()

train()