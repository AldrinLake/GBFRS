# Clustering algorithm suitable for mixed data (continuous, nominal)
import sys

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist


class kMeans:
    def __init__(self, X, init_cluster_center):
        """
        Initialize
        :param X: Data table, each row represents a data vector
        :param init_cluster_center: Initialize cluster centers, each row represents a cluster center
        :param nominal_feature_index: Array of indices for nominal features
        """
        self.X = X
        self.init_cluster_center = np.array(init_cluster_center)
        self.n_cluster = len(init_cluster_center)

    # Euclidean distance clustering
    def cluster_euclidean(self):
        # Calculate distance matrix between sample matrix (n,d) and cluster center matrix (k,d), get sample distance matrix (n, k)
        distance_matrix = cdist(self.X, self.init_cluster_center, metric='euclidean')
        return distance_matrix.T, np.argsort(distance_matrix).T[0]

    # Chebyshev distance clustering
    def cluster_chebyshev(self):
        # Calculate distance matrix between sample matrix (n,d) and cluster center matrix (k,d), get sample distance matrix (n, k)
        distance_matrix = cdist(self.X, self.init_cluster_center, metric='chebyshev')
        return distance_matrix.T, np.argsort(distance_matrix).T[0]

    # Manhattan distance clustering
    def cluster_manhattan(self):
        # Calculate distance matrix between sample matrix (n,d) and cluster center matrix (k,d), get sample distance matrix (n, k)
        distance_matrix = cdist(self.X, self.init_cluster_center, metric='cityblock')
        return distance_matrix.T, np.argsort(distance_matrix).T[0]


if __name__ == '__main__':
    pass
