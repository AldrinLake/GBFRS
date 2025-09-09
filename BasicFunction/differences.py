"""
@Author: xfk
@Date: 2022-12-14
Calculate the difference matrix of sample labels
Note: This difference metric is applicable to labels of label distribution data and multi-label data
"""
import numpy as np
from scipy.spatial.distance import cdist


def difference(sample_matrix):
    """
    Calculate the difference matrix of sample label vectors
    :param sample_matrix: Each row represents the label group vector corresponding to a sample
    :return:
    """
    sample_matrix = np.array(sample_matrix)
    # step1 Manhattan distance between each row in the vector matrix
    difference_matrix = cdist(sample_matrix, sample_matrix, metric='cityblock')
    return difference_matrix



if __name__ == '__main__':
    # data = [[0.1, 0.5, 0, 0.4],
    #         [0, 0.5, 0.1, 0.4],
    #         [0, 1, 0, 0]]

    data = [[0, 1, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 0]]

    diff = difference(data)
    print(diff)
