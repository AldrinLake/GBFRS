import time

import scipy.io as scio
from numpy import *
import numpy as np
import warnings
import PySimpleGUI as sg
from scipy.spatial.distance import cdist
from numpy.distutils.fcompiler import none
from sklearn.preprocessing import StandardScaler
import random
from BasicFunction import differences, LabelSignificance
from BasicFunction.kMeans import kMeans
from BasicFunction import InformationGranular

"""
Granular Ball class: Used to initialize and construct granular balls
Granular ball splitting conditions:
    1. Each split selects the two samples with the largest label space difference within the ball as cluster centers
    2. When the number of samples in the ball is less than K, the ball stops splitting
"""

class GranularBall:
    """ class of the granular ball
        data format: [{attribute1,attribute2,...,attributeN},{index}]
    """

    def __init__(self, X_with_index, selected_feature_list, label_difference_matrix, distance_metric=''):
        """
        param X_with_index:  data set, the last column is the index of each line and each of the preceding columns corresponds to a feature
        param label_difference_matrix: label label_difference_matrix
        """
        self.distance_metric = distance_metric
        self.center = None
        self.label_distribution = None
        self.radius = None
        self.purity = None
        self.X_with_index = X_with_index[:, :]
        self.X = X_with_index[:,:-1]
        self.num, self.dim = self.X.shape
        self.selected_feature_list = selected_feature_list
        self.label_difference_matrix = label_difference_matrix


    def init_center(self):
        if len(self.X_with_index) > 0:
            self.center = self.X.mean(0)  # Compress rows, calculate mean for each column


    def __get_purity(self):
        """
        In multi-label data, the purity of a granular ball is determined by 1 - (average of label space differences within the ball)
        :return: the purity of the granular ball.
        """

        # Rewrite the purity of granular ball, use the mean of upper triangular matrix (excluding diagonal) of label space differences as purity
        if len(self.label_difference_matrix) > 1:
            difference = (np.sum(self.label_difference_matrix) - np.sum(np.diag(self.label_difference_matrix))) / (
                    np.square(len(self.label_difference_matrix)) - len(np.diag(self.label_difference_matrix)))
        else:
            difference = 0
        purity = 1 - difference
        return purity

    def init_purity(self):
        self.purity = self.__get_purity()

    def init_label_distribution(self, Y_all):
        """
        Define the label distribution of the ball as the proportion of each type of label among samples within the ball
        Y_all: Complete label space corresponding to the complete dataset
        Idea: The last value of each sample's feature vector represents the index of that sample in the original dataset. Through the indices of samples within the ball in the original dataset, extract their labels from the complete label set Y
        :return:
        """
        Y_all = np.array(Y_all)
        X_index = list(map(int, self.X_with_index[:, -1]))  # Get indices of samples within the ball in the original dataset
        Y_array = Y_all[X_index, :]
        # After transposing the label matrix, each row represents a vector of values for a label across samples, use the proportion of each label to represent the label distribution value of the granular ball
        # LD = [np.sum(row) / np.sum(Y_array) for row in Y_array.T]
        LD = []
        for row in Y_array.T:
            if np.sum(Y_array) == 0:
                LD.append(0)
            else:
                LD.append(np.sum(row) / np.sum(Y_array))

        self.label_distribution = LD


    def init_radius(self):
        distance_matrix = cdist([self.center[self.selected_feature_list]], self.X[:, self.selected_feature_list], metric=self.distance_metric)
        self.radius = np.mean(distance_matrix)

    def split_2balls(self):
        """
        split the granular ball to 2 new balls by using 2_means.
        Specify the two cluster centers of 2-means as the two samples with the largest difference in label space.
        """
        # Based on the label difference matrix, select the two elements with the largest label space difference as cluster centers
        h, w = self.label_difference_matrix.shape
        position = self.label_difference_matrix.argmax()
        row, col = position // w, position % w
        index_x1 = row
        index_x2 = col
        # print('---Selecting two cluster centers, sample indices: '+str(index_x1)+','+str(index_x2))
        # If all label space differences within the ball are 0, the split centers will be selected as 0,0. This needs to be corrected to two random samples
        if np.max(self.label_difference_matrix) == 0 and row == col == 0 and len(self.label_difference_matrix) > 1:
            ranges = range(len(self.label_difference_matrix))
            index_x1, index_x2 = random.sample(ranges, 2)
        k_centers = [self.X[index_x1, :], self.X[index_x2, :]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ====================== Use kmeans package ====================
            # res = k_means(X=self.X, n_clusters=2, n_init=1, init=np.array(k_centers))
            # ====================== Custom kmeans ====================
            km = kMeans(X=self.X, init_cluster_center=k_centers)
            if self.distance_metric == 'chebyshev':
                res = km.cluster_chebyshev()  # Chebyshev
            elif self.distance_metric == 'euclidean':
                res = km.cluster_euclidean()  # Euclidean
            elif self.distance_metric == 'manhattan':
                res = km.cluster_manhattan()  # Manhattan
            else:
                print("No distance metric selected!")
            # distance_cluster = res[0]   # Distance matrix from samples to each cluster center
            label_cluster = res[1]  # Clustering result of samples
            # ======================================================

        if sum(label_cluster == 0) and sum(label_cluster == 1):
            X0 = label_cluster == 0
            X1 = label_cluster == 1
            X0_lsm = self.label_difference_matrix[X0]  # First take rows
            X0_lsm = X0_lsm[:, X0]  # Then take columns
            X1_lsm = self.label_difference_matrix[X1]
            X1_lsm = X1_lsm[:, X1]
            # print('---self.label_similarity_matrix[X0, X0]:' + str(X0_lsm.shape))
            # print('---self.label_similarity_matrix[X1, X1]:' + str(X1_lsm.shape))
            ball1 = GranularBall(self.X_with_index[X0, :],self.selected_feature_list, X0_lsm, distance_metric=self.distance_metric)
            ball2 = GranularBall(self.X_with_index[X1, :],self.selected_feature_list, X1_lsm, distance_metric=self.distance_metric)
        else:
            ball1 = GranularBall(self.X_with_index[0:1, :],self.selected_feature_list, self.label_difference_matrix[0:1, 0:1],
                                 distance_metric=self.distance_metric)
            ball2 = GranularBall(self.X_with_index[1:, :],self.selected_feature_list, self.label_difference_matrix[1:, 1:],
                                 distance_metric=self.distance_metric)
        return ball1, ball2


class GBList:
    """ class of the list of granular ball
        X_with_index 格式 [{attribute1,attribute2,...,attributeN}{index}]
     """

    def __init__(self, X_with_index=None, selected_feature_list=None, label_difference_matrix=None, distance_metric=''):
        self.distance_metric = distance_metric
        self.X_with_index = X_with_index[:, :]
        self.selected_feature_list = selected_feature_list
        self.label_difference_matrix = label_difference_matrix
        self.granular_balls = [GranularBall(self.X_with_index, self.selected_feature_list, self.label_difference_matrix, distance_metric=self.distance_metric)]

    def init_granular_balls(self, T_purity=None, min_sample=1):
        """
        Split the balls, initialize the balls list.
        :param T_purity: If the purity of a ball is greater than this value, stop splitting.
        :param min_sample: If the number of samples of a ball is less than this value, stop splitting.
        """
        ll = len(self.granular_balls)
        i = 0
        while True:
            if self.granular_balls[i].num > min_sample:
                # print('---This granular ball needs to split')
                split_balls = self.granular_balls[i].split_2balls()
                self.granular_balls[i] = split_balls[0]
                self.granular_balls.append(split_balls[1])
                ll += 1
            else:
                i += 1
            if i >= ll:
                break
        self.X_with_index = self.get_data()

    # Calculate label distribution of all granular balls in the granular ball list as attributes of the granular balls themselves
    def init_granular_ball_label_distribution(self, Y_all):
        for gb in self.granular_balls:
            gb.init_label_distribution(Y_all)

    def init_granular_ball_center(self):
        for gb in self.granular_balls:
            gb.init_center()

    def init_granular_ball_radius(self):
        for gb in self.granular_balls:
            gb.init_radius()

    def init_granular_ball_purity(self):
        for gb in self.granular_balls:
            gb.init_purity()

    def get_data_size(self):
        """
         :return: the number of samples of each ball.
        """
        return list(map(lambda x: len(x.X_with_index), self.granular_balls))

    def get_data(self):
        """
        :return: Data from all existing granular balls in the GBlist.
        """
        list_data = [ball.X_with_index for ball in self.granular_balls]
        if len(list_data) > 0:
            return np.vstack(list_data)  # Convert to column relationship
        else:
            return []

    def get_center(self):
        """
        :return: the center of each ball.
        """
        return np.array(list(map(lambda x: x.center, self.granular_balls)))

    def get_purity(self):
        """
        :return: the center of each ball.
        """
        return np.array(list(map(lambda x: x.purity, self.granular_balls)))

    def re_k_means(self):
        """
        Global k-means clustering for data with the center of the ball as the initial center point.
        """
        k = len(self.granular_balls)
        with warnings.catch_warnings():

            # ====================== Use kmeans package ====================
            # label_cluster = k_means(X=self.X_with_index[:, :-1], n_init=1, n_clusters=k, init=self.get_center())[1]
            # ====================== Custom kmeans ====================

            km = kMeans(X=self.X_with_index[:, self.selected_feature_list], init_cluster_center=self.get_center()[:,self.selected_feature_list])

            if self.distance_metric == 'chebyshev':
                res = km.cluster_chebyshev()  # Chebyshev
            elif self.distance_metric == 'euclidean':
                res = km.cluster_euclidean()  # Euclidean
            elif self.distance_metric == 'manhattan':
                res = km.cluster_manhattan()  # Manhattan
            else:
                print("No distance metric selected!")
            # distance_cluster = res[0] 
            label_cluster = res[1]  # Clustering result of samples
            # # ======================================================

        for i in range(k):
            Xi = label_cluster == i
            X = self.X_with_index[Xi, :]
            X_index = list(map(int, X[:, -1]))
            Xi_lsm = self.label_difference_matrix[X_index, :]  # First take rows
            Xi_lsm = Xi_lsm[:, X_index]  # Then take columns
            self.granular_balls[i] = GranularBall(self.X_with_index[Xi, :],self.selected_feature_list, Xi_lsm, distance_metric=self.distance_metric)


    def re_division(self, selected_features):
        """
        Under the new feature subset, recalculate the distance from samples to previous cluster centers
        Data division with the center of the ball.
        :return: a list of new granular balls after divisions.
        """
        k = len(self.granular_balls)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ====================== Use kmeans package ====================
            # label_cluster = k_means(X=self.X_with_index[:, attributes], n_clusters=k, n_init=1, init=self.get_center()[:, attributes], max_iter=1)[1]
            # ====================== Custom kmeans ====================
            km = kMeans(X=self.X_with_index[:, selected_features], init_cluster_center=self.get_center()[:, selected_features])
            if self.distance_metric == 'chebyshev':
                res = km.cluster_chebyshev()  # Chebyshev
            elif self.distance_metric == 'euclidean':
                res = km.cluster_euclidean()  # Euclidean
            elif self.distance_metric == 'manhattan':
                res = km.cluster_manhattan()  # Manhattan
            else:
                print("No distance metric selected!")
            # distance_cluster = res[0] 
            label_cluster = res[1]  # Clustering result of samples
            # ======================================================

        granular_balls_division = []
        for i in range(k):
            Xi = label_cluster == i
            X = self.X_with_index[Xi, :]
            X_index = list(map(int, X[:, -1]))  # Get indices of samples in the original dataset
            Xi_lsm = self.label_difference_matrix[X_index, :]  # First take rows
            Xi_lsm = Xi_lsm[:, X_index]  # Then take columns
            gb = GranularBall(self.X_with_index[Xi, :],self.selected_feature_list, Xi_lsm, distance_metric=self.distance_metric)
            gb.init_purity()
            granular_balls_division.append(gb)
        return granular_balls_division

    def del_balls(self, num_data=2):
        """
        Deleting the balls that meets following conditions from the list, updating self.granular_balls and self.data.
        :param num_data: delete the balls that the number of samples is smaller than this value.
        :return: None
        """
        self.granular_balls = [ball for ball in self.granular_balls if ball.num >= num_data]
        # self.X_with_index = self.get_data()



def AttributeReduction(X, Y, min_sample=9, miss_class_threshold=None, distance_metric='euclidean', dataset_name=None):
    """
    The main function of attribute reduction.
    :param distance_metric: distance metric for cluster ['chebyshev','euclidean','manhattan']
    :param min_sample: control the minimal size stopping split the ball
    :param miss_class_threshold: the threshold control the extent to recognize the sample with different class
    :param X: data set, row denotes instance
    :param Y: label set
    :return: granular-ball centers and label distribution
    """
    start_time = time.time()
    # Integrate label importance into label space and standardize
    label_sig = LabelSignificance.label_significance_based_on_label_co_occurrence_degree(Y=Y)
    temp = Y * label_sig
    row_sum = np.sum(temp, axis=1)
    temp = temp.T / row_sum
    Y = temp.T
    # ===========================
    num, dim = X.shape
    # Retain number of features proportionally based on the size of the complete feature set
    select_feature_number = None
    if dim > 1000:
        select_feature_number = int(dim * 0.1)
    if 1000 >= dim > 500:
        select_feature_number = int(dim * 0.2)
    if 500 >= dim > 100:
        select_feature_number = int(dim * 0.3)
    if dim <= 100:
        select_feature_number = int(dim * 0.4)
    select_feature_number = select_feature_number + 2
    index = np.array(range(num)).reshape(num, 1)  # column of index
    X_with_index = np.hstack((X, index))  # Add the index column to the last column of the data
    label_difference_matrix = differences.difference(np.array(Y))  # Calculate the difference matrix of sample label space

    candidate_features = list(range(X.shape[1]))
    selected_features = []

    # Select the first two features
    while True:
        flag = None
        max_value = float('-inf')
        g = InformationGranular.InformationGranular(granular_ball_list=None, X_with_index=X_with_index, Y=Y, selected_features=selected_features, distance_metric=distance_metric, miss_class_threshold=miss_class_threshold)
        instance_label_difference = g.calculate_difference_relation_in_label_space2()
        for f in candidate_features:
            selected_features.append(f)
            g = InformationGranular.InformationGranular(granular_ball_list=None, X_with_index=X_with_index, Y=Y, selected_features=selected_features, distance_metric=distance_metric, miss_class_threshold=miss_class_threshold)
            POS = g.calculate_positive_region_for_first_feature(instance_label_difference=instance_label_difference)
            if POS > max_value:
                flag = f
                max_value = POS
            selected_features.remove(f)
        selected_features.append(flag)
        candidate_features.remove(flag)
        if len(selected_features) >=2:
            break
    # ==============================

    while True:
        flag = None
        max_value = float('-inf')

        granular_balls = GBList(X_with_index=X_with_index, selected_feature_list=selected_features, label_difference_matrix=label_difference_matrix, distance_metric=distance_metric)
        granular_balls.init_granular_balls(min_sample=min_sample)  # initialize the list
        granular_balls.init_granular_ball_center()
        granular_balls.re_k_means()
        granular_balls.del_balls(num_data=1)
        # Use purity mean to calculate local sample label consistency
        granular_balls.init_granular_ball_purity()
        local_label_consistency = np.mean(granular_balls.get_purity())
        # Calculate the lower approximation of granular ball fuzzy rough set
        granular_balls.init_granular_ball_center()
        granular_balls.init_granular_ball_label_distribution(Y_all=Y)
        g = InformationGranular.InformationGranular(granular_ball_list=granular_balls, X_with_index=X_with_index, Y=Y, selected_features=selected_features, distance_metric=distance_metric, miss_class_threshold=miss_class_threshold)
        POS = g.calculate_positive_region()
        instance_label_difference = g.calculate_difference_relation_in_label_space()
        for f in candidate_features:
            selected_features.append(f)
            # Calculate granular ball fuzzy rough set, new lower approximation after adding one-dimensional feature
            g = InformationGranular.InformationGranular(granular_ball_list=granular_balls, X_with_index=X_with_index, Y=Y, selected_features=selected_features, distance_metric=distance_metric, miss_class_threshold=miss_class_threshold)
            POS_new = g.calculate_positive_region_new(instance_label_difference=instance_label_difference)
            # Calculate purity mean after granular ball redivision after adding one-dimensional feature
            granular_balls_after_division = granular_balls.re_division(selected_features=selected_features)
            local_label_consistency_new = np.mean(list(map(lambda x: x.purity, granular_balls_after_division)))
            # Calculate feature importance, considering both lower approximation and local label consistency
            sig_f = (POS_new - POS) + (local_label_consistency_new - local_label_consistency)
            if sig_f > max_value:
                flag = f
                max_value = sig_f
            selected_features.remove(f)
            # Real-time progress bar
            sg.one_line_progress_meter('progress bar', len(selected_features), select_feature_number,
                                       "distance:{} \nmax_ball:{}\nmiss_class_threshold:{}\n dim:X{},Y{}\nselect:{}".format(
                                           distance_metric, min_sample, miss_class_threshold, str(X.shape),
                                           str(Y.shape), str(flag)))
        selected_features.append(flag)
        candidate_features.remove(flag)

        if len(selected_features) >= select_feature_number:
            break

    end_time = time.time()
    # Final feature result is array concatenation [sequentially selected features, remaining unselected features]
    final_result = selected_features + candidate_features
    # Add 1 here to make feature selection results start counting from 1
    final_result = list(np.array(final_result) + 1)
    return final_result, end_time - start_time


if __name__ == '__main__':
    datasetName = 'CAL500'
    data = scio.loadmat('../../ldl_data/' + datasetName + '.mat')
    X = data['features'][:, :]
    Y = data['labels'][:, :]
    # Normalize feature values
    Ss = StandardScaler()  # Normalize data
    X = Ss.fit_transform(X[:, :])

    res, time = AttributeReduction(X=X,Y=Y, min_sample=10, miss_class_threshold=0.5,distance_metric='euclidean')
    print(res)
    print(time)

