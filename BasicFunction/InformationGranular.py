from numpy import *
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler



class InformationGranular:
    def __init__(self, X_with_index, Y, selected_features, granular_ball_list=None, distance_metric=None, miss_class_threshold=None):
        if distance_metric is None:
            print("must choose a distance metric")
        self.granular_ball_list = granular_ball_list
        self.X_with_index = X_with_index
        self.Y = Y
        self.selected_features = selected_features
        self.distance_metric = distance_metric
        self.miss_class_threshold = miss_class_threshold
        # Extract (center, radius, label distribution) of each ball
        ball_center_list = []
        ball_radius_list = []
        ball_label_distribution = []
        if granular_ball_list != None:
            for ball in granular_ball_list.granular_balls:
                ball_center_list.append(ball.center)
                ball_radius_list.append(ball.radius)
                ball_label_distribution.append(ball.label_distribution)
            self.ball_center_list = np.array(ball_center_list)
            self.ball_radius_list = np.array(ball_radius_list)
            self.ball_label_distribution = np.array(ball_label_distribution)
            self.center_to_sample_distance_matrix = cdist(self.ball_center_list[:,self.selected_features], self.X_with_index[:, self.selected_features], metric=self.distance_metric)



    def calculate_difference_relation_in_feature_space(self):
        """
        :return: difference relation matrix between ball approximation center and all samples
        """
        distance_matrix = self.center_to_sample_distance_matrix
        instance_similarity_relationship = 1/(1+distance_matrix)
        return 1-instance_similarity_relationship

    def calculate_instance_neighborhood_relation(self):
        """
        :return: neighborhood relation matrix related to ball center
        """
        distance_matrix = self.center_to_sample_distance_matrix
        neighborhood_relation_matrix = []
        for i in range(len(distance_matrix)):
            where_are_instances_belong_to_this_granule = distance_matrix[i] <= self.ball_radius_list[i]
            neighborhood_relation_matrix.append(where_are_instances_belong_to_this_granule)
        neighborhood_relation_matrix = np.array(neighborhood_relation_matrix, dtype=object)
        return neighborhood_relation_matrix


    def calculate_difference_relation_in_label_space(self):
        """
        :return: the difference between ball and each instance in label space
        """
        distance_matrix = cdist(self.ball_label_distribution, self.Y, metric='cityblock')
        instance_label_difference = distance_matrix * 0.5
        # Normalize each row of the label difference matrix using min-max scaling
        minMax = MinMaxScaler()  # Normalize data
        instance_label_difference = minMax.fit_transform(instance_label_difference.T).T
        return instance_label_difference

    def calculate_difference_relation_in_label_space2(self):
        """
        :return: the labels difference among instances
        """
        distance_matrix = cdist(self.Y, self.Y, metric='cityblock')
        instance_label_difference =  distance_matrix * 0.5
        # Normalize each row of the label difference matrix using min-max scaling
        minMax = MinMaxScaler()  # Normalize data
        instance_label_difference = minMax.fit_transform(instance_label_difference.T).T
        return instance_label_difference


    def calculate_positive_region(self):
        # Calculate sample feature space difference
        instance_feature_difference = self.calculate_difference_relation_in_feature_space()
        # Calculate label space difference
        instance_label_difference = self.calculate_difference_relation_in_label_space()

        # Find the positions of k different class samples for each object participating in the calculation
        where_are_nearest_different_class_samples_participate_in_compute_lower_approximation = self.calculate_k_nearest_different_class_samples_that__participate_in_compute_of_lower_approximation(
            instance_label_difference=instance_label_difference,
            instance_feature_difference=instance_feature_difference)

        temp_matrix = where_are_nearest_different_class_samples_participate_in_compute_lower_approximation * instance_feature_difference * instance_label_difference

        res = np.sum(temp_matrix, axis=1) / (1 + np.sum(where_are_nearest_different_class_samples_participate_in_compute_lower_approximation, axis=1))
        res = np.mean(res)
        return res

    def calculate_positive_region_new(self,  instance_label_difference):
        # Calculate sample feature space difference
        instance_feature_difference = self.calculate_difference_relation_in_feature_space()
        # To reduce redundant calculations, sample label space difference is passed through function parameters
        # Find the positions of k different class samples for each object participating in the calculation
        where_are_nearest_different_class_samples_participate_in_compute_lower_approximation = self.calculate_k_nearest_different_class_samples_that__participate_in_compute_of_lower_approximation(
            instance_label_difference=instance_label_difference,
            instance_feature_difference=instance_feature_difference)

        temp_matrix = where_are_nearest_different_class_samples_participate_in_compute_lower_approximation * instance_feature_difference * instance_label_difference

        res = np.sum(temp_matrix, axis=1) / (1 + np.sum(where_are_nearest_different_class_samples_participate_in_compute_lower_approximation, axis=1))
        res = np.mean(res)
        return res

    def calculate_k_nearest_different_class_samples_that__participate_in_compute_of_lower_approximation(self, instance_label_difference, instance_feature_difference):
        """
        *instance_label_difference_large_than_threshold
        *instance_feature_difference
        # ① Samples with label difference greater than α and less than 1, quantity is k/2
        # ② Samples with label difference equal to 1, select nearest samples according to nearest neighbor principle, quantity is k/2
        """
        # Based on the already calculated label difference matrix greater than threshold, continue to calculate which label differences are less than 1
        where_are_instance_label_difference_large_than_threshold_and_small_than_one = self.find(arr=instance_label_difference,min=self.miss_class_threshold,max=1)
        half_of_k = np.sum(where_are_instance_label_difference_large_than_threshold_and_small_than_one, axis=1) # Row sum
        # Calculate feature space difference
        # Calculate feature space differences of completely different class samples
        where_are_instance_label_difference_equal_to_one = instance_label_difference >= 1
        instance_feature_difference_with_truly_different_labels = instance_feature_difference * where_are_instance_label_difference_equal_to_one

        # For each object, find k nearest samples that are completely different classes in feature space, specified quantity half_of_k
        distance_sort = np.argsort(instance_feature_difference_with_truly_different_labels)
        nearest_relation_matrix_for_truly_different_labels = np.zeros_like(distance_sort)

        for i in range(len(distance_sort)):
            # print(half_of_k[i])
            nearest_relation_matrix_for_truly_different_labels[i, distance_sort[i, 0:max(half_of_k[i],1)]] = 1
        # Merge parts ①②
        where_are_nearest_different_class_samples_participate_in_compute_lower_approximation = where_are_instance_label_difference_large_than_threshold_and_small_than_one + nearest_relation_matrix_for_truly_different_labels
        return where_are_nearest_different_class_samples_participate_in_compute_lower_approximation

    def calculate_positive_region_for_first_feature(self, instance_label_difference):
        # Calculate sample feature space difference
        distance_matrix_features = cdist(self.X_with_index[:, self.selected_features], self.X_with_index[:, self.selected_features], metric=self.distance_metric)
        instance_feature_similarity = 1 / (1 + distance_matrix_features)
        instance_feature_difference = 1 - instance_feature_similarity

        # Label space difference, to reduce redundant calculations, this item is passed through function parameters


        # Calculate the positions of k nearest different class samples for each sample, neighbors consist of two parts:
        # ① Samples with label difference greater than α and less than 1, quantity is k/2
        # ② Samples with label difference equal to 1, select nearest samples according to nearest neighbor principle, quantity is k/2
        where_are_nearest_different_class_samples_participate_in_compute_lower_approximation = self.calculate_k_nearest_different_class_samples_that__participate_in_compute_of_lower_approximation(
            instance_label_difference=instance_label_difference,
            instance_feature_difference=instance_feature_difference)

        temp_matrix = where_are_nearest_different_class_samples_participate_in_compute_lower_approximation * instance_feature_difference * instance_label_difference

        res = np.sum(temp_matrix, axis=1) / (1+np.sum(where_are_nearest_different_class_samples_participate_in_compute_lower_approximation, axis=1))
        res = np.mean(res)
        return res

    def find(self, arr, min, max):
        # Find elements in the matrix whose values belong to a certain interval
        pos_min = arr > min
        pos_max = arr < max
        pos_rst = pos_min * pos_max
        return pos_rst