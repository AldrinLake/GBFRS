import os
import time
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import scipy.io as scio
import csv
import sys
import socket
import fcntl
import GranularBall
import multiprocessing
import numpy as np
import itertools

# Directory for recording runtime data table
record_filepath = 'file/dataset_record.csv'

# Directory for storing attribute reduction results
reduction_result_file_path = 'file/result/'
# Dataset to be used
dataset_use_path = 'file/datasets_use.txt'

# Directory of dataset files
dataset_file_url = r'../../标记分布数据集/'
# Headers of the runtime record table
record_head = ['数据集', '样本个数', '特征个数', '标签个数', '数据预处理方式', '聚类准则', '球内最大样本', '约简后特征个数', '运行时间', '记录时刻', '运行设备']

# Number of processes
processing_num = 10

def main():
    # =======================  Create folders  ======================
    # Check if the directory for storing feature selection result files exists
    if not os.path.exists(reduction_result_file_path):
        os.makedirs(reduction_result_file_path)
    # Check if the file recording the datasets to be used exists
    if not os.path.exists(dataset_use_path):
        print('数据集 ' + dataset_use_path + ' 不存在')
        sys.exit()
    # If the record table csv file does not exist, create it and write headers
    if not os.path.exists(record_filepath):
        csv_file = csv.writer(open(record_filepath, 'w', newline='', encoding='utf_8_sig'))
        csv_file.writerow(record_head)
    # =========================================================

    # =======================  Read dataset list to be used  ======================
    # dataset = ['CHD_49', 'Society', ]
    datasets = []
    f = open(dataset_use_path)
    line = f.readline()
    while line:
        # Read datasets that are not commented out
        if line.find('//') == -1:
            datasets.append(line.replace('\n', ''))
        line = f.readline()

    f.close()
    print(datasets)
    # =======================Parameter Settings==================================
    preProcessMethod = ['standard']  # minMax, standard, mM_std, std_mM
    distance_metric = ['euclidean']  # ['chebyshev','euclidean','manhattan']
    # [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
    miss_class_threshold = [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
    for dataset_name in datasets:
        data = scio.loadmat("{}/{}.mat".format(dataset_file_url, dataset_name))
        X = data['features'][:, :]
        samples_num = X.shape[0]
        ball_max_sample_number_list = []
        if samples_num <= 3000:
            ball_max_sample_number_list = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45]
        elif 3000 < samples_num <= 5000:
            ball_max_sample_number_list = [17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55]
        elif samples_num > 5000:
            ball_max_sample_number_list = [27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65]
        temp = list(itertools.product(datasets, preProcessMethod, distance_metric, miss_class_threshold, ball_max_sample_number_list))  # Cartesian product for parameter permutation combination, each tuple (dataset_name, parameter1, parameter2, ...)
        temp.reverse()
        parameters_list = multiprocessing.Manager().list(temp)
        Lock = multiprocessing.Manager().Lock()
        proc_list = []
        for proc_index in range(processing_num):
            parameter_dist = {
                'proc_index': proc_index,
                'parameters_list': parameters_list,
                "lock": Lock
            }
            proc = multiprocessing.Process(target=SingleProcess,args=(parameter_dist,))
            proc_list.append(proc)
            proc.start()
            time.sleep(0.5)
        # Wait for processes to finish
        for proc in proc_list:
            proc.join()
        # Terminate processes
        for proc in proc_list:
            proc.close()


def SingleProcess(parameter_dist):
    processing_index = parameter_dist['proc_index']
    share_lock = parameter_dist['lock']
    params_list = parameter_dist['parameters_list']
    while True:
        share_lock.acquire()
        if len(params_list) == 0:
            break
        params = params_list.pop()
        dataset_name = params[0]
        preProcessMethod = params[1]
        distance_metric = params[2]
        miss_class_threshold = params[3]
        max_sample_number = params[4]

        share_lock.release()
        print("==== process {}, dataset:{},{},{}, miss_class_threshold:{}, max_sample_number:{}, time:{}".format(processing_index, dataset_name, preProcessMethod, distance_metric,miss_class_threshold,max_sample_number,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        # Prevent program termination due to errors on a dataset, wrap with try
        try:
            # read a dataset
            data = scio.loadmat("{}/{}.mat".format(dataset_file_url, dataset_name))
            X = data['features'][:, :]
            Y = data['labels'][:, :]
            # normalize
            if preProcessMethod == 'minMax':
                # Normalize feature values
                minMax = MinMaxScaler()  # Normalize data
                X = minMax.fit_transform(X[:, :])
            elif preProcessMethod == 'standard':
                # Standardize features
                Ss = StandardScaler()
                X = Ss.fit_transform(X[:, :])
            else:
                print("Data not preprocessed")
            features_rank, run_time = GranularBall.AttributeReduction(X=X,Y=Y, min_sample=max_sample_number, miss_class_threshold=miss_class_threshold, distance_metric=distance_metric, dataset_name=dataset_name)
            # write the ranked features to txt file
            file_path = "{}/{}_{}/{}/".format(reduction_result_file_path, distance_metric, preProcessMethod, dataset_name)
            if os.path.exists(file_path) is False:
                os.makedirs(file_path)
            note = open("{}/{}_{}.txt".format(file_path, miss_class_threshold, max_sample_number), mode='w')
            note.write(','.join(str(i) for i in features_rank))
            note.close()
        except Exception as e:
            print(e)
            with open(record_filepath, 'a', newline='') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
                writer = csv.writer(f)
                writer.writerow([dataset_name, X.shape[0], X.shape[1], Y.shape[1], preProcessMethod, distance_metric,str(max_sample_number), 'Exception:' + str(e), '',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), socket.gethostname()])
                fcntl.flock(f, fcntl.LOCK_UN)
            continue
            # Record feature selection process information in the record table
        with open(record_filepath, 'a', newline='') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
            writer = csv.writer(f)
            writer.writerow([dataset_name, X.shape[0], X.shape[1], Y.shape[1], preProcessMethod, distance_metric,str(max_sample_number), len(features_rank), run_time, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), socket.gethostname()])
            fcntl.flock(f, fcntl.LOCK_UN)
    share_lock.release()




if __name__ == '__main__':
    main()
