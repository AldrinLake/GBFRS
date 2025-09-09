"""
Label importance measurement in label distribution space, based on label co-occurrence degree
data 2023-2-25
code by xfk
"""
import scipy.io as scio
import numpy as np
import os

def label_significance_based_on_label_co_occurrence_degree(Y):
    label_num = Y.shape[1]
    label_sig_list = []
    # Traverse each label vector, calculate label importance
    for i in range(label_num):
        sum_i = 0
        current_label_vector = Y[:,i]
        for j in range(label_num):
            if i == j:
                continue
            similarity_between_i_and_j = 1-abs(current_label_vector - Y[:,j])
            where_are_label_occurrence = np.where(Y[:,[i,j]]>0,1,0)
            co_occurrence_degree_between_i_and_j = (2*np.sum(similarity_between_i_and_j))/np.sum(where_are_label_occurrence)
            sum_i = sum_i + co_occurrence_degree_between_i_and_j
        label_sig_list.append(sum_i)
    # Each importance divided by the sum of all importance
    label_sig_list_std = 1+(label_sig_list / np.sum(label_sig_list))
    return label_sig_list_std

if __name__ == '__main__':
    datasets = [
        'CHD_49',
        'Yeast_spo5',
        'Yeast_spoem',
        'Yeast_alpha',
        'Yeast_cdc',
        'Yeast_cold',
        'Yeast_diau',
        'Yeast_dtt',
        'Yeast_elu',
        'Yeast_heat',
        'Yeast_spo',
        'SJAFFE',
        'SBU_3DFE',
        'Natural_Scene',
        'CAL500',
        'Water-quality',
        'Flags',
        'Emotions',
        'VirusPseAAC',
        'Birds',
        'GpositivePseAAC',
        'PlantPseAAC',
        'GnegativePseAAC',
        'Image',
        'Scene',
        'HumanPseAAC',
        'VirusGO',
        'GpositiveGO',
    ]
    result_path = "../file/label_significance/"
    if os.path.exists(result_path) is False:
        os.makedirs(result_path)
    for dataset_name in datasets:
        data = scio.loadmat('../../../ldl_data/' + dataset_name + '.mat')
        Y = data['labels']
        labels_significance = label_significance_based_on_label_co_occurrence_degree(Y=Y)
        np.save('{}/{}.npy'.format(result_path, dataset_name), labels_significance)
        # ls = np.load('{}/{}.npy'.format(result_path, dataset_name))
        # print(len(ls))