import cv2
import numpy as np
import os
from tqdm import tqdm


def merge_txt(obj_num,):
    '''

    :return:
    '''


    start_evaluations = 0
    end_evaluations = 200
    step_num = 10


    save_point = np.linspace(int((end_evaluations - start_evaluations) / step_num + start_evaluations),
                             end_evaluations*all_num_txt, step_num*all_num_txt)

    info_per_object = 7
    for m in tqdm(range(solution_num)):

        after_path = target_path + 'labels_after_%s/' % m
        output_path = target_path + 'num_%d_after_%d.txt' % (obj_num, m)
        output_name_path = target_path + 'num_%d_after_name_%d.txt' % (obj_num, m)

        total_data = []
        total_data_name = []
        for s in save_point:
            data = np.loadtxt(after_path + 'num_%d_%d.txt' % (obj_num, int(s)))
            data_name = np.loadtxt(after_path + 'num_%d_%d_name.txt' % (obj_num, int(s)), dtype=str)
            total_data.append(data)
            total_data_name.append(data_name)
        total_data = np.asarray(total_data).reshape(-1, obj_num * info_per_object)
        total_data_name = np.asarray(total_data_name, dtype=str).reshape(-1, obj_num)

        np.savetxt(output_path, total_data)
        np.savetxt(output_name_path, total_data_name, fmt='%s')
        # if m == 0:
        #     total_data = []
        #     for s in save_point:
        #         results = np.loadtxt(before_path + 'num_%d_%d.txt' % (num, int(s)))
        #         total_data.append(results)
        #     total_data = np.asarray(total_data).reshape(-1, num * 5)
        #     np.savetxt(target_path + 'num_%d_before_%d.txt' % (num, m), total_data)

# def add():
#     '''
#     deprecated function
#     :return:
#     '''
#     base_path = '../../dataset/learning_data_817/'
#     add_path = '../dataset/learning_data_817_add/'
#     # for m in tqdm(range(solution_num)):
#     #     data_base = np.loadtxt(base_path + 'labels_after_%s/num_%d.txt' % (m, num_range[0]))
#     #     data_add = np.loadtxt(add_path + 'labels_after_%s/num_%d.txt' % (m, num_range[0]))
#     #     data_new = np.concatenate((data_base, data_add), axis=0)
#     #     np.savetxt(base_path + 'labels_after_%s/num_%d_new.txt' % (m, num_range[0]), data_new)
#     for m in tqdm(range(1)):
#         data_base = np.loadtxt(base_path + 'labels_before_%s/num_%d.txt' % (m, num_range[0]))
#         data_add = np.loadtxt(add_path + 'labels_before_%s/num_%d.txt' % (m, num_range[0]))
#         data_new = np.concatenate((data_base, data_add), axis=0)
#         np.savetxt(base_path + 'labels_before_%s/num_%d_new.txt' % (m, num_range[0]), data_new)

if __name__ == "__main__":


    obj_num = 10
    source_path = f'C:/Users/yuhan/PycharmProjects/Knolling_data/dataset/VAE_1118_obj{obj_num}/'
    target_path = source_path

    all_num_txt = 10

    label_num_start = 0
    label_num_end = all_num_txt*200

    solution_num = 12

    merge_txt(obj_num=obj_num)