import cv2
import numpy as np
import os
from tqdm import tqdm

def preprocess_img(grey_flag=False):
    '''

    :param grey_flag:
    :return:
    '''

    source_path = '../../../knolling_dataset/VAE_329_obj4/'
    target_path = '../../../knolling_dataset/VAE_329_obj4/'

    label_num_start = 0
    label_num_end = 1000
    sol_num = 12

    img_path = target_path + 'images_before/'
    os.makedirs(img_path, exist_ok=True)
    # img_num = os.listdir(img_path)

    num = label_num_start * sol_num
    for i in tqdm(range(label_num_start, label_num_end)):

        for j in range(sol_num):

            orig_img = cv2.imread(target_path + 'origin_images_before/label_%d_%d.png' % (i, j))
            img = cv2.resize(orig_img, (128, 128))
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # cv2.namedWindow('zzz', 0)
            # cv2.resizeWindow('zzz', 1280, 960)
            # cv2.imshow("zzz", img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            cv2.imwrite(img_path + '%d.png' % num, img)

            num += 1

def merge_txt():
    '''

    :return:
    '''

    num_per_scenario = 4

    start_evaluations = 0
    end_evaluations = 200
    step_num = 10
    solution_num = 12
    save_point = np.linspace(int((end_evaluations - start_evaluations) / step_num + start_evaluations),
                             end_evaluations, step_num)

    info_per_object = 7
    for m in tqdm(range(solution_num)):

        target_path = '../../dataset/VAE_1008_obj4/'
        after_path = target_path + 'labels_after_%s/' % m
        output_path = target_path + 'num_%d_after_%d.txt' % (num_per_scenario, m)
        output_name_path = target_path + 'num_%d_after_name_%d.txt' % (num_per_scenario, m)

        total_data = []
        total_data_name = []
        for s in save_point:
            data = np.loadtxt(after_path + 'num_%d_%d.txt' % (num_per_scenario, int(s)))
            data_name = np.loadtxt(after_path + 'num_%d_%d_name.txt' % (num_per_scenario, int(s)), dtype=str)
            total_data.append(data)
            total_data_name.append(data_name)
        total_data = np.asarray(total_data).reshape(-1, num_per_scenario * info_per_object)
        total_data_name = np.asarray(total_data_name, dtype=str).reshape(-1, num_per_scenario)

        np.savetxt(output_path, total_data)
        np.savetxt(output_name_path, total_data_name, fmt='%s')
        # if m == 0:
        #     total_data = []
        #     for s in save_point:
        #         results = np.loadtxt(before_path + 'num_%d_%d.txt' % (num, int(s)))
        #         total_data.append(results)
        #     total_data = np.asarray(total_data).reshape(-1, num * 5)
        #     np.savetxt(target_path + 'num_%d_before_%d.txt' % (num, m), total_data)

def add():
    '''
    deprecated function
    :return:
    '''
    base_path = '../../dataset/learning_data_817/'
    add_path = '../dataset/learning_data_817_add/'
    # for m in tqdm(range(solution_num)):
    #     data_base = np.loadtxt(base_path + 'labels_after_%s/num_%d.txt' % (m, num_range[0]))
    #     data_add = np.loadtxt(add_path + 'labels_after_%s/num_%d.txt' % (m, num_range[0]))
    #     data_new = np.concatenate((data_base, data_add), axis=0)
    #     np.savetxt(base_path + 'labels_after_%s/num_%d_new.txt' % (m, num_range[0]), data_new)
    for m in tqdm(range(1)):
        data_base = np.loadtxt(base_path + 'labels_before_%s/num_%d.txt' % (m, num_range[0]))
        data_add = np.loadtxt(add_path + 'labels_before_%s/num_%d.txt' % (m, num_range[0]))
        data_new = np.concatenate((data_base, data_add), axis=0)
        np.savetxt(base_path + 'labels_before_%s/num_%d_new.txt' % (m, num_range[0]), data_new)

if __name__ == "__main__":

    preprocess_img()