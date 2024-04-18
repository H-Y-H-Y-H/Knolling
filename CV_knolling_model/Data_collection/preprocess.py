import cv2
import numpy as np
import os
from tqdm import tqdm

def change_name(source_path, target_path, start_idx, end_idx):

    label_num = 100
    sol_num = 12

    os.makedirs(target_path + 'images/', exist_ok=True)
    input_path = target_path + 'img_input/'
    output_path = target_path + 'img_output/'
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    num = 0
    for i in range(label_num):

        for j in range(sol_num):

            orig_img = cv2.imread(target_path + 'origin_images/label_%d_%d.png' % (i, j))
            img_input = orig_img[:, :640, :]
            img_output = orig_img[:, 640:, :]
            # print('here')
            cv2.imwrite(input_path + '%d.png' % num, img_input)
            cv2.imwrite(output_path + '%d.png' % num, img_output)

            num += 1
    pass

def transform(source_path, target_path):

    label_num_start = 7000
    label_num_end = 10000
    sol_num = 12

    img_path = target_path + 'images_before/'
    os.makedirs(img_path, exist_ok=True)
    # img_num = os.listdir(img_path)

    num = label_num_start * sol_num
    for i in tqdm(range(label_num_start, label_num_end)):

        for j in range(sol_num):

            orig_img = cv2.imread(target_path + 'origin_images_before/label_%d_%d.png' % (i, j))
            img = cv2.resize(orig_img, (128, 128))

            # cv2.namedWindow('zzz', 0)
            # cv2.resizeWindow('zzz', 1280, 960)
            # cv2.imshow("zzz", img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            cv2.imwrite(img_path + '%d.png' % num, img)

            num += 1

def check(source_path, target_path):

    num = 120000
    img_path = target_path + 'images_after/'
    for i in tqdm(range(num)):

        img = cv2.imread(img_path + '%d.png' % i)

if __name__ == "__main__":

    source_path = '../../../knolling_dataset/VAE_329_obj4/'
    target_path = '../../../knolling_dataset/VAE_329_obj4/'
    start_idx = 0
    end_idx = 1000
    # change_name(source_path=source_path, target_path=target_path, start_idx=start_idx, end_idx=end_idx)

    transform(source_path=source_path, target_path=target_path)
    # check(source_path=source_path, target_path=target_path)