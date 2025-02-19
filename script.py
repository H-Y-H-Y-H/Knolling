

import shutil
import os


import time
print()


for obj_num in range(2,11):
    load_path = f'C:/Users/yuhan/PycharmProjects/Knolling_data/dataset/VAE_1118_obj{obj_num}/'
    save_path = f'C:/Users/yuhan/PycharmProjects/Knolling_data/0212dataset/obj{obj_num}/'
    os.makedirs(save_path,exist_ok=True)
    for m in range(12):
        data_path  = load_path + 'num_%d_after_%d.txt' % (obj_num, m)
        name_path = load_path + 'num_%d_after_name_%d.txt' % (obj_num, m)

        save_data_path = save_path + f'{obj_num}obj_tidy_data_cdn{m}.txt'
        save_name_path = save_path + f'{obj_num}obj_tidy_name_cdn{m}.txt'

        shutil.copyfile(data_path,save_data_path)
        shutil.copyfile(name_path,save_name_path)