import numpy as np
from tqdm import tqdm

configuration = [[2, 1],
                 [1, 2],
                 [1, 1]]
num = 4

start_evaluations = 0
end_evaluations =   50000
step_num = 100
solution_num = 12
save_point = np.linspace(int((end_evaluations - start_evaluations) / step_num + start_evaluations), end_evaluations, step_num)

def generate_rdm():


    # generate rdm pos and ori before the knolling
    collect_ori = []
    collect_pos = []
    restrict = np.max(self.xyz_list)
    gripper_height = 0.012
    last_pos = np.array([[0, 0, 1]])
    for i in range(len(self.xyz_list)):
        rdm_pos = np.array([random.uniform(self.x_low_obs, self.x_high_obs),
                            random.uniform(self.y_low_obs, self.y_high_obs), 0.0])
        ori = [0, 0, random.uniform(0, np.pi)]
        # ori = [0, 0, 0]
        collect_ori.append(ori)
        check_list = np.zeros(last_pos.shape[0])

        while 0 in check_list:
            rdm_pos = [random.uniform(self.x_low_obs, self.x_high_obs),
                       random.uniform(self.y_low_obs, self.y_high_obs), 0.0]
            for z in range(last_pos.shape[0]):
                if np.linalg.norm(last_pos[z] - rdm_pos) < restrict + gripper_height:
                    check_list[z] = 0
                else:
                    check_list[z] = 1
        collect_pos.append(rdm_pos)

        last_pos = np.append(last_pos, [rdm_pos], axis=0)
    collect_pos = np.asarray(collect_pos)[:, :2]
    collect_ori = np.asarray(collect_ori)[:, 2]
    # generate rdm pos and ori before the knolling

def merge(): # after that, the structure of dataset is cfg0_0, cfg0_1, cfg0_2,
                                                     # cfg1_0, cfg1_1, cfg1_2,
                                                     # cfg2_0, cfg2_1, cfg2_2
    info_per_object = 7
    for m in tqdm(range(solution_num)):

        target_path = '../../../knolling_dataset/VAE_329_obj4/'
        after_path = target_path + 'labels_after_%s/' % m
        output_path = target_path + 'num_%d_after_%d.txt' % (num, m)
        output_name_path = target_path + 'num_%d_after_name_%d.txt' % (num, m)

        total_data = []
        total_data_name = []
        for s in save_point:
            data = np.loadtxt(after_path + 'num_%d_%d.txt' % (num, int(s)))
            data_name = np.loadtxt(after_path + 'num_%d_%d_name.txt' % (num, int(s)), dtype=str)
            total_data.append(data)
            total_data_name.append(data_name)
        total_data = np.asarray(total_data).reshape(-1, num * info_per_object)
        total_data_name = np.asarray(total_data_name, dtype=str).reshape(-1, num)

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
    base_path = '../../../knolling_dataset/learning_data_817/'
    add_path = '../../knolling_dataset/learning_data_817_add/'
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

def add_noise():

    for m in range(solution_num):

        target_path = '../../../knolling_dataset/learning_data_1013/'
        after_path = target_path + 'labels_after_%s/' % m

        raw_data = np.loadtxt(after_path + 'num_5.txt')
        noise_mask = np.random.rand(5, 2) * 0.005

        new_data = []
        for i in range(len(raw_data)):
            one_img_data = raw_data[i].reshape(5, -1)
            one_img_data[:, 2:4] += noise_mask
            new_data.append(one_img_data.reshape(-1, ))
        new_data = np.asarray(new_data)
        np.savetxt(after_path + 'num_5_new.txt', new_data, fmt='%.05f')
        pass

def tuning():

    num_object_per_scenario = 10
    info_per_object = 8

    for m in tqdm(range(12)):

        target_path = '../../../knolling_dataset/learning_data_0126/'
        output_path = target_path + 'num_%d_after_%d.txt' % (num, m)
        output_name_path = target_path + 'num_%d_after_name_%d.txt' % (num, m)

        data = np.loadtxt(target_path + 'num_%d_after_%d.txt' % (num, m)).reshape(-1, num_object_per_scenario, info_per_object)
        output_data = np.delete(data, [2], axis=2).reshape(data.shape[0], -1)

        np.savetxt(output_path, output_data)

def manual_padding():

    info_per_object = 7
    max_seq_length = 10
    num_per_config = 1000
    num_list = np.random.choice(np.arange(4, max_seq_length + 1), num_per_config)
    for m in tqdm(range(solution_num)):

        target_path = '../../../knolling_dataset/learning_data_0131/'
        raw_data = np.loadtxt(target_path + 'num_%d_after_%d.txt' % (num, m))
        raw_name = np.loadtxt(target_path + 'num_%d_after_name_%d.txt' % (num, m), dtype=str)

        mask_data = ~(np.arange(max_seq_length * info_per_object) < (num_list * info_per_object)[:, None])
        raw_data[mask_data] = 0
        mask_name = ~(np.arange(max_seq_length) < (num_list)[:, None])
        raw_name[mask_name] = 'Null'

        output_path = target_path + 'num_%d_after_%d_mask.txt' % (num, m)
        output_name_path = target_path + 'num_%d_after_name_%d_mask.txt' % (num, m)

        np.savetxt(output_path, raw_data)
        np.savetxt(output_name_path, raw_name, fmt='%s')

merge()
# manual_padding()
# tuning()
# merge_test()
# add()
# add_noise()