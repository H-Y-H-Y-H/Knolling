import numpy as np
import pybullet_data as pd
import random
import pybullet as p
import os
import cv2
import torch
from tqdm import tqdm
# from urdfpy import URDF
import shutil
import json
import csv
import pandas
import glob,time


class Collection_env:
    def __init__(self, is_render, arrange_policy):

        self.kImageSize = {'width': 480, 'height': 480}
        self.urdf_path = '../ASSET/urdf/'
        self.dataset_path = '../ASSET/sundry/'
        self.pybullet_path = pd.getDataPath()
        self.is_render = is_render
        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.num_motor = 5

        self.low_scale = np.array([0.03, -0.14, 0.0, - np.pi / 2, 0])
        self.high_scale = np.array([0.27, 0.14, 0.05, np.pi / 2, 0.4])
        self.low_act = -np.ones(5)
        self.high_act = np.ones(5)
        self.x_low_obs = self.low_scale[0]
        self.x_high_obs = self.high_scale[0]
        self.y_low_obs = self.low_scale[1]
        self.y_high_obs = self.high_scale[1]
        self.z_low_obs = self.low_scale[2]
        self.z_high_obs = self.high_scale[2]
        self.table_boundary = 0.03
        self.total_offset = arrange_policy['total_offset']

        self.lateral_friction = 1
        self.spinning_friction = 1
        self.rolling_friction = 0

        self.camera_parameters = {
            'width': 640.,
            'height': 480,
            'fov': 42,
            'near': 0.1,
            'far': 100.,
            'eye_position': [0.59, 0, 0.8],
            'target_position': [0.55, 0, 0.05],
            'camera_up_vector':
                [1, 0, 0],  # I really do not know the parameter's effect.
            'light_direction': [
                0.5, 0, 1
            ],  # the direction is from the light source position to the origin of the world frame.
        }
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.15, 0, 0],
            distance=0.4,
            yaw=90,
            pitch=-90,
            roll=0,
            upAxisIndex=2)
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_parameters['fov'],
            aspect=self.camera_parameters['width'] / self.camera_parameters['height'],
            nearVal=self.camera_parameters['near'],
            farVal=self.camera_parameters['far'])

        if random.uniform(0, 1) > 0.5:
            p.configureDebugVisualizer(lightPosition=[random.randint(1, 3), random.randint(1, 2), 5])
        else:
            p.configureDebugVisualizer(lightPosition=[random.randint(1, 3), random.randint(-2, -1), 5])
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        # p.configureDebugVisualizer(lightPosition=[random.randint(1, 3), random.randint(1, 2), 5],
        #                            shadowMapResolution=8192, shadowMapIntensity=np.random.randint(5, 8) / 10)
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0])
        p.setAdditionalSearchPath(pd.getDataPath())

        self.arrange_policy = arrange_policy
        # Set up the scene (walls, debug lines, plane, etc.)
        self.setup_scene()

    def setup_scene(self):
        """
        Set up static objects: walls, debug lines, and the plane.
        This code is called in __init__ and after every simulation reset.
        """
        # --- Add walls ---
        wall_id = []
        wall_pos = np.array([
            [self.x_low_obs - self.table_boundary, 0, 0],
            [(self.x_low_obs + self.x_high_obs) / 2, self.y_low_obs - self.table_boundary, 0],
            [self.x_high_obs + self.table_boundary, 0, 0],
            [(self.x_low_obs + self.x_high_obs) / 2, self.y_high_obs + self.table_boundary, 0]
        ])
        wall_ori = np.array([
            [0, 1.57, 0],
            [0, 1.57, 1.57],
            [0, 1.57, 0],
            [0, 1.57, 1.57]
        ])
        for i in range(len(wall_pos)):
            wall_id.append(p.loadURDF(os.path.join(self.urdf_path, "wall.urdf"),
                                      basePosition=wall_pos[i],
                                      baseOrientation=p.getQuaternionFromEuler(wall_ori[i]),
                                      useFixedBase=1,
                                      flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
            p.changeVisualShape(wall_id[i], -1, rgbaColor=(1, 1, 1, 0))

        # --- Set gravity ---
        p.setGravity(0, 0, -10)

        # # --- Draw workspace lines ---
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
        #     lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
        #     lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
        #     lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
        #     lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])

        # --- Load the plane ---
        self.baseid = p.loadURDF(self.urdf_path + "plane.urdf",
                                 useFixedBase=1,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        p.changeDynamics(self.baseid, -1,
                         lateralFriction=1,
                         spinningFriction=1,
                         rollingFriction=0.002,
                         linearDamping=0.5,
                         angularDamping=0.5)


    def change_sequence(self, pos, flag=None):
        if flag == 'distance':
            origin_point = np.array([0, 0])
            delete_index = np.where(pos == 0)[0]
            distance = np.linalg.norm(pos[:, :2] - origin_point, axis=1)
            order = np.argsort(distance)
        if flag == 'area':
            order = 1
        if flag == 'length':
            order = 1
        return order

    def get_data_virtual(self):

        # shuffle the class_num and color_num for each scenario
        self.arrange_policy['class_num'] = np.random.randint(2, self.arrange_policy['max_class_num'] + 1)
        self.arrange_policy['color_num'] = np.random.randint(2, self.arrange_policy['max_color_num'] + 1)

        if self.arrange_policy['object_type'] == 'box':
            length_data = np.round(np.random.uniform(low=self.arrange_policy['length_range'][0],
                                                     high=self.arrange_policy['length_range'][1],
                                                     size=(self.arrange_policy['object_num'], 1)), decimals=3)
            width_data = np.round(np.random.uniform(low=self.arrange_policy['width_range'][0],
                                                    high=self.arrange_policy['width_range'][1],
                                                    size=(self.arrange_policy['object_num'], 1)), decimals=3)
            height_data = np.round(np.random.uniform(low=self.arrange_policy['height_range'][0],
                                                     high=self.arrange_policy['height_range'][1],
                                                     size=(self.arrange_policy['object_num'], 1)), decimals=3)
            class_data = np.random.randint(low=0,
                                           high=self.arrange_policy['class_num'],
                                           size=(self.arrange_policy['object_num'], 1))
            color_index_data = np.random.randint(low=0,
                                                 high=self.arrange_policy['color_num'],
                                                 size=(self.arrange_policy['object_num'], 1))
            data = np.concatenate((length_data, width_data, height_data, class_data, color_index_data), axis=1).round(
                decimals=3)
            return data, None

        elif self.arrange_policy['object_type'] == 'sundry':
            class_index = np.random.choice(a=self.arrange_policy['max_class_num'],
                                           size=self.arrange_policy['class_num'], replace=False)
            class_index_data = np.random.choice(a=class_index, size=self.arrange_policy['object_num'])
            class_name_list = os.listdir(self.dataset_path + 'generated_stl/')
            class_name_list.sort()
            class_name = [class_name_list[n] for n in class_index_data]
            self.class_name_test = class_name
            object_name_list = []
            object_lwh_list = []
            for i in range(self.arrange_policy['object_num']):
                object_path = self.dataset_path + 'generated_stl/' + class_name[i] + '/'
                obj_list = glob.glob(object_path + '*.csv')
                obj_list = [path.replace('\\', '/') for path in obj_list]
                candidate_object = np.random.choice(obj_list).split('/')[-1][:-4]

                object_csv_path = object_path + candidate_object + '.csv'
                object_name_list.append(candidate_object)
                object_lwh_list.append(pandas.read_csv(object_csv_path).iloc[0, [3, 4, 5]].values)

            color_index = np.random.choice(a=self.arrange_policy['max_color_num'],
                                           size=self.arrange_policy['color_num'],
                                           replace=False)
            color_index_data = np.random.choice(a=color_index, size=self.arrange_policy['object_num'])

            object_lwh_list = np.around(np.asarray(object_lwh_list) * 0.001, decimals=4)
            data = np.concatenate((object_lwh_list,
                                   class_index_data.reshape(self.arrange_policy['object_num'], 1),
                                   color_index_data.reshape(self.arrange_policy['object_num'], 1)), axis=1)
            object_name_list = np.asarray(object_name_list)

            return data, object_name_list

    def get_obs(self, order, evaluation):
        def get_images():
            (width, length, image, _, _) = p.getCameraImage(width=640,
                                                            height=480,
                                                            viewMatrix=self.view_matrix,
                                                            projectionMatrix=self.projection_matrix,
                                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
            return image

        if order == 'images':
            image = get_images()
            temp = np.copy(image[:, :, 0])
            image[:, :, 0] = image[:, :, 2]
            image[:, :, 2] = temp
            return image

    def is_too_close(self, new_pos, existing_objects, min_distance=0.1):
        for obj_pos in existing_objects:
            if np.linalg.norm(np.array(new_pos) - np.array(obj_pos)) < min_distance:
                return True
        return False

    def label2image(self, labels_data, labels_name=None):

        random_ground_img = np.random.randint(0,100)
        textureId = p.loadTexture(self.urdf_path + f"temp/aug_floor_{random_ground_img}.png")


        p.changeVisualShape(self.baseid, -1, textureUniqueId=textureId,
                            rgbaColor=[np.random.uniform(0.9, 1), np.random.uniform(0.9, 1), np.random.uniform(0.9, 1), 1])

        # print(index_flag)
        # index_flag = index_flag.reshape(2, -1)

        labels_data = labels_data.reshape(-1, 7)
        # labels_data = labels_data[:np.random.randint(low=4, high=len(labels_data)-2), :]
        obj_num = np.sum(np.any(labels_data, axis=1))

        pos_data = np.concatenate((labels_data[:, :2], np.ones((len(labels_data), 1)) * 0.003), axis=1)
        pos_data[:, 0] += self.total_offset[0]
        pos_data[:, 1] += self.total_offset[1]
        lw_data = labels_data[:, 2:5]
        # ori_data = labels_data[:, 3:6]
        ori_data = np.zeros((len(lw_data), 3))
        color_index = labels_data[:, -1]
        class_index = labels_data[:, -2]

        # Converting dictionary keys to integers
        dict_map = {i: v for i, (k, v) in enumerate(color_dict.items())}

        # Mapping array values to dictionary values
        rdm_color_index = np.random.choice(10, len(color_index))
        mapped_color_values = []
        for i in range(len(color_index)):
            mapped_color_values.append(dict_map[color_index[i]][rdm_color_index[i]])
        # mapped_color_values = [dict_map[value] for value in color_index]

        object_idx = []
        for i in range(obj_num):
            object_path = self.dataset_path + 'generated_stl/' + labels_name[i][:-2] + '/' + labels_name[i]
            object_csv = object_path + '.csv'

            csv_lwh = np.asarray(pandas.read_csv(object_csv).iloc[0, [3, 4, 5]].values) * 0.001
            pos_data[i, 2] = csv_lwh[2] / 2

            if lw_data[i, 0] < lw_data[i, 1]:
                ori_data[i, 2] += np.pi / 2
            obj_id = p.loadURDF(self.dataset_path + 'urdf_file/' + labels_name[i] + '.urdf',
                                         basePosition=pos_data[i],
                                         baseOrientation=p.getQuaternionFromEuler(ori_data[i]),
                                         useFixedBase=False,
                                         flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
            object_idx.append(obj_id)
            p.changeVisualShape(obj_id, -1, rgbaColor=mapped_color_values[i] + [1])

        neat_img = self.get_obs('images', None)

        self.init_ori_range = [[0, 0], [0, 0], [-np.pi / 4, np.pi / 4]]
        self.init_pos_range = [[0.03, 0.27], [-0.15, 0.15], [0.01, 0.01]]



        while True:

            for i in object_idx:
                p.removeBody(i)

            object_idx = []
            rdm_obj_pos_list = []
            rdm_obj_ori_list = []
            data_pass = False
            test_num = 0

            for i in range(obj_num):
                # Initial random position and orientation
                pos_data, ori_data = self.randomize_position_and_orientation()

                object_path = self.dataset_path + 'generated_stl/' + labels_name[i][:-2] + '/' + labels_name[i]
                object_csv = object_path + '.csv'
                csv_lwh = np.asarray(pandas.read_csv(object_csv).iloc[0, [3, 4, 5]].values) * 0.001

                while 1:
                    # Check if the new position is too close to any existing objects
                    if not self.is_too_close(pos_data[:2], rdm_obj_pos_list):
                        pos_data[2] = csv_lwh[2] / 2
                        obj_i =p.loadURDF(self.dataset_path + 'urdf_file/' + labels_name[i] + '.urdf',
                                        basePosition=pos_data,
                                        baseOrientation=p.getQuaternionFromEuler(ori_data), useFixedBase=False,
                                        flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

                        object_idx.append(obj_i)
                        rdm_obj_pos_list.append(pos_data[:2])
                        rdm_obj_ori_list.append([ori_data[2]])

                        p.changeVisualShape(obj_i, -1,rgbaColor=mapped_color_values[i] + [1])
                        data_pass = True
                        break
                    else:
                        pos_data, ori_data = self.randomize_position_and_orientation()
                        test_num += 1

                    if test_num == max_test_times:
                        print(test_num)
                        data_pass = False
                        break
                if (data_pass == False) and test_num == max_test_times:
                    break
            if data_pass:
                break
        # shutil.rmtree(save_urdf_path_one_img)
        for i in range(20):
            p.stepSimulation()
        rdm_img = self.get_obs('images', None)

        for i in object_idx:
            p.removeBody(i)

        # ---------------------------
        # Prepare a label array for the messy configuration.
        # Here we save:
        #  - Columns 0-2: randomized (x, y)
        #  - Columns 3-5: randomized (yaw)
        #  - Columns 6-8: original object dimensions (l, w, h) from tidy labels
        #  - Column 9: class index; Column 10: color index
        messy_label_data = np.hstack([
            np.array(rdm_obj_pos_list),  # (obj_num, 2)
            np.array(rdm_obj_ori_list),  # (obj_num, 1)
            lw_data[:obj_num],  # (obj_num, 3)
            class_index[:obj_num].reshape(-1, 1),  # (obj_num, 1)
            color_index[:obj_num].reshape(-1, 1)  # (obj_num, 1)
        ])
        messy_label_data = messy_label_data.flatten()

        p.resetSimulation()
        self.setup_scene()


        return rdm_img, neat_img, messy_label_data

    def randomize_position_and_orientation(self):
        """
        Randomizes position and orientation within defined ranges.
        Returns:
            pos_data (list): Randomized [x, y, z] position.
            ori_data (list): Randomized [roll, pitch, yaw] orientation.
        """
        rdm_pos_x = np.random.uniform(self.init_pos_range[0][0], self.init_pos_range[0][1])
        rdm_pos_y = np.random.uniform(self.init_pos_range[1][0], self.init_pos_range[1][1])
        rdm_pos_z = np.random.uniform(self.init_pos_range[2][0], self.init_pos_range[2][1])
        pos_data = [rdm_pos_x, rdm_pos_y, rdm_pos_z]

        rdm_ori_roll = np.random.uniform(self.init_ori_range[0][0], self.init_ori_range[0][1])
        rdm_ori_pitch = np.random.uniform(self.init_ori_range[1][0], self.init_ori_range[1][1])
        rdm_ori_yaw = np.random.uniform(self.init_ori_range[2][0], self.init_ori_range[2][1])
        ori_data = [rdm_ori_roll, rdm_ori_pitch, rdm_ori_yaw]

        return pos_data, ori_data




if __name__ == '__main__':


    for obj_num in range(9,10):
        each_run_data_num = 2000

        multi_threads_id = 0

        start_evaluations = each_run_data_num * multi_threads_id
        # start_evaluations = 800
        # multi_threads_id = start_evaluations//200

        end_evaluations   = each_run_data_num * (multi_threads_id + 1)

        print(multi_threads_id,start_evaluations,end_evaluations)

        target_path = f'C:/Users/yuhan/PycharmProjects/Knolling_data/dataset_texture/obj{obj_num}/'
        os.makedirs(target_path, exist_ok=True)

        arrange_policy = {
            'length_range': [0.036, 0.06], 'width_range': [0.016, 0.036], 'height_range': [0.01, 0.02],
            'object_num': obj_num, 'output_per_cfg': 3, 'object_type': 'sundry',  # sundry or box
            'iteration_time': 10,
            'x_max': 0.27, 'y_max': 0.28,  # Define maximum allowable x-coordinate
            'area_num': None, 'ratio_num': None, 'area_classify_flag': None, 'ratio_classify_flag': None,
            'class_num': None, 'color_num': None, 'max_class_num': 9, 'max_color_num': 6,
            'type_classify_flag': None, 'color_classify_flag': None,  # classification range
            'arrangement_policy': 'Type*3, Color*3, Area*3, Ratio*3',  # customized setting
            'object_even': True, 'block_even': True, 'upper_left_max': False, 'forced_rotate_box': False,
            'total_offset': [0.016, -0.20 + 0.016, 0], 'gap_item': 0.016, 'gap_block': 0.016  # inverval and offset of the arrangement
        }

        solution_num = 12
        max_test_times = 10000
        os.makedirs(target_path + 'messy', exist_ok=True)
        os.makedirs(target_path + 'tidy', exist_ok=True)
        env = Collection_env(is_render=False, arrange_policy=arrange_policy)
        # Turn on rendering may cause the simulation achieve maximum object loading.

        with open('../ASSET/urdf/object_color/rgb_info.json') as f:
            color_dict = json.load(f)
        names = locals()

        save_urdf_path = []
        for m in range(solution_num):
            names['data_' + str(m)] = np.loadtxt(target_path+ f'{obj_num}obj_tidy_data_cdn{m}.txt')
            names['name_' + str(m)] = np.loadtxt(target_path + f'{obj_num}obj_tidy_name_cdn{m}.txt', dtype=str)


        messy_data_all = {m: [] for m in range(solution_num)}

        for j in range(start_evaluations, end_evaluations):
            for m in range(solution_num):
                print(f'this is results {j}, {m}')
                one_img_data = names['data_' + str(m)][j].reshape(-1, 7)

                rdm_img, neat_img, messy_label_data  = env.label2image(names['data_' + str(m)][j],labels_name=names['name_' + str(m)][j])

                rdm_img = rdm_img[..., :3]
                neat_img = neat_img[..., :3]

                cv2.imwrite(target_path + 'messy/%d_%d.png' % (j, m), rdm_img)
                cv2.imwrite(target_path + 'tidy/%d_%d.png' % (j, m), neat_img)

                # Accumulate the messy label row.
                messy_data_all[m].append(messy_label_data)

        # After processing all evaluations, save one text file per solution.
        for m in range(solution_num):
            messy_data_array = np.array(messy_data_all[m])
            np.savetxt(target_path + f'{obj_num}obj_messy_data_cdn{m}.txt', messy_data_array, fmt='%.4f')
