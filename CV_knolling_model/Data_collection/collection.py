import csv

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
import glob

from arrange_policy import arrangement

torch.manual_seed(42)

class Collection_env:

    def __init__(self, is_render, arrange_policy,total_offset):

        self.kImageSize = {'width': 480, 'height': 480}
        self.urdf_path = '../../ASSET/urdf/'
        self.dataset_path = '../../ASSET/sundry/'
        # self.obj_urdf = '../../../knolling_dataset/'
        self.pybullet_path = pd.getDataPath()
        self.is_render = is_render
        if self.is_render:
            # p.connect(p.GUI, options="--width=1280 --height=720")
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
        self.total_offset = total_offset


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
            data = np.concatenate((length_data, width_data, height_data, class_data, color_index_data), axis=1).round(decimals=3)
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
        for _, obj_pos in existing_objects:
            if np.linalg.norm(np.array(new_pos) - np.array(obj_pos)) < min_distance:
                return True
        return False

    def label2image(self, labels_data, labels_name=None):

        rdm_img = np.zeros((480, 640, 4))
        neat_img = np.zeros((480, 640, 4))

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

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Draw workspace lines
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])

        baseid = p.loadURDF(self.urdf_path + "plane.urdf", useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture(self.urdf_path + "floor_white.png")
        # p.changeVisualShape(baseid, -1, textureUniqueId=textureId,
        #                     rgbaColor=[np.random.uniform(0.9, 1), np.random.uniform(0.9, 1), np.random.uniform(0.9, 1), 1])
        p.changeDynamics(baseid, -1, lateralFriction=1, spinningFriction=1, rollingFriction=0.002, linearDamping=0.5,
                         angularDamping=0.5)

        self.state_blank = p.saveState()

        if self.arrange_policy['object_type'] == 'sundry':
            object_idx = []

            if before_after == 'after' or before_after == 'before_after':
                for i in range(obj_num):
                    object_path = self.dataset_path + 'generated_stl/' + labels_name[i][:-2] + '/' + labels_name[i]
                    object_csv = object_path + '.csv'

                    csv_lwh = np.asarray(pandas.read_csv(object_csv).iloc[0, [3, 4, 5]].values) * 0.001
                    pos_data[i, 2] = csv_lwh[2] / 2

                    if lw_data[i, 0] < lw_data[i, 1]:
                        ori_data[i, 2] += np.pi / 2
                    object_idx.append(p.loadURDF(self.dataset_path + 'urdf_file/' + labels_name[i] + '.urdf',
                                                 basePosition=pos_data[i],
                                                 baseOrientation=p.getQuaternionFromEuler(ori_data[i]), useFixedBase=False,
                                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
                    p.changeVisualShape(p.getBodyUniqueId(i+1), -1, rgbaColor=mapped_color_values[i] + [1])

                neat_img = self.get_obs('images', None)
        ################### recover urdf boxes based on lw_data ###################

            if before_after == 'before' or before_after == 'before_after':
                # p.restoreState(self.state_blank)
                self.init_ori_range = [[0, 0], [0, 0], [-np.pi / 4, np.pi / 4]]
                rdm_ori_roll = np.random.uniform(self.init_ori_range[0][0], self.init_ori_range[0][1],
                                                 size=(obj_num, 1))
                rdm_ori_pitch = np.random.uniform(self.init_ori_range[1][0], self.init_ori_range[1][1],
                                                  size=(obj_num, 1))
                rdm_ori_yaw = np.random.uniform(self.init_ori_range[2][0], self.init_ori_range[2][1],
                                                size=(obj_num, 1))
                rdm_ori = np.hstack([rdm_ori_roll, rdm_ori_pitch, rdm_ori_yaw])
                self.init_pos_range = [[0.03, 0.27], [-0.15, 0.15], [0.01, 0.01]]
                for i in object_idx:
                    p.removeBody(i)
                object_idx = []

                # add the wall while generating rdm positions
                wall_id = []
                wall_pos = np.array([[self.x_low_obs - self.table_boundary, 0, 0],
                                     [(self.x_low_obs + self.x_high_obs) / 2, self.y_low_obs - self.table_boundary, 0],
                                     [self.x_high_obs + self.table_boundary, 0, 0],
                                     [(self.x_low_obs + self.x_high_obs) / 2, self.y_high_obs + self.table_boundary, 0]])
                wall_ori = np.array([[0, 1.57, 0],
                                     [0, 1.57, 1.57],
                                     [0, 1.57, 0],
                                     [0, 1.57, 1.57]])
                for i in range(len(wall_pos)):
                    wall_id.append(p.loadURDF(os.path.join(self.urdf_path, "wall.urdf"), basePosition=wall_pos[i],
                                              baseOrientation=p.getQuaternionFromEuler(wall_ori[i]), useFixedBase=1,
                                              flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
                    p.changeVisualShape(wall_id[i], -1, rgbaColor=(1, 1, 1, 0))

                for i in range(obj_num):

                    rdm_pos_x = np.random.uniform(self.init_pos_range[0][0], self.init_pos_range[0][1])
                    rdm_pos_y = np.random.uniform(self.init_pos_range[1][0], self.init_pos_range[1][1])
                    rdm_pos_z = np.random.uniform(self.init_pos_range[2][0], self.init_pos_range[2][1])
                    pos_data = [rdm_pos_x, rdm_pos_y, rdm_pos_z]

                    object_path = self.dataset_path + 'generated_stl/' + labels_name[i][:-2] + '/' + labels_name[i]
                    object_csv = object_path + '.csv'

                    # print(f'this is matching urdf{i}')
                    placement_successful = False
                    while not placement_successful:

                        # Check if the new position is too close to any existing objects
                        if not self.is_too_close(pos_data, object_idx):
                            csv_lwh = np.asarray(pandas.read_csv(object_csv).iloc[0, [3, 4, 5]].values) * 0.001
                            pos_data[2] = csv_lwh[2] / 2

                            object_idx.append((p.loadURDF(self.dataset_path + 'urdf_file/' + labels_name[i] + '.urdf',
                                                         basePosition=pos_data,
                                                         baseOrientation=p.getQuaternionFromEuler(rdm_ori[i]), useFixedBase=False,
                                                         flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT), pos_data))
                            p.changeVisualShape(p.getBodyUniqueId(i+5), -1, rgbaColor=mapped_color_values[i] + [1])
                            placement_successful = True
                        else:
                            rdm_pos_x = np.random.uniform(self.init_pos_range[0][0], self.init_pos_range[0][1])
                            rdm_pos_y = np.random.uniform(self.init_pos_range[1][0], self.init_pos_range[1][1])
                            rdm_pos_z = np.random.uniform(self.init_pos_range[2][0], self.init_pos_range[2][1])
                            pos_data = [rdm_pos_x, rdm_pos_y, rdm_pos_z]

                # shutil.rmtree(save_urdf_path_one_img)
                for i in range(20):
                    p.stepSimulation()
                rdm_img = self.get_obs('images', None)

        # return self.get_obs('images', None)
        # return np.concatenate((rdm_img, neat_img), axis=1)
        return rdm_img, neat_img

    def change_config(self):  # this is the main function!!!!!!!!!

        # get the standard xyz and corresponding index from files in the computer
        arranger = arrangement(self.arrange_policy)
        data_before, object_name_list = self.get_data_virtual()
        data_after_distance_total = []
        name_after_distance_total = []
        data_after_default_total = []
        name_after_default_total = []

        # type, color, area, ratio
        for i in range(len(policy_switch)):
            arranger.arrange_policy['type_classify_flag'] = policy_switch[i][0]
            arranger.arrange_policy['color_classify_flag'] = policy_switch[i][1]
            arranger.arrange_policy['area_classify_flag'] = policy_switch[i][2]
            arranger.arrange_policy['ratio_classify_flag'] = policy_switch[i][3]

            times = 0
            while True: # generate several results for each configuration, now is 4
                data_after, name_after = arranger.generate_arrangement(data=data_before, data_name=object_name_list)
                # the sequence of self.items_pos_list, self.items_ori_list are the same as those in self.xyz_list

                data_after[:, :3] = data_after[:, :3] + self.arrange_policy['total_offset']
                order = self.change_sequence(data_after[:, :2], flag='distance')
                data_after = np.delete(data_after, [2, 3, 4, 5], axis=1)

                temp_criterion = np.argsort(data_after[:, 5])
                data_after_default = data_after[temp_criterion]
                data_after_distance = data_after[order]
                data_after_default_total.append(data_after_default)
                data_after_distance_total.append(data_after_distance)

                name_after_default = name_after[temp_criterion]
                name_after_distance = name_after[order]
                name_after_default_total.append(name_after_default)
                name_after_distance_total.append(name_after_distance)

                times += 1
                if times >= self.arrange_policy['output_per_cfg']:
                    break

        return data_after_distance_total, name_after_distance_total, data_after_default_total, name_after_default_total


if __name__ == '__main__':

    #- run ```CV_knolling_model/collection.py``` with ```command``` equal to ```collection```,
    # to generate all txt data of objects in the neat/random arrangement.

    # you can run multiple instances of the program to speed up the data collection
    # - run ```merge_txt``` function in ```CV_knolling_model/preprocess.py```
    # to merge txt data of different instances into one

    # - run ```CV_knolling_model/collection.py``` with ```command``` equal to ```recover```,
    # to convert txt data of the neat/random arrangement to image data of it.

    # you can run multiple instances of the program to speed up the data collection

    # command = 'collection'
    command = 'recover'

    for before_after in ['after','before']:

        obj_num = 10
        SHIFT_DATASET_ID = 0
        print(SHIFT_DATASET_ID)
        total_offset = [0.016, -0.20 + 0.016, 0]

        step_num = 10
        num_each_step = 20

        start_evaluations = step_num*num_each_step* SHIFT_DATASET_ID
        end_evaluations   = step_num*num_each_step*(SHIFT_DATASET_ID+1)

        save_point = np.linspace(num_each_step+start_evaluations, end_evaluations, step_num, dtype=int)

        target_path = f'C:/Users/yuhan/PycharmProjects/Knolling_data/dataset/VAE_1118_obj{obj_num}/'
        os.makedirs(target_path, exist_ok=True)

        # target_path = f'../../dataset/VAE_1020_obj{obj_num}/'

        arrange_policy = {
                        'length_range': [0.036, 0.06], 'width_range': [0.016, 0.036], 'height_range': [0.01, 0.02], # objects 3d range
                        'object_num': obj_num, 'output_per_cfg': 3, 'object_type': 'sundry', # sundry or box
                        'iteration_time': 10,
                        'x_max': 0.27, 'y_max': 0.28,  # Define maximum allowable x-coordinate
                        'area_num': None, 'ratio_num': None, 'area_classify_flag': None, 'ratio_classify_flag': None,
                        'class_num': None, 'color_num': None, 'max_class_num': 9, 'max_color_num': 7,
                        'type_classify_flag': None, 'color_classify_flag': None, # classification range
                        'arrangement_policy': 'Type*3, Color*3, Area*3, Ratio*3', # customized setting
                        'object_even': True, 'block_even': True, 'upper_left_max': False, 'forced_rotate_box': False,
                        'total_offset': [0, 0, 0], 'gap_item': 0.016, 'gap_block': 0.016 # inverval and offset of the arrangement
                        }

        policy_switch = [[True, False, False, False],
                         [False, True, False, False],
                         [False, False, True, False],
                         [False, False, False, True]]

        solution_num = int(arrange_policy['output_per_cfg'] * len(policy_switch))

        if command == 'collection': # save the parameters of results collection
            with open(target_path[:-1] + "_readme.json", "w") as f:
                json.dump(arrange_policy, f, indent=4)

            env = Collection_env(is_render=False, arrange_policy=arrange_policy, total_offset=total_offset)
            after_path = []
            names = locals()
            for m_id in range(solution_num):
                after_path.append(target_path + 'labels_after_%s/' % m_id)
                os.makedirs(after_path[m_id], exist_ok=True)
                names['data_after_distance_' + str(m_id)] = []
                names['name_after_distance_' + str(m_id)] = []

            for j in range(num_each_step*step_num):
                index_point = j // num_each_step
                save_id = save_point[index_point]

                while True:
                    # Generate arrangements
                    data_after_distance_total, name_after_distance_total, data_after_default_total, name_after_default_total = env.change_config()

                    # Check if all objects are within boundaries
                    is_within_boundaries = True
                    for data_after_distance in data_after_distance_total:
                        # Extract positions and dimensions
                        positions = data_after_distance[:, :2]  # Assuming first two columns are x, y positions
                        dimensions = data_after_distance[:, 2:4]  # Assuming next two columns are length, width

                        # Find the object with the maximum x and y positions
                        max_x_index = np.argmax(positions[:, 0])  # Index of the object with max x
                        max_y_index = np.argmax(positions[:, 1])  # Index of the object with max y

                        # Calculate boundaries
                        max_x_end = positions[max_x_index, 0] + (
                                    dimensions[max_x_index, 1] / 2)  # max x position + half width
                        max_y_end = positions[max_y_index, 1] + (
                                    dimensions[max_y_index, 0] / 2)  # max y position + half length

                        # Check if these exceed the boundaries
                        if max_x_end > arrange_policy['x_max'] or max_y_end > arrange_policy['y_max']:
                            is_within_boundaries = False

                            break

                    if is_within_boundaries:
                        print('succ')
                        break  # Exit the loop if arrangement is valid



                for m in range(solution_num):
                    names['data_after_distance_' + str(m)].append(data_after_distance_total[m].reshape(-1))
                    names['name_after_distance_' + str(m)].append(name_after_distance_total[m])

                    if len(names['data_after_distance_' + str(m)]) == num_each_step:
                        names['data_after_distance_' + str(m)] = np.asarray(names['data_after_distance_' + str(m)])
                        names['name_after_distance_' + str(m)] = np.asarray(names['name_after_distance_' + str(m)], dtype=str)

                        np.savetxt(after_path[m] + 'num_%s_%s.txt' % (arrange_policy['object_num'], save_id),
                                   names['data_after_distance_' + str(m)])
                        np.savetxt(after_path[m] + 'num_%s_%s_name.txt' % (arrange_policy['object_num'], save_id),
                                   names['name_after_distance_' + str(m)], fmt='%s')

                        names['data_after_distance_' + str(m)] = []
                        names['name_after_distance_' + str(m)] = []
                        print('save results in:' + after_path[m] + 'num_%s_%s.txt' % (arrange_policy['object_num'], save_id))

        if command == 'recover':
            os.makedirs(target_path + 'origin_images_before/', exist_ok=True)
            os.makedirs(target_path + 'origin_images_after/', exist_ok=True)

            env = Collection_env(is_render=False, arrange_policy=arrange_policy, total_offset = total_offset)
            with open('../../ASSET/urdf/object_color/rgb_info.json') as f:
                color_dict = json.load(f)
            names = locals()
            # data_before = []
            save_urdf_path = []
            for m in range(solution_num):
                print('load results')
                names['data_' + str(m)] = np.loadtxt(target_path + 'num_%d_after_%d.txt' % (arrange_policy['object_num'], m))
                if arrange_policy['object_type'] == 'sundry':
                    names['name_' + str(m)] = np.loadtxt(target_path + 'num_%d_after_name_%d.txt' % (arrange_policy['object_num'], m), dtype=str)

                if len(names['data_' + str(m)].shape) == 1:
                    names['data_' + str(m)] = names['data_' + str(m)].reshape(1, len(names['data_' + str(m)]))

                box_num = arrange_policy['object_num']
                print('this is len results', len(names['data_' + str(m)]))

            for j in range(start_evaluations, end_evaluations):
                for m in range(solution_num):
                    print(f'this is results {j}')
                    one_img_data = names['data_' + str(m)][j].reshape(-1, 7)

                    rdm_img, neat_img = env.label2image(names['data_' + str(m)][j], labels_name=names['name_' + str(m)][j])
                    rdm_img = rdm_img[..., :3]
                    neat_img = neat_img[..., :3]

                    if before_after == 'before' or before_after == 'before_after':
                        cv2.imwrite(target_path + 'origin_images_before/label_%d_%d.png' % (j, m), rdm_img)
                    if before_after == 'after' or before_after == 'before_after':
                        cv2.imwrite(target_path + 'origin_images_after/label_%d_%d.png' % (j, m), neat_img)

