#!/usr/bin/env python3
import cv2
import time
import rclpy
import numpy as np
import math
import logging
import sys
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from interbotix_common_modules.common_robot.robot import robot_startup, robot_shutdown
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from knolling_detector import YOLODetector, DetectedObject, CONFIG as DETECTOR_CONFIG

CONFIG = {
    'ROBOT': {
        'MODEL': 'wx200',
        'NAME': 'wx200',
        'GRIPPER_PRESSURE': 0.5,
        'GRIPPER_TIMEOUT_GRASP': 2.0,
        'GRIPPER_TIMEOUT_RELEASE': 2.0,
        'DEFAULT_POSITION': [0.25, 0.0, 0.2, 1, 0],
        'SLEEP_POSITION': [0.05, 0.0, 0.2, 1, 0],
        'MOVE_SLEEP': 0.2
    },
    'PICKING': {
        'APPROACH_Z_OFFSET': 0.05,
        'LIFT_Z_OFFSET': 0.1,
        'APPROACH_PITCH': 1.35,
        'GRASP_PITCH': 1.45
    },
    'PLACING': {
        'APPROACH_Z_OFFSET': 0.1,
        'APPROACH_PITCH': 1.45,
        'PLACE_PITCH': 1.45,
        'ANGLE_OFFSET': 1.571
    },
    'WORKSPACE': {
        'X_MIN': 0.10,
        'X_MAX': 0.35,
        'Y_MIN': -0.20,
        'Y_MAX': 0.20,
        'POSITION_TOLERANCE': 0.01,
        'ANGLE_TOLERANCE': 0.175,
        'OBJECT_CLEARANCE': 0.05,
        'GRID_RESOLUTION': 0.02,
        'EMPTY_SPACE_MIN_SIZE': 0.05
    }
}

class ObjectWrapper:
    def __init__(self, obj):
        self.obj = obj
        self.id = id(obj)
    
    def __eq__(self, other):
        if isinstance(other, ObjectWrapper):
            return self.id == other.id
        return False
    
    def __hash__(self):
        return self.id

class WorkspaceManager:
    def __init__(self, workspace_bounds):
        self.bounds = workspace_bounds
        self.occupied_positions = {}
        self.object_positions = {}
        self.object_dimensions = {}
        self.grid_resolution = CONFIG['WORKSPACE']['GRID_RESOLUTION']
        self.empty_spaces = []
        self.kdtree = None
        self.object_map = {}
        
    def initialize_from_detection(self, detected_objects):
        self.occupied_positions = {}
        self.object_positions = {}
        self.object_dimensions = {}
        self.object_map = {}
        occupied_points = []
        
        for obj in detected_objects:
            x, y, z = obj.object_position
            wrapper = ObjectWrapper(obj)
            self.object_map[wrapper] = obj
            
            width, height = self._get_object_dimensions(obj)
            
            pos_id = f"{x:.3f}_{y:.3f}"
            self.occupied_positions[pos_id] = wrapper
            self.object_positions[wrapper.id] = (x, y, z)
            self.object_dimensions[wrapper.id] = (width, height)
            occupied_points.append((x, y))
        
        x_range = np.arange(self.bounds['X_MIN'], self.bounds['X_MAX'], self.grid_resolution)
        y_range = np.arange(self.bounds['Y_MIN'], self.bounds['Y_MAX'], self.grid_resolution)
        
        self.empty_spaces = []
        
        if occupied_points:
            tree = KDTree(occupied_points)
            self.kdtree = tree
            
            for x in x_range:
                for y in y_range:
                    if not self.is_position_occupied((x, y, 0)):
                        avg_z = sum([obj.object_position[2] for obj in detected_objects]) / len(detected_objects) if detected_objects else 0.035
                        self.empty_spaces.append((x, y, avg_z))
        else:
            for x in x_range:
                for y in y_range:
                    self.empty_spaces.append((x, y, 0.035))
        
        center_x = (self.bounds['X_MIN'] + self.bounds['X_MAX']) / 2
        center_y = (self.bounds['Y_MIN'] + self.bounds['Y_MAX']) / 2
        self.empty_spaces.sort(key=lambda p: ((p[0]-center_x)**2 + (p[1]-center_y)**2))
    
    def _get_object_dimensions(self, obj):
        if hasattr(obj, 'dimensions') and obj.dimensions and all(d > 0 for d in obj.dimensions):
            return obj.dimensions
        elif obj.bbox and 'width' in obj.bbox and 'height' in obj.bbox:
            return obj.bbox['width'], obj.bbox['height']
        else:
            return (0.05, 0.05)
    
    def _check_oriented_rectangle_collision(self, pos1, size1, angle1, pos2, size2, angle2):
        x1, y1 = pos1[0], pos1[1]
        width1, height1 = size1
        x2, y2 = pos2[0], pos2[1]
        width2, height2 = size2
        max_radius1 = math.sqrt(width1**2 + height1**2) / 2
        max_radius2 = math.sqrt(width2**2 + height2**2) / 2
        
        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        min_dist = CONFIG['WORKSPACE']['OBJECT_CLEARANCE']
        
        if dist > (max_radius1 + max_radius2 + min_dist):
            return False
        
        def get_corners(x, y, width, height, angle):
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            hw, hh = width/2, height/2
            corners = [
                (x + hw * cos_a - hh * sin_a, y + hw * sin_a + hh * cos_a),
                (x - hw * cos_a - hh * sin_a, y - hw * sin_a + hh * cos_a),
                (x - hw * cos_a + hh * sin_a, y - hw * sin_a - hh * cos_a),
                (x + hw * cos_a + hh * sin_a, y + hw * sin_a - hh * cos_a),
            ]
            return corners
        
        corners1 = get_corners(x1, y1, width1, height1, angle1)
        corners2 = get_corners(x2, y2, width2, height2, angle2)
        
        edges = []
        for i in range(4):
            edges.append((
                corners1[(i+1)%4][0] - corners1[i][0],
                corners1[(i+1)%4][1] - corners1[i][1]
            ))
            edges.append((
                corners2[(i+1)%4][0] - corners2[i][0],
                corners2[(i+1)%4][1] - corners2[i][1]
            ))
        
        normals = []
        for edge in edges:
            if abs(edge[0]) < 1e-10 and abs(edge[1]) < 1e-10:
                continue  # Skip zero-length edges
            normal = (-edge[1], edge[0])
            length = math.sqrt(normal[0]**2 + normal[1]**2)
            if length > 0:
                normals.append((normal[0]/length, normal[1]/length))
        
        for normal in normals:
            min1, max1 = float('inf'), float('-inf')
            min2, max2 = float('inf'), float('-inf')
            
            for c in corners1:
                proj = c[0]*normal[0] + c[1]*normal[1]
                min1 = min(min1, proj)
                max1 = max(max1, proj)
            
            for c in corners2:
                proj = c[0]*normal[0] + c[1]*normal[1]
                min2 = min(min2, proj)
                max2 = max(max2, proj)
            
            if max1 < (min2 - min_dist/2) or max2 < (min1 - min_dist/2):
                return False
        
        return True

    def is_position_occupied(self, position, obj_size=None, obj_angle=0, target_positions=None):
        x, y, _ = position
        
        if self.kdtree is not None:
            distances, _ = self.kdtree.query([(x, y)], k=1)
            if distances[0] < CONFIG['WORKSPACE']['OBJECT_CLEARANCE']:
                return True
        
        if obj_size is None:
            obj_size = (0.05, 0.05)
        
        for wrapper_id, (ox, oy, oz) in self.object_positions.items():
            if oz < -90:
                continue
                
            other_size = self.object_dimensions.get(wrapper_id, (0.05, 0.05))
            
            for wrapper, obj in self.object_map.items():
                if wrapper.id == wrapper_id:
                    other_angle = obj.angle
                    break
            else:
                other_angle = 0
            
            if self._check_oriented_rectangle_collision(
                (x, y), obj_size, obj_angle,
                (ox, oy), other_size, other_angle
            ):
                return True
        
        if target_positions:
            for target_obj in target_positions:
                tx, ty, _ = target_obj.object_position
                target_size = self._get_object_dimensions(target_obj)
                
                if self._check_oriented_rectangle_collision(
                    (x, y), obj_size, obj_angle,
                    (tx, ty), target_size, target_obj.angle
                ):
                    return True
        
        return False

    def find_empty_position(self, obj_size):
        for pos in self.empty_spaces:
            if not self.is_position_occupied(pos, obj_size):
                return pos
                
        center_x = (self.bounds['X_MIN'] + self.bounds['X_MAX']) / 2
        center_y = (self.bounds['Y_MIN'] + self.bounds['Y_MAX']) / 2
        z = self.empty_spaces[0][2] if self.empty_spaces else 0.035
        return (center_x, center_y, z)
    
    def update_position(self, obj, new_position):
        if not isinstance(obj, ObjectWrapper):
            wrapper = ObjectWrapper(obj)
            self.object_map[wrapper] = obj
        else:
            wrapper = obj
            
        self.object_positions[wrapper.id] = new_position
        if new_position[2] >= 0:
            x, y, z = new_position
            pos_id = f"{x:.3f}_{y:.3f}"
            self.occupied_positions[pos_id] = wrapper
        
        occupied_points = []
        for obj_id, (x, y, z) in self.object_positions.items():
            if z >= 0:
                occupied_points.append((x, y))
                
        if occupied_points:
            self.kdtree = KDTree(occupied_points)
        else:
            self.kdtree = None

    def get_object_at_position(self, position, tolerance=0.03):
        x, y, _ = position
        
        pos_id = f"{x:.3f}_{y:.3f}"
        if pos_id in self.occupied_positions:
            wrapper = self.occupied_positions[pos_id]
            return self.object_map.get(wrapper)
        
        for wrapper_id, (ox, oy, oz) in self.object_positions.items():
            if oz < 0:
                continue
                
            dist = math.sqrt((x - ox)**2 + (y - oy)**2)
            if dist <= tolerance:
                for wrapper, obj in self.object_map.items():
                    if wrapper.id == wrapper_id:
                        return obj
        
        return None

class ArrangeObjects:
    def __init__(self, layout_image_path):
        self.detector = YOLODetector()
        self.bot = InterbotixManipulatorXS(
            robot_model=CONFIG['ROBOT']['MODEL'],
            robot_name=CONFIG['ROBOT']['NAME'],
            node=self.detector.global_node,
            moving_time=1.5,  
            accel_time=0.3 
        )
        self.layout_image = cv2.imread(layout_image_path)
        self.target_objects = []
        self.scene_objects = []
        self.workspace_manager = WorkspaceManager(CONFIG['WORKSPACE'])
        self.blocking_map = {}
        
    def _get_current_position(self):
        positions = []
        for _ in range(3):
            rclpy.spin_once(self.detector.global_node)
            time.sleep(0.02)
            positions.append(self.bot.gripper.get_finger_position())
        return np.median(positions)
        
    def _reset_gripper_state(self):
        self.bot.gripper.gripper_command.cmd = 0.0
        self.bot.gripper.core.pub_single.publish(self.bot.gripper.gripper_command)
        rclpy.spin_once(self.bot.gripper.core.get_node())
        time.sleep(0.1)
        
    def knolling_release(self, open_amount=0.0008, delay=1.0):
        self._reset_gripper_state()
        current_pos = self._get_current_position()
        target_pos = min(current_pos + open_amount, self.bot.gripper.left_finger_upper_limit)
        effort = self.bot.gripper.gripper_pressure_lower_limit * 0.8
        start_pos = current_pos
        last_mvt_time = time.time()
        self.bot.gripper.gripper_command.cmd = effort
        self.bot.gripper.core.pub_single.publish(self.bot.gripper.gripper_command)
        start_time = time.time()
        while (time.time() - start_time) < delay:
            current_pos = self._get_current_position()
            if current_pos >= target_pos:
                self._reset_gripper_state()
                return True
            movement = abs(current_pos - start_pos)
            if movement > 0.0001:
                last_mvt_time = time.time()
                start_pos = current_pos
            elif (time.time() - last_mvt_time) > 0.3:
                effort = min(effort * 1.2, self.bot.gripper.gripper_pressure_upper_limit)
                self.bot.gripper.gripper_command.cmd = effort
                self.bot.gripper.core.pub_single.publish(self.bot.gripper.gripper_command)
                last_mvt_time = time.time()
            time.sleep(0.02)
            rclpy.spin_once(self.bot.gripper.core.get_node())
        self._reset_gripper_state()
        return False
        
    def knolling_grasp(self, delay=1.0):
        self._reset_gripper_state()
        initial_pos = self._get_current_position()
        self.bot.gripper.grasp(delay)
        final_pos = self._get_current_position()
        rclpy.spin_once(self.bot.gripper.core.get_node())
        time.sleep(0.1)
        return abs(final_pos - initial_pos) > 0.001
        
    def _execute_move_sequence(self, moves, close_gripper=False):
        for i, (x, y, z, pitch, roll, desc) in enumerate(moves):
            if not (CONFIG['WORKSPACE']['X_MIN'] <= x <= CONFIG['WORKSPACE']['X_MAX'] and
                    CONFIG['WORKSPACE']['Y_MIN'] <= y <= CONFIG['WORKSPACE']['Y_MAX']):
                x = np.clip(x, CONFIG['WORKSPACE']['X_MIN'], CONFIG['WORKSPACE']['X_MAX'])
                y = np.clip(y, CONFIG['WORKSPACE']['Y_MIN'], CONFIG['WORKSPACE']['Y_MAX'])
            logging.info(f"Move: {desc} to ({x:.3f}, {y:.3f}, {z:.3f})")
            if not self.bot.arm.set_ee_pose_components(x=x, y=y, z=z, pitch=pitch, roll=roll):
                logging.error(f"Failed: {desc}")
                continue
            time.sleep(CONFIG['ROBOT']['MOVE_SLEEP'])
            if desc == "Descend for grasp" and close_gripper:
                self.knolling_grasp(CONFIG['ROBOT']['GRIPPER_TIMEOUT_GRASP'])
            elif desc == "Descend for placement":
                self.knolling_release(open_amount=0.0004, delay=CONFIG['ROBOT']['GRIPPER_TIMEOUT_RELEASE'])
            elif desc == "Move to default position":
                self.knolling_release(open_amount=0.015, delay=CONFIG['ROBOT']['GRIPPER_TIMEOUT_RELEASE'])

    def _calculate_roll(self, x, y, angle, movement):
        disp_angle = math.atan2(-y, x)
        roll = angle - disp_angle + CONFIG['PLACING']['ANGLE_OFFSET']
        while roll > math.pi/2: roll -= math.pi
        while roll < -math.pi/2: roll += math.pi
        return roll
    
    def detect_target_objects(self):
        if self.layout_image is None:
            return False
        
        if self.detector.roi:
            roi_width = self.detector.roi[2]
            roi_height = self.detector.roi[3]
        else:
            roi_width = self.detector.latest_color_image.shape[1]
            roi_height = self.detector.latest_color_image.shape[0]
            
        layout_width = self.layout_image.shape[1]
        layout_height = self.layout_image.shape[0]
        x_scaling = layout_width / roi_width
        y_scaling = layout_height / roi_height
        
        results = self.detector.layout_model.predict(self.layout_image, conf=DETECTOR_CONFIG['YOLO_CONFIDENCE_LAYOUT'], verbose=False)
        if not results or len(results) == 0:
            return False
        
        self.target_objects = []
        
        for result in results:
            self._process_obb_results(result, y_scaling, y_scaling)
                
        return len(self.target_objects) > 0
    
    def _process_obb_results(self, result, x_scaling, y_scaling):
        if not hasattr(result, 'obb') or result.obb is None:
            return
            
        try:
            obb_data = self.detector.tensor_to_numpy(result.obb.xywhr) if hasattr(result.obb, 'xywhr') else None
            cls_data = self.detector.tensor_to_numpy(result.obb.cls) if hasattr(result.obb, 'cls') else None
            conf_data = self.detector.tensor_to_numpy(result.obb.conf) if hasattr(result.obb, 'conf') else None
            obb_poly = self.detector.tensor_to_numpy(result.obb.xyxyxyxy) if hasattr(result.obb, 'xyxyxyxy') else None
            
            if obb_data is None or cls_data is None or conf_data is None:
                return
                
            for i in range(len(cls_data)):
                if i >= len(obb_data):
                    continue
                    
                class_id = int(cls_data[i]) if isinstance(cls_data[i], (int, float)) else int(cls_data[i].item())
                confidence = float(conf_data[i]) if isinstance(conf_data[i], (int, float)) else float(conf_data[i].item())
                xywhr = obb_data[i]
                
                if len(xywhr) < 5:
                    continue
                    
                cx, cy, w, h, angle = xywhr[:5]
                camera_x = int(cx / x_scaling)
                camera_y = int(cy / y_scaling)
                if self.detector.roi:
                    camera_x += self.detector.roi[0]
                    camera_y += self.detector.roi[1]
                
                z = self.detector.image_to_z(camera_x, camera_y)
                if z is None:
                    continue
                    
                world_xy = self.detector.image_to_xy(camera_x, camera_y, z)
                if world_xy is None:
                    continue
                    
                world_coords = world_xy + (z,)
                
                x1 = max(0, int(cx - w/2))
                y1 = max(0, int(cy - h/2)) 
                x2 = min(self.layout_image.shape[1]-1, int(cx + w/2))
                y2 = min(self.layout_image.shape[0]-1, int(cy + h/2))
                
                points = []
                if obb_poly is not None and i < len(obb_poly):
                    polygon = obb_poly[i].reshape(-1, 2)
                    for point in polygon:
                        px, py = point
                        px_cam = int(px / x_scaling)
                        py_cam = int(py / y_scaling)
                        if self.detector.roi:
                            px_cam += self.detector.roi[0]
                            py_cam += self.detector.roi[1]
                        world_point = self.detector.image_to_xy(px_cam, py_cam, z)
                        if world_point:
                            points.append(world_point)
                
                world_w, world_h = 0.0, 0.0
                if w > 0 and h > 0:
                    edge_points = []
                    for p in [(cx - w/2, cy), (cx + w/2, cy), (cx, cy - h/2), (cx, cy + h/2)]:
                        px_cam = int(p[0] / x_scaling)
                        py_cam = int(p[1] / y_scaling)
                        if self.detector.roi:
                            px_cam += self.detector.roi[0]
                            py_cam += self.detector.roi[1]
                        world_point = self.detector.image_to_xy(px_cam, py_cam, z)
                        if world_point:
                            edge_points.append(world_point)
                    
                    if len(edge_points) >= 4:
                        world_w = np.linalg.norm(np.array(edge_points[1]) - np.array(edge_points[0]))
                        world_h = np.linalg.norm(np.array(edge_points[3]) - np.array(edge_points[2]))
                
                bbox = {
                    'width': world_w,
                    'height': world_h,
                    'points': points,
                    'area': world_w * world_h
                }
                
                avg_color = (0, 0, 0)
                if x1 < x2 and y1 < y2: 
                    try:
                        roi_crop = self.layout_image[y1:y2, x1:x2]
                        if roi_crop.size > 0: 
                            avg_color = cv2.mean(roi_crop)[:3]
                    except Exception as e:
                        logging.error(f"Error extracting color: {e}")
                
                self.target_objects.append(DetectedObject(
                    object_position=world_coords,
                    class_name=self.detector.layout_model.names[class_id],
                    confidence=confidence,
                    angle=angle,
                    avg_color=avg_color,
                    bbox=bbox,
                    dimensions=(world_w, world_h)
                ))
                
        except Exception as e:
            logging.error(f"Error processing OBB results: {e}")
    
    def match_objects(self, detected_objects, target_objects):
        matches = []
        classes = set(obj.class_name for obj in target_objects)
        for cls in classes:
            target_cls = [obj for obj in target_objects if obj.class_name == cls]
            detected_cls = [obj for obj in detected_objects if obj.class_name == cls]
            if not detected_cls: continue
            cost_matrix = np.zeros((len(target_cls), len(detected_cls)))
            for i, target in enumerate(target_cls):
                for j, detected in enumerate(detected_cls):
                    color_cost = np.linalg.norm(np.array(target.avg_color) - np.array(detected.avg_color))
                    cost_matrix[i, j] = color_cost
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                matches.append((detected_cls[c], target_cls[r]))
        return matches
    
    def _object_in_correct_position(self, detected_obj, target_obj):
        x1, y1, z1 = detected_obj.object_position
        x2, y2, z2 = target_obj.object_position
        
        pos_diff = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        angle_diff = abs(detected_obj.angle - target_obj.angle) % np.pi
        angle_diff = min(angle_diff, np.pi - angle_diff)
        
        return (pos_diff < CONFIG['WORKSPACE']['POSITION_TOLERANCE'] and 
                angle_diff < CONFIG['WORKSPACE']['ANGLE_TOLERANCE'])
    
    def _get_object_size(self, obj):
        if hasattr(obj, 'dimensions') and obj.dimensions and all(d > 0 for d in obj.dimensions):
            return obj.dimensions
        elif obj.bbox and 'width' in obj.bbox and 'height' in obj.bbox:
            return (obj.bbox['width'], obj.bbox['height'])
        else:
            return (0.05, 0.05)
    
    def _pick_object(self, obj):
        x, y, z = obj.object_position
        pick_roll = self._calculate_roll(x, y, obj.angle, 'pick')
        pick_moves = [
            (*CONFIG['ROBOT']['DEFAULT_POSITION'], "Move to default position"),
            (x, y, z + CONFIG['PICKING']['APPROACH_Z_OFFSET'], 
             CONFIG['PICKING']['APPROACH_PITCH'], pick_roll, "Approach above object"),
            (x, y, z, CONFIG['PICKING']['GRASP_PITCH'], pick_roll, "Descend for grasp"),
            (x, y, z + CONFIG['PICKING']['LIFT_Z_OFFSET'], 
             CONFIG['PICKING']['APPROACH_PITCH'], pick_roll, "Lift the object")
        ]
        self._execute_move_sequence(pick_moves, close_gripper=True)
        self.workspace_manager.update_position(obj, (x, y, -99))
        return True
    
    def _place_object(self, position, angle, description="target"):
        x, y, z = position
        place_roll = self._calculate_roll(x, y, angle, 'place')
        place_moves = [
            (x, y, z + CONFIG['PLACING']['APPROACH_Z_OFFSET'], 
             CONFIG['PLACING']['APPROACH_PITCH'], place_roll, f"Approach above {description}"),
            (x, y, z, CONFIG['PLACING']['PLACE_PITCH'], place_roll, "Descend for placement"),
            (x, y, z + CONFIG['PLACING']['APPROACH_Z_OFFSET'], 
             CONFIG['PLACING']['APPROACH_PITCH'], place_roll, "Retract")
        ]
        self._execute_move_sequence(place_moves, close_gripper=False)
        return True
    
    def _find_blocking_objects(self, matches):
        blocked_objects = {}
        already_in_position = set()
        
        for det_obj, tgt_obj in matches:
            if self._object_in_correct_position(det_obj, tgt_obj):
                already_in_position.add(det_obj)
        
        for det_obj, tgt_obj in matches:
            if det_obj in already_in_position:
                continue
                
            target_pos = tgt_obj.object_position
            target_size = self._get_object_size(tgt_obj)
            target_angle = tgt_obj.angle
            
            collision_tolerance = 0.005
            adjusted_target_size = (
                max(0.01, target_size[0] - collision_tolerance),
                max(0.01, target_size[1] - collision_tolerance)
            )
            
            for other_det, other_tgt in matches:
                if other_det == det_obj:
                    continue
                    
                if other_det in already_in_position:
                    continue
                    
                other_pos = other_det.object_position
                other_size = self._get_object_size(other_det)
                other_angle = other_det.angle
                
                adjusted_other_size = (
                    max(0.01, other_size[0] - collision_tolerance),
                    max(0.01, other_size[1] - collision_tolerance)
                )
                
                dx = target_pos[0] - other_pos[0]
                dy = target_pos[1] - other_pos[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                max_r1 = math.sqrt(target_size[0]**2 + target_size[1]**2) / 2
                max_r2 = math.sqrt(other_size[0]**2 + other_size[1]**2) / 2
                
                if distance > (max_r1 + max_r2):
                    continue
                
                if self.workspace_manager._check_oriented_rectangle_collision(
                    (target_pos[0], target_pos[1]), adjusted_target_size, target_angle,
                    (other_pos[0], other_pos[1]), adjusted_other_size, other_angle
                ):
                    blocked_objects[det_obj] = other_det
                    break
                        
        return blocked_objects
    
    def _sort_objects_to_move(self, matches):
        sorted_objects = []
        already_in_position = []
        
        for det_obj, tgt_obj in matches:
            if self._object_in_correct_position(det_obj, tgt_obj):
                already_in_position.append((det_obj, tgt_obj))
            else:
                sorted_objects.append((det_obj, tgt_obj))
        
        blocking_map = self._find_blocking_objects(matches)
        
        priority_objects = []
        standard_objects = []
        
        for det_obj, tgt_obj in sorted_objects:
            if det_obj in blocking_map.values():
                priority_objects.append((det_obj, tgt_obj))
            else:
                standard_objects.append((det_obj, tgt_obj))
        
        return priority_objects + standard_objects, already_in_position, blocking_map
    
    def execute(self):
        robot_startup(self.detector.global_node)
        self.bot.arm.set_ee_pose_components(
            *CONFIG['ROBOT']['SLEEP_POSITION'][:3],
            pitch=CONFIG['ROBOT']['SLEEP_POSITION'][3],
            roll=CONFIG['ROBOT']['SLEEP_POSITION'][4]
        )
        self.bot.gripper.release()
        time.sleep(0.5)
        
        if not self.detector.wait_for_image() or not self.detector.detect_objects():
            logging.error("Failed to detect any objects")
            robot_shutdown(self.detector.global_node)
            return
        
        self.scene_objects = self.detector.detected_objects.copy()
        logging.info(f"Detected {len(self.scene_objects)} objects")
        
        self.workspace_manager.initialize_from_detection(self.scene_objects)
        
        if not self.detect_target_objects():
            logging.error("Failed to detect target objects from layout")
            robot_shutdown(self.detector.global_node)
            return
        
        matches = self.match_objects(self.scene_objects, self.target_objects)
        if not matches:
            logging.error("No matching objects found")
            robot_shutdown(self.detector.global_node)
            return
        logging.info(f"Found {len(matches)} matching pairs")
        
        objects_to_move, correct_positions, blocking_map = self._sort_objects_to_move(matches)
        
        logging.info(f"{len(correct_positions)} objects already in correct position")
        logging.info(f"{len(objects_to_move)} objects need to be moved")
        logging.info(f"{len(blocking_map)} objects are blocking target positions")
        
        if not objects_to_move:
            logging.info("All objects are already in their correct positions. No movement needed.")
            self.bot.arm.set_ee_pose_components(
                *CONFIG['ROBOT']['SLEEP_POSITION'][:3],
                pitch=CONFIG['ROBOT']['SLEEP_POSITION'][3],
                roll=CONFIG['ROBOT']['SLEEP_POSITION'][4]
            )
            robot_shutdown(self.detector.global_node)
            return
        
        # Track objects in various states
        objects_in_position = set(det for det, _ in correct_positions)  # Objects already in correct position initially
        objects_in_final_position = set(objects_in_position)  # Objects in final position (updated dynamically)
        objects_moved_to_temp = set()  # Objects that have been moved to a temporary position
        temp_positions = {}  # Map of objects to their temporary positions
        
        # Temporarily modify the WorkspaceManager to check against target positions
        all_target_positions = [tgt_obj for _, tgt_obj in matches]
        original_is_position_occupied = self.workspace_manager.is_position_occupied
        
        def enhanced_is_position_occupied(position, obj_size=None, obj_angle=0):
            return original_is_position_occupied(position, obj_size, obj_angle, target_positions=all_target_positions)
            
        # Replace the method temporarily
        self.workspace_manager.is_position_occupied = enhanced_is_position_occupied
        
        # First phase: Move blockers to temporary positions if needed
        blockers_to_move = set()
        for det_obj, tgt_obj in objects_to_move:
            if det_obj in blocking_map:
                blocker = blocking_map[det_obj]
                if blocker not in objects_in_final_position and blocker not in objects_moved_to_temp:
                    blockers_to_move.add(blocker)
                    
        for blocker in blockers_to_move:
            if blocker in objects_in_final_position:
                continue
                
            blocker_target = next((tgt for det, tgt in matches if det == blocker), None)
            if blocker_target:
                # Try to place blocker directly at its target position if possible
                can_place_at_target = True
                for other_det, other_tgt in objects_to_move:
                    if other_det != blocker and other_det not in objects_in_final_position:
                        if self.workspace_manager._check_oriented_rectangle_collision(
                            (blocker_target.object_position[0], blocker_target.object_position[1]),
                            self._get_object_size(blocker_target), blocker_target.angle,
                            (other_det.object_position[0], other_det.object_position[1]),
                            self._get_object_size(other_det), other_det.angle
                        ):
                            can_place_at_target = False
                            break
                
                if can_place_at_target:
                    logging.info(f"Placing blocker {blocker.class_name} directly at target position")
                    if self._pick_object(blocker):
                        self._place_object(blocker_target.object_position, blocker_target.angle, "final position")
                        blocker.object_position = blocker_target.object_position
                        self.workspace_manager.update_position(blocker, blocker_target.object_position)
                        objects_in_final_position.add(blocker)
                        
                        self.bot.arm.set_ee_pose_components(
                            *CONFIG['ROBOT']['DEFAULT_POSITION'][:3],
                            pitch=CONFIG['ROBOT']['DEFAULT_POSITION'][3],
                            roll=CONFIG['ROBOT']['DEFAULT_POSITION'][4]
                        )
                        time.sleep(CONFIG['ROBOT']['MOVE_SLEEP'])
                        continue
            
            # Place in temporary position if can't place directly at target
            logging.info(f"Moving blocker {blocker.class_name} to temporary position")
            obj_size = self._get_object_size(blocker)
            temp_pos = self.workspace_manager.find_empty_position(obj_size)
            
            if self._pick_object(blocker):
                self._place_object(temp_pos, blocker.angle, "temporary position")
                blocker.object_position = temp_pos
                self.workspace_manager.update_position(blocker, temp_pos)
                temp_positions[blocker] = temp_pos
                objects_moved_to_temp.add(blocker)
                
                self.bot.arm.set_ee_pose_components(
                    *CONFIG['ROBOT']['DEFAULT_POSITION'][:3],
                    pitch=CONFIG['ROBOT']['DEFAULT_POSITION'][3],
                    roll=CONFIG['ROBOT']['DEFAULT_POSITION'][4]
                )
                time.sleep(CONFIG['ROBOT']['MOVE_SLEEP'])
                
        # Second phase: Move objects to their target positions
        for det_obj, tgt_obj in objects_to_move:
            if det_obj in objects_in_final_position:
                logging.info(f"Object {det_obj.class_name} already in final position, skipping")
                continue
                
            logging.info(f"Moving {det_obj.class_name} to target position")
            
            if self._pick_object(det_obj):
                self._place_object(tgt_obj.object_position, tgt_obj.angle)
                det_obj.object_position = tgt_obj.object_position
                self.workspace_manager.update_position(det_obj, tgt_obj.object_position)
                objects_in_final_position.add(det_obj)
            
            self.bot.arm.set_ee_pose_components(
                *CONFIG['ROBOT']['DEFAULT_POSITION'][:3],
                pitch=CONFIG['ROBOT']['DEFAULT_POSITION'][3],
                roll=CONFIG['ROBOT']['DEFAULT_POSITION'][4]
            )
            time.sleep(CONFIG['ROBOT']['MOVE_SLEEP'])
        
        # Third phase: Move objects from temporary positions to their target positions if needed
        for obj in objects_moved_to_temp:
            if obj in objects_in_final_position:
                logging.info(f"Object {obj.class_name} already moved to final position, skipping")
                continue
                
            target_match = next((m for m in matches if m[0] == obj), None)
            if not target_match:
                continue
                
            _, target = target_match
            
            # Double-check if the object is already at the target position
            current_pos = obj.object_position
            target_pos = target.object_position
            
            pos_diff = np.sqrt((current_pos[0]-target_pos[0])**2 + (current_pos[1]-target_pos[1])**2)
            angle_diff = abs(obj.angle - target.angle) % np.pi
            angle_diff = min(angle_diff, np.pi - angle_diff)
            
            if pos_diff < CONFIG['WORKSPACE']['POSITION_TOLERANCE'] and angle_diff < CONFIG['WORKSPACE']['ANGLE_TOLERANCE']:
                logging.info(f"Object {obj.class_name} already at target position, skipping")
                objects_in_final_position.add(obj)
                continue
                
            logging.info(f"Moving {obj.class_name} from temporary position to target")
            
            if self._pick_object(obj):
                self._place_object(target.object_position, target.angle)
                obj.object_position = target.object_position
                self.workspace_manager.update_position(obj, target.object_position)
                objects_in_final_position.add(obj)
            
            self.bot.arm.set_ee_pose_components(
                *CONFIG['ROBOT']['DEFAULT_POSITION'][:3],
                pitch=CONFIG['ROBOT']['DEFAULT_POSITION'][3],
                roll=CONFIG['ROBOT']['DEFAULT_POSITION'][4]
            )
            time.sleep(CONFIG['ROBOT']['MOVE_SLEEP'])
        
        # Restore original method
        self.workspace_manager.is_position_occupied = original_is_position_occupied
        
        logging.info(f"Completed organization with {len(objects_in_final_position)} objects in final position")
        
        self.bot.arm.set_ee_pose_components(
            *CONFIG['ROBOT']['SLEEP_POSITION'][:3],
            pitch=CONFIG['ROBOT']['SLEEP_POSITION'][3],
            roll=CONFIG['ROBOT']['SLEEP_POSITION'][4]
        )
        robot_shutdown(self.detector.global_node)

def main():
    if len(sys.argv) < 2:
        print("Usage: knolling_control.py <layout_image_path>")
        sys.exit(1)
        
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    cv2.destroyAllWindows()
    rclpy.init()
    try:
        arranger = ArrangeObjects(layout_image_path=sys.argv[1])
        arranger.execute()
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()