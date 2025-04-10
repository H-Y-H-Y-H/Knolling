#!/usr/bin/env python3
import logging
import cv2
import time
import os
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from ultralytics import YOLO
from interbotix_common_modules.common_robot.robot import create_interbotix_global_node
import tf2_ros

CONFIG = {
    'ROBOT_MODEL': 'wx200',
    'ROBOT_NAME': 'wx200',
    'ARM_BASE_FRAME': 'wx200/base_link',
    'CAMERA_FRAME': 'camera_color_optical_frame',
    'ROI_CONFIG_FILE': 'roi_config.yaml',
    'YOLO_CONFIDENCE_CAMERA': 0.7,
    'YOLO_CONFIDENCE_LAYOUT': 0.6,
    'CAMERA_MODEL_PATH': './models/multitask.pt',
    'LAYOUT_MODEL_PATH': './models/multitask.pt',
    'TABLE': {
        'POINT': [0.360, 0.123, 0.060],
        'X_MIN': 0.1,
        'X_MAX': 0.4,
        'Z_MIN': 0.03,
        'Z_MAX': 0.06
    }
}

@dataclass(frozen=False, eq=True)
class DetectedObject:
    object_position: Tuple[float, float, float]
    bbox: Optional[Dict[str, Any]] = None
    class_name: str = ""
    confidence: float = 0.0
    angle: float = 0.0
    avg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dimensions: Tuple[float, float] = (0.0, 0.0)
    
    def __hash__(self):
        return hash((
            self.object_position,
            self.class_name,
            self.confidence,
            self.angle,
            self.avg_color,
            self.dimensions
        ))
    
    def __eq__(self, other):
        if not isinstance(other, DetectedObject):
            return False
        return (
            self.object_position == other.object_position and
            self.class_name == other.class_name and
            self.confidence == other.confidence and
            self.angle == other.angle and
            self.avg_color == other.avg_color and
            self.dimensions == other.dimensions
        )

class YOLODetector:
    def __init__(self, camera_model_path=None, layout_model_path=None):
        self.camera_model_path = camera_model_path or CONFIG['CAMERA_MODEL_PATH']
        self.layout_model_path = layout_model_path or CONFIG['LAYOUT_MODEL_PATH']
        self.camera_model = YOLO(self.camera_model_path)
        self.camera_model.verbose = False
        if self.layout_model_path != self.camera_model_path:
            self.layout_model = YOLO(self.layout_model_path)
            self.layout_model.verbose = False
        else:
            self.layout_model = self.camera_model
        self.bridge = CvBridge()
        self.latest_color_image = None
        self.camera_matrix = None
        self.roi = None
        self.detected_objects: List[DetectedObject] = []
        self.global_node = create_interbotix_global_node()
        self.color_sub = self.global_node.create_subscription(
            Image, '/camera/camera/color/image_raw', self._color_callback, 10)
        self.info_sub = self.global_node.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self._info_callback, 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.global_node)
        self.load_roi()

    def _color_callback(self, msg):
        try:
            self.latest_color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            logging.error(f"Error converting color image: {e}")

    def _info_callback(self, msg):
        if self.camera_matrix is None:
            try:
                self.camera_matrix = np.array(msg.k).reshape(3, 3)
                logging.info("Camera matrix set")
            except Exception as e:
                logging.error(f"Error parsing camera info: {e}")

    def tensor_to_numpy(self, tensor):
        if hasattr(tensor, 'cpu') and callable(getattr(tensor, 'cpu')):
            tensor = tensor.cpu()
        if hasattr(tensor, 'numpy') and callable(getattr(tensor, 'numpy')):
            return tensor.numpy()
        return tensor

    def load_roi(self):
        if os.path.exists(CONFIG['ROI_CONFIG_FILE']):
            try:
                import yaml
                with open(CONFIG['ROI_CONFIG_FILE'], 'r') as f:
                    config = yaml.safe_load(f)
                    roi_loaded = config.get('roi')
                    if roi_loaded is not None:
                        self.roi = tuple(roi_loaded)
            except: pass

    def save_roi(self, roi):
        try:
            import yaml
            with open(CONFIG['ROI_CONFIG_FILE'], 'w') as f:
                yaml.dump({'roi': list(roi)}, f)
        except: pass

    def wait_for_image(self, timeout=5.0) -> bool:
        start = time.time()
        while self.latest_color_image is None or self.camera_matrix is None:
            if time.time() - start > timeout:
                logging.error("Timeout waiting for image or camera info")
                return False
            rclpy.spin_once(self.global_node, timeout_sec=0.1)
        return True

    def select_roi(self) -> bool:
        if not self.wait_for_image(): return False
        cv2.namedWindow('Select ROI')
        self.roi = cv2.selectROI('Select ROI', self.latest_color_image, False)
        cv2.destroyWindow('Select ROI')
        valid = bool(self.roi[2] > 0 and self.roi[3] > 0)
        if valid: self.save_roi(self.roi)
        return valid
    
    def get_transform(self, target_frame, source_frame, timeout=2.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                return self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            except:
                rclpy.spin_once(self.global_node, timeout_sec=0.1)
        return None

    def get_camera_ray(self, pixel_x, pixel_y):
        if self.camera_matrix is None: return None
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        ray = np.array([(pixel_x - cx) / fx, (pixel_y - cy) / fy, 1.0])
        return ray / np.linalg.norm(ray)

    def get_camera_transform(self):
        transform = self.get_transform(CONFIG['CAMERA_FRAME'], CONFIG['ARM_BASE_FRAME'])
        if transform is None: return None, None
        trans = transform.transform.translation
        t = np.array([trans.x, trans.y, trans.z])
        rot = transform.transform.rotation
        qx, qy, qz, qw = rot.x, rot.y, rot.z, rot.w
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])
        u, s, vh = np.linalg.svd(R)
        R = u @ vh
        return -R.T @ t, R

    def image_to_z(self, pixel_x, pixel_y) -> float:
        ray_cam = self.get_camera_ray(pixel_x, pixel_y)
        if ray_cam is None: return None
        camera_pos, R = self.get_camera_transform()
        if camera_pos is None: return None
        ray_base = R.T @ ray_cam
        table_point = np.array(CONFIG['TABLE']['POINT'])
        table_normal = R[:, 2]
        denom = np.dot(table_normal, ray_base)
        if np.abs(denom) < 1e-6: return None
        D = -np.dot(table_normal, table_point)
        lambda_param = -(np.dot(table_normal, camera_pos) + D) / denom
        intersection = camera_pos + lambda_param * ray_base
        normalized_x = (intersection[0] - CONFIG['TABLE']['X_MIN']) / (CONFIG['TABLE']['X_MAX'] - CONFIG['TABLE']['X_MIN'])
        return CONFIG['TABLE']['Z_MIN'] + normalized_x * (CONFIG['TABLE']['Z_MAX'] - CONFIG['TABLE']['Z_MIN'])

    def image_to_xy(self, pixel_x, pixel_y, plane_z=0.0) -> Optional[Tuple[float, float]]:
        ray_cam = self.get_camera_ray(pixel_x, pixel_y)
        if ray_cam is None: return None
        camera_pos, R = self.get_camera_transform()
        if camera_pos is None: return None
        ray_base = R.T @ ray_cam
        if abs(ray_base[2]) < 1e-6: return None
        lambda_param = (plane_z - camera_pos[2]) / ray_base[2]
        intersection = camera_pos + lambda_param * ray_base
        return (intersection[0], intersection[1])
    
    def detect_objects(self) -> bool:
        if self.latest_color_image is None: return False
        if self.roi:
            x, y, w, h = self.roi
            detect_image = self.latest_color_image[y:y+h, x:x+w]
            roi_offset = (x, y)
        else:
            detect_image = self.latest_color_image
            roi_offset = (0, 0)

        results = self.camera_model.predict(detect_image, conf=CONFIG['YOLO_CONFIDENCE_CAMERA'], verbose=False)
        if not results or len(results) == 0: return False
        
        self.detected_objects.clear()
        
        for result in results:
            try:
                obb_data = self.tensor_to_numpy(result.obb.xywhr) if hasattr(result.obb, 'xywhr') else None
                cls_data = self.tensor_to_numpy(result.obb.cls) if hasattr(result.obb, 'cls') else None
                conf_data = self.tensor_to_numpy(result.obb.conf) if hasattr(result.obb, 'conf') else None
                poly_data = self.tensor_to_numpy(result.obb.xyxyxyxy) if hasattr(result.obb, 'xyxyxyxy') else None
                
                if obb_data is None or cls_data is None or conf_data is None:
                    continue
                
                for i in range(len(obb_data)):
                    if i >= len(cls_data) or i >= len(conf_data):
                        continue
                    
                    xywhr = obb_data[i]
                    class_id = int(cls_data[i]) if isinstance(cls_data[i], (int, float)) else int(cls_data[i].item())
                    confidence = float(conf_data[i]) if isinstance(conf_data[i], (int, float)) else float(conf_data[i].item())
                    
                    if len(xywhr) < 5:
                        continue
                    
                    cx, cy, w, h, angle = xywhr[:5]
                    
                    if self.roi:
                        cx += roi_offset[0]
                        cy += roi_offset[1]
                    
                    z = self.image_to_z(cx, cy)
                    if z is None: continue
                    
                    world_xy = self.image_to_xy(cx, cy, z)
                    if world_xy is None: continue
                    
                    world_coords = world_xy + (z,)
                    
                    world_w, world_h = 0.0, 0.0
                    if w > 0 and h > 0:
                        corner1 = self.image_to_xy(cx - w/2, cy, z)
                        corner2 = self.image_to_xy(cx + w/2, cy, z)
                        corner3 = self.image_to_xy(cx, cy - h/2, z)
                        corner4 = self.image_to_xy(cx, cy + h/2, z)
                        
                        if corner1 and corner2:
                            world_w = np.linalg.norm(np.array(corner2) - np.array(corner1))
                        if corner3 and corner4:
                            world_h = np.linalg.norm(np.array(corner4) - np.array(corner3))
                    
                    polygon_points = []
                    if poly_data is not None and i < len(poly_data):
                        img_polygon = poly_data[i].reshape(-1, 2)
                        for point in img_polygon:
                            px, py = point
                            if self.roi:
                                px += roi_offset[0]
                                py += roi_offset[1]
                            world_point = self.image_to_xy(px, py, z)
                            if world_point:
                                polygon_points.append(world_point)
                    
                    x1, y1 = int(cx - w/2), int(cy - h/2)
                    x2, y2 = int(cx + w/2), int(cy + h/2)
                    
                    if self.roi:
                        x1 += roi_offset[0]
                        y1 += roi_offset[1]
                        x2 += roi_offset[0]
                        y2 += roi_offset[1]
                    
                    avg_color = (0, 0, 0)
                    if x1 < x2 and y1 < y2:
                        try:
                            roi_crop = self.latest_color_image[y1:y2, x1:x2]
                            avg_color = cv2.mean(roi_crop)[:3]
                        except: pass
                    
                    bbox = {
                        'width': world_w,
                        'height': world_h,
                        'points': polygon_points
                    }

                    print(f'BOX: {bbox}, {world_w}, {world_h}')
                    
                    self.detected_objects.append(DetectedObject(
                        object_position=world_coords,
                        class_name=self.camera_model.names[class_id],
                        confidence=confidence,
                        angle=angle,
                        avg_color=avg_color,
                        bbox=bbox,
                        dimensions=(world_w, world_h)
                    ))
                    logging.info(f"Detected {self.camera_model.names[class_id]} at {world_coords} with angle {angle:.4f}")
            except Exception as e:
                logging.error(f"Error with detection: {e}")
                
        return len(self.detected_objects) > 0

if __name__ == '__main__':
    import sys
    rclpy.init()
    detector = YOLODetector()
    if not detector.wait_for_image(): sys.exit(1)
    detector.select_roi()
    if not detector.detect_objects(): sys.exit(1)
    rclpy.shutdown()