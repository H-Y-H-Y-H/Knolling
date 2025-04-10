#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
import torch
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        print(f"Using device: {device}")
        
        self.model = YOLO(model_path)
        self.model.to(device)
    
    def predict(self, image_path, conf_threshold=0.6):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return []
        
        results = self.model.predict(image, conf=conf_threshold, verbose=False)
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        
        if not hasattr(result, 'obb') or not hasattr(result.obb, 'xywhr'):
            print("YOLO-OBB data not available")
            return []
            
        obb_data = result.obb.xywhr.cpu().numpy() if hasattr(result.obb.xywhr, 'cpu') else result.obb.xywhr
        conf_data = result.obb.conf.cpu().numpy() if hasattr(result.obb.conf, 'cpu') else result.obb.conf
        cls_data = result.obb.cls.cpu().numpy() if hasattr(result.obb.cls, 'cpu') else result.obb.cls
        
        if len(obb_data) == 0:
            return []
        
        detected_objects = []
        for i in range(len(obb_data)):
            xywhr = obb_data[i]
            
            cx, cy, w, h, angle = xywhr[:5]            
            class_id = int(cls_data[i]) if i < len(cls_data) else -1
            class_name = self.model.names[class_id] if class_id >= 0 and class_id in self.model.names else "unknown"            
            confidence = float(conf_data[i]) if i < len(conf_data) else 0.0
            
            detected_objects.append({
                'x': float(cx),
                'y': float(cy),
                'yaw': float(angle),
                'width': float(w),
                'height': float(h),
                'class_name': class_name,
                'class_id': class_id,
                'confidence': confidence
            })
        
        detected_objects.sort(key=lambda obj: obj['confidence'], reverse=True)
        return detected_objects

def get_object_data(model, image_path, conf_threshold=0.6):
    objects = model.predict(image_path, conf_threshold)
    
    if objects:
        print(f"Detected {len(objects)} objects:")
        for i, obj in enumerate(objects):
            print(f"  Object {i+1}: {obj['class_name']} at x: {obj['x']:.2f}, y: {obj['y']:.2f}, yaw: {obj['yaw']:.2f} (conf: {obj['confidence']:.2f})")
        return objects
    else:
        print("No objects detected.")
        return []

def save_results(objects, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for i, obj in enumerate(objects):
            f.write(f"Object {i+1}:\n")
            f.write(f"  x: {obj['x']}\n")
            f.write(f"  y: {obj['y']}\n")
            f.write(f"  yaw: {obj['yaw']}\n")
            f.write(f"  class: {obj['class_name']}\n")
            f.write(f"  confidence: {obj['confidence']}\n")
            f.write("\n")
    
    print(f"Results saved to {output_file}")

def process_folder(model, folder_path, output_dir, conf_threshold=0.6):
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')]
    image_files.sort(key=lambda x: int(x.split('_')[0]) if '_' in x and x.split('_')[0].isdigit() else 0)
    
    all_results = []
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing {image_path}")
        
        objects = get_object_data(model, image_path, conf_threshold)
        
        if objects:
            base_name = os.path.splitext(image_file)[0]
            output_file = os.path.join(output_dir, f"{base_name}_positions.txt")
            save_results(objects, output_file)
            all_results.append((base_name, objects))
    
    if all_results:
        combined_file = os.path.join(output_dir, "all_positions.txt")
        with open(combined_file, 'w') as f:
            for name, objects in all_results:
                f.write(f"{name}:\n")
                for i, obj in enumerate(objects):
                    f.write(f"  Object {i+1}: {obj['class_name']} at x={obj['x']:.2f}, y={obj['y']:.2f}, yaw={obj['yaw']:.2f}\n")
                f.write("\n")
        print(f"Combined results saved to {combined_file}")
    
    data_format_file = os.path.join(output_dir, "objects_data.txt")
    with open(data_format_file, 'w') as f:
        for name, objects in all_results:
            line_data = []
            for obj in objects[:5]: 
                line_data.extend([
                    str(obj['x']), 
                    str(obj['y']), 
                    str(obj['yaw']), 
                    str(obj['width']), 
                    str(obj['height']), 
                    str(obj.get('height', 0.01)), 
                    str(obj['class_id']), 
                    "0"  
                ])
            
            while len(line_data) < 40:  
                line_data.append("0")
                
            f.write(" ".join(line_data) + "\n")
    
    print(f"Data format results saved to {data_format_file}")

def main():
    parser = argparse.ArgumentParser(description='Detect objects in images and save position data')
    parser.add_argument('--image', type=str, help='Path to a single image')
    parser.add_argument('--folder', type=str, default='./data/tidy', help='Path to folder containing images')
    parser.add_argument('--output-dir', type=str, default='./results/tidy', help='Output directory for results')
    parser.add_argument('--model-path', type=str, default='./models/multitask.pt', help='Path to YOLO model')
    parser.add_argument('--confidence', type=float, default=0.6, help='Detection confidence threshold')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], help='Device to run on (default: auto)')
    args = parser.parse_args()
    
    model = YOLODetector(args.model_path, args.device)
    if args.image:
        objects = get_object_data(model, args.image, args.confidence)
        if objects:
            output_file = os.path.join(args.output_dir, "object_positions.txt")
            save_results(objects, output_file)
    else:
        process_folder(model, args.folder, args.output_dir, args.confidence)

if __name__ == "__main__":
    main()