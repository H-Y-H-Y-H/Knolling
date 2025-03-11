import os
import torch
from yolo_v11 import YOLOv11  # Assuming YOLOv11 is implemented as a module

# Load YOLOv11 model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'yolov11.pt')
model = YOLOv11(MODEL_PATH)

def get_object_data(image_path):
    # Run object detection
    results = model.predict(image_path)

    if results:
        x, y, yaw = results['x'], results['y'], results['yaw']

        # TODO: Convert pixel coordinates (x_pixel, y_pixel) and yaw to real-world position and orientation.

        save_results(x, y, yaw)
        print(f"Detected object at x: {x}, y: {y}, yaw: {yaw}")
    else:
        print("No objects detected.")

def save_results(x, y, yaw):
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'object_position.txt'), 'w') as f:
        f.write(f"x: {x}\n")
        f.write(f"y: {y}\n")
        f.write(f"Yaw: {yaw}\n")

if __name__ == "__main__":
    image_path = "path_to_input_image.jpg"  # Change to actual path
    get_object_data(image_path)
