# Object Detection using YOLO-OBB

This project detects objects in images, extracts their positions (x, y coordinates) and orientations (yaw angle), and saves the results in structured text files. It uses YOLO with Oriented Bounding Box (OBB) support to accurately identify object positions and rotations.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/object-detection.git
   cd object-detection
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the YOLO model (if not included), and place it in the `models` directory.

## Usage

### Process a single image:
```bash
python get_objects.py --image path_to_image.jpg
```

### Process a folder of images (defaults to ./data/tidy):
```bash
python get_objects.py
```

### Specify a different folder:
```bash
python get_objects.py --folder ./data/messy
```

### Specify device for acceleration:
```bash
python get_objects.py --device mps  # For Apple Silicon
python get_objects.py --device cuda  # For NVIDIA GPUs
```

## Output Format

The script produces three types of output files:

1. **Individual image results** (e.g. `0_0_positions.txt`):
   - Detailed information about each detected object in an image
   - Includes x, y, yaw, class, and confidence values

2. **Combined summary** (`all_positions.txt`):
   - Summary of all detections across all processed images
   - Format: `image_name: Object 1: class at x=X, y=Y, yaw=Z`

3. **Data format file** (`objects_data.txt`):
   - Compatible with the dataset format described below
   - Each row contains data for up to 5 objects
   - Each object has 8 values: x, y, yaw, width, height, height, class_id, color_id

# Data Description  

This dataset contains information about objects in both messy and tidy configurations. The data is organized into three text files and corresponding image folders:

## **1. Data Files**  
### **(a) `5obj_tidy_data_cdn0.txt`**  
- Each row in this file represents data for **5 objects** arranged in a tidy condition 0.  
- Each row contains **35 values** (5 objects × 7 attributes), which can be reshaped into a `[5, 7]` matrix where each row corresponds to one object.  

**Object Information Format** (in order):  

| Attribute      | Description                                  |
|---------------|----------------------------------------------|
| `x`           | X-coordinate of the object in the tidy configuration |
| `y`           | Y-coordinate of the object in the tidy configuration |
| `length`       | Length of the object                          |
| `width`        | Width of the object                           |
| `height`       | Height of the object                          |
| `category_id`  | Integer ID representing the object's category |
| `color_id`     | Integer ID representing the object's color    |

### **(b) `5obj_messy_data_cdn0.txt`**  
- Each row in this file represents data for **5 objects** arranged in a messy configuration.  
- Each row contains **40 values** (5 objects × 8 attributes), which can be reshaped into a `[5, 8]` matrix where each row corresponds to one object.  

**Object Information Format** (in order):  

| Attribute      | Description                                  |
|---------------|----------------------------------------------|
| `x`           | X-coordinate of the object in the messy configuration |
| `y`           | Y-coordinate of the object in the messy configuration |
| `yaw`          | Orientation (rotation) of the object in radians |
| `length`       | Length of the object                          |
| `width`        | Width of the object                           |
| `height`       | Height of the object                          |
| `category_id`  | Integer ID representing the object's category |
| `color_id`     | Integer ID representing the object's color    |

### **(c) `5obj_tidy_name_cdn0.txt`**  
- Each row in this file contains the **string names** of the 5 objects listed in the same order as they appear in `5obj_tidy_data_cdn0.txt`.  
- This provides a human-readable description of the objects in the tidy configuration.  

**Example:**  
```text
wrench_2 wrench_2 ballpointpen_1 utilityknife_2 stapler_1
wrench_2 ballpointpen_1 wrench_2 gear_3 spiralnotebook_1
```

## Project Components

- `get_objects.py` → Runs YOLO with OBB support to detect objects, extract `(x, y, yaw)`, and save the results in multiple formats.
- `requirements.txt` → Lists necessary dependencies.
- `README.md` → Provides clear instructions for setup and execution.
- `/data` → Contains example images in tidy and messy configurations.
- `/models` → Contains the YOLO model used for object detection.

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy

See `requirements.txt` for the full list of dependencies.