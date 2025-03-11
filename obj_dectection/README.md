# Object Detection using YOLOv11

# Data Description  

This dataset contains information about objects in both messy and tidy configurations. The data is organized into three text files and corresponding image folders:

---

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
| `category_id`  | Integer ID representing the object’s category |
| `color_id`     | Integer ID representing the object’s color    |

---

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
| `category_id`  | Integer ID representing the object’s category |
| `color_id`     | Integer ID representing the object’s color    |

---

### **(c) `5obj_tidy_name_cdn0.txt`**  
- Each row in this file contains the **string names** of the 5 objects listed in the same order as they appear in `5obj_tidy_data_cdn0.txt`.  
- This provides a human-readable description of the objects in the tidy configuration.  

**Example:**  
```text
wrench_2 wrench_2 ballpointpen_1 utilityknife_2 stapler_1
wrench_2 ballpointpen_1 wrench_2 gear_3 spiralnotebook_1
```

### ✅ **Summary of Functionality:**
- `get_obj.py` → Runs YOLOv11 to detect objects, extract `(x, y, Yaw)`, and save them.
- `__init__.py` → Makes the module importable.
- `requirements.txt` → Lists necessary dependencies.
- `README.md` → Provides clear instructions for setup and execution.




