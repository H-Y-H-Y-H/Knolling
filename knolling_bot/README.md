# Robotic Knolling System

This repository contains a robotic knolling system that uses computer vision and a robotic arm to automatically organize objects in a visually appealing arrangement based on a reference layout image.

## Overview

The system uses a YOLO-based object detector to recognize objects in the workspace and then controls an Interbotix robotic arm to arrange them according to a predefined layout. The name "knolling" refers to the process of arranging objects at 90° angles to create a visually organized layout.

The system leverages accurate camera-to-robot calibration using AprilTags, which enables precise determination of object positions in 3D space and allows the robotic arm to manipulate objects with high accuracy.

## Requirements

### Hardware
- Interbotix WX200 Robotic Arm
- RGB-D Camera (e.g., Intel RealSense)
- Workspace with diffuse lighting

### Software
- **Operating System:** Ubuntu 20.04 LTS
- **ROS2:** Humble Hawksbill
- **Python:** 3.8 or higher
- **CUDA:** 11.7+ (for optimal YOLO performance)
- **Interbotix ROS2 Packages**
- **OpenCV:** 4.6+
- **Ultralytics YOLO:** v8.0+

## Installation

1. **Install ROS2 Humble:**
```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop
```

2. **Install Interbotix Packages:**
```bash
# Add Interbotix repository
sudo curl 'https://raw.githubusercontent.com/Interbotix/interbotix-ros-manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh
chmod +x xsarm_amd64_install.sh
./xsarm_amd64_install.sh -d humble -p wx200
```

3. **Install Python Dependencies:**
```bash
pip install ultralytics opencv-python numpy scipy dataclasses typing
```

4. **Clone and Build Repository:**
```bash
mkdir -p ~/knolling_ws/src
cd ~/knolling_ws/src
git clone https://github.com/your-username/robotic-knolling.git
cd ..
colcon build
source install/setup.bash
```

## Usage

1. **Camera Setup:**
   - Position the camera to view the entire workspace
   - Ensure good lighting conditions for reliable detection

2. **Define Layout:**
   - Create a layout image showing the desired arrangement of objects
   - Place objects in your workspace

3. **Run the System:**
```bash
# In terminal 1: Start the Interbotix ROS2 drivers
ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=wx200

# In terminal 2: Start the camera
ros2 launch realsense2_camera rs_launch.py

# In terminal 3: Run the knolling system with your layout image
cd ~/knolling_ws
source install/setup.bash
python3 src/robotic-knolling/knolling_control.py path/to/your/layout_image.jpg
```

## AprilTag Camera-Robot Calibration

We use AprilTags to sync up the camera and robot coordinate systems - essential for the robot to grab objects in the right spots.

### How It Works

The calibration process creates a map between camera coordinates and robot coordinates:

1. **Tag Detection**: Camera sees an AprilTag mounted on the robot arm
2. **Transform Chain**: We calculate two key transformations:
   - Camera→Tag: Where the tag appears in camera view (T_CamTag)
   - Tag→Robot: Where the tag sits on the robot from design specs (T_TagBase)
3. **Complete Calculation**: Multiply these transforms to get Camera→Robot (T_CamBase = T_CamTag × T_TagBase)

### Technical Implementation

The `InterbotixArmTagInterface` handles the calibration process with these key steps:

1. **Multi-Sample Collection**: The system takes multiple snapshots to improve accuracy:
   ```python
   for x in range(num_samples):
       ps = self.apriltag.find_pose()
       point.x += ps.position.x / float(num_samples)
       point.y += ps.position.y / float(num_samples)
       point.z += ps.position.z / float(num_samples)
       
       # Convert quaternion to Euler angles and average
       quat_list = [ps.orientation.x, ps.orientation.y, ps.orientation.z, ps.orientation.w]
       rpy_sample = euler_from_quaternion(quat_list)
       rpy[0] += rpy_sample[0] / float(num_samples)  # Roll
       rpy[1] += rpy_sample[1] / float(num_samples)  # Pitch
       rpy[2] += rpy_sample[2] / float(num_samples)  # Yaw
   ```

2. **Homogeneous Transform Pipeline**: The system calculates and composes transformation matrices:
   ```python
   # Convert averaged pose to transformation matrix
   T_CamTag = poseToTransformationMatrix([point.x, point.y, point.z, rpy[0], rpy[1], rpy[2]])
   
   # Get tag to base transform from URDF (via TF2)
   T_TagBase = self.get_transform(tfBuffer, self.arm_tag_frame, arm_base_frame)
   
   # Calculate camera to base transform
   T_CamBase = np.dot(T_CamTag, T_TagBase)
   
   # If needed, transform to another reference frame
   if ref_frame != self.apriltag.image_frame_id:
       T_RefCam = self.get_transform(tfBuffer, ref_frame, self.apriltag.image_frame_id)
       T_RefBase = np.dot(T_RefCam, T_CamBase)
   else:
       T_RefBase = T_CamBase
   ```

3. **Position-Only Refinement** (Optional): For improved accuracy, sometimes only the position from the camera detection is used:
   ```python
   if position_only:
       # Get the transform from the camera to the actual tag (from URDF)
       T_CamActualTag = self.get_transform(tfBuffer, self.apriltag.image_frame_id, self.arm_tag_frame)
       # Use position from detection but orientation from URDF
       T_CamTag[:3,:3] = T_CamActualTag[:3,:3]
   ```

4. **Transform Publication**: The final transform is published to the ROS TF tree as a static transform:
   ```python
   # Extract translation and rotation
   self.trans.transform.translation.x = T_RefBase[0,3]
   self.trans.transform.translation.y = T_RefBase[1,3]
   self.trans.transform.translation.z = T_RefBase[2,3]
   
   # Convert rotation matrix to quaternion
   quat = quaternion_from_euler(self.rpy[0], self.rpy[1], self.rpy[2])
   self.trans.transform.rotation = Quaternion(quat[0], quat[1], quat[2], quat[3])
   
   # Publish to TF tree
   self.apriltag.pub_transforms.publish(self.trans)
   ```

This calibration is crucial - without it, the robot would try to pick up objects in the wrong locations. Once calibrated, any object the camera detects can be accurately located in the robot's coordinate system.

## Key Features

- **Object Detection**: Uses YOLO-based detection with oriented bounding boxes for accurate pose estimation
- **Workspace Management**: Tracks object positions and detects collisions in the workspace
- **Advanced Planning**: Determines optimal order of operations to avoid blocking objects
- **Error Recovery**: Handles temporary positions for objects that block final destinations

## Configuration

The system's behavior can be customized by modifying the `CONFIG` parameters in both Python files:

- **Robot Parameters**: Gripper pressure, timeouts, default positions
- **Workspace Bounds**: Define the usable area on the table
- **Detection Confidence**: Adjust YOLO detection thresholds
- **Approach Parameters**: Fine-tune the robot's picking and placing movements

## Troubleshooting

- **Detection Issues**: 
  - Check lighting conditions
  - Ensure objects have sufficient contrast with workspace
  - Try adjusting confidence thresholds

- **Movement Failures**:
  - Check if objects are within the workspace bounds
  - Verify that the arm has a clear path to the object
  - Adjust gripper parameters for different object sizes

- **Calibration Issues**:
  - Make sure the AprilTag is clearly visible to the camera
  - Reduce glare and ensure consistent lighting
  - Try increasing the number of samples for better averaging