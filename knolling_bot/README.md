# Robotic Knolling System

This repository contains a robotic knolling system that uses computer vision and a robotic arm to automatically organize objects in a visually appealing arrangement based on a reference layout image.

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

## Project Structure

- `knolling_detector.py`: Handles object detection using YOLO and camera integration
- `knolling_control.py`: Controls the robotic arm and implements the knolling algorithm

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

## Extending the System

### Adding New Object Classes
1. Train your YOLO model to recognize new objects
2. Update the model files in the `models/` directory
3. No code changes needed as the system dynamically reads classes from the model

### Using Different Robot Models
1. Update the `CONFIG` parameters to match your robot
2. Install appropriate Interbotix drivers
3. Adjust approach/grasp parameters if your gripper differs

## Troubleshooting

- **Detection Issues**: 
  - Check lighting conditions
  - Ensure objects have sufficient contrast with workspace
  - Try adjusting confidence thresholds

- **Movement Failures**:
  - Check if objects are within the workspace bounds
  - Verify that the arm has a clear path to the object
  - Adjust gripper parameters for different object sizes

- **TF Errors**:
  - Ensure all required ROS2 transforms are being published
  - Check if camera_info topic is correctly configured

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ultralytics for the YOLO implementation
- Interbotix for the robotic arm and ROS2 packages
- The ROS2 community for the robust robotics framework