# Autonomous Drone Person Tracker with YOLOv5 + MOT

**An intelligent drone system that autonomously tracks and follows a person using computer vision and multi-object tracking.**

---

## 🎥 Demo Video

https://github.com/user-attachments/assets/841a7ea1-2277-4efb-b34f-da88efb6ba4d

*Watch the autonomous drone in action: detection, tracking, approach, hover for 10 sec and land back to it's initial position.*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Mission Workflow](#mission-workflow)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)

---

## 🎯 Overview

This project implements an autonomous drone system capable of:
- Detecting and tracking a person using YOLOv5 object detection
- Following the person with precise visual servoing
- Maintaining safe distance (5 meters)
- Hovering at target position
- Returning to launch position automatically

The system uses **Multi-Object Tracking (MOT)** with ByteTrack algorithm to maintain persistent tracking even through occlusions and detection failures.

---

## ✨ Features

### Core Capabilities
- ✅ **YOLOv5 Person Detection** - Fast and accurate person detection
- ✅ **Multi-Object Tracking (MOT)** - Persistent ID tracking with ByteTrack
- ✅ **Person Re-Identification** - ResNet50-based appearance matching
- ✅ **Hybrid Trajectory System** - Precise calculation + emergency edge safety
- ✅ **Visual Servoing** - PID-controlled centering and approach
- ✅ **Autonomous Navigation** - Full mission automation from takeoff to landing
- ✅ **Edge Detection & Avoidance** - Prevents target loss at frame boundaries
- ✅ **Distance Estimation** - Monocular depth estimation using bbox height

### Mission Sequence
1. **Takeoff** → 10 meters altitude
2. **Search** → 360° rotation to find person
3. **Detection** → Lock onto target with persistent ID
4. **Centering** → Center target in frame (PID control)
5. **Approach** → Move to 5 meters from target
6. **Hover** → Hold position for 10 seconds
7. **RTL** → Return to initial position
8. **Land** → Safe landing

---

## 💻 System Requirements

### Hardware
- **Drone**: ArduCopter-compatible (tested with Iris quadcopter)
- **Computer**: x86_64 Linux system with GPU (recommended)
- **RAM**: Minimum 8GB (16GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (for real-time performance)

### Software Dependencies
- **Ubuntu**: 20.04 LTS (recommended)
- **ROS**: Noetic (full desktop installation)
- **Gazebo**: 11.x (simulation environment)
- **Python**: 3.8+
- **CUDA**: 11.x+ (for GPU acceleration)

> **Note**: This README assumes you already have ROS Noetic and Gazebo 11 installed. For installation instructions, refer to:
> - [ROS Noetic Installation](http://wiki.ros.org/noetic/Installation/Ubuntu)
> - [Gazebo 11 Installation](https://robots.uc3m.es/installation-guides/install-gazebo.html)

---

## 📦 Installation

### 1. Clone ArduPilot

```bash
cd ~
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot
git submodule update --init --recursive
```

### 2. Install ArduPilot Dependencies

```bash
cd ~/ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile
```

### 3. Clone ArduPilot Gazebo Plugin

```bash
cd ~
git clone https://github.com/khancyr/ardupilot_gazebo.git
cd ardupilot_gazebo
mkdir build && cd build
cmake ..
make -j4
sudo make install
```

### 4. Set Up Gazebo Environment

```bash
echo 'export GAZEBO_MODEL_PATH=~/ardupilot_gazebo/models:${GAZEBO_MODEL_PATH}' >> ~/.bashrc
echo 'export GAZEBO_RESOURCE_PATH=~/ardupilot_gazebo/worlds:${GAZEBO_RESOURCE_PATH}' >> ~/.bashrc
source ~/.bashrc
```

### 5. Set Up Catkin Workspace

If you don't have this workspace already:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace
```

Clone this repository:

```bash
cd ~/catkin_ws/src
git clone <your-repo-url> obj_detect
```

### 6. Install ROS Dependencies

```bash
cd ~/catkin_ws/src
git clone https://github.com/Intelligent-Quads/iq_sim.git
git clone https://github.com/mavlink/mavros.git
cd mavros
git checkout noetic-devel
```

Install MAVROS dependencies:

```bash
sudo apt-get install -y \
    ros-noetic-mavros \
    ros-noetic-mavros-extras \
    ros-noetic-mavros-msgs
```

Install GeographicLib datasets:

```bash
wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
sudo bash ./install_geographiclib_datasets.sh
```

### 7. Build Workspace

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

Add to your `.bashrc`:

```bash
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
```

### 8. Set Up Python Virtual Environment

```bash
cd ~/catkin_ws/src/obj_detect/yolov5
python3 -m venv venv
source venv/bin/activate
```

### 9. Install Python Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python
pip install numpy
pip install scipy
pip install dronekit
pip install pymavlink
pip install rospkg
```

Install YOLOv5 requirements:

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
catkin_ws/src/obj_detect/
├── CMakeLists.txt              # Catkin build configuration
├── package.xml                 # ROS package manifest
├── README.md                   # This file
├── launch/                     # ROS launch files
│   ├── autonomous_tracker.launch
│   └── person_tracker.launch
├── msg/                        # Custom ROS messages
│   └── Detection.msg
├── rviz/                       # RViz configuration
│   └── person_tracking.rviz
└── yolov5/                     # YOLOv5 + Detection system
    ├── integrated_drone_detect_with_mot.py  # Main detection script
    ├── models/                 # YOLOv5 model architectures
    ├── utils/                  # YOLOv5 utilities
    ├── data/                   # Dataset configs
    ├── venv/                   # Python virtual environment
    ├── yolov5s.pt             # YOLOv5 Small weights
    ├── yolov5m.pt             # YOLOv5 Medium weights
    ├── yolov5l.pt             # YOLOv5 Large weights
    ├── yolov5x.pt             # YOLOv5 XLarge weights
    ├── requirements.txt        # Python dependencies
    └── LICENSE                 # YOLOv5 license
```

---

## 🚀 Usage

### Quick Start (3 Terminals)

#### Terminal 1: Launch Gazebo Simulation
```bash
roslaunch iq_sim hills.launch
```

This will start:
- Gazebo world with terrain
- Spawn Iris quadcopter
- Start ROS nodes

#### Terminal 2: Start ArduPilot SITL
```bash
cd ~/ardupilot/ArduCopter/
sim_vehicle.py -v ArduCopter -f gazebo-iris --console
```

Wait for:
- "APM: EKF2 IMU0 is using GPS"
- "APM: EKF2 IMU1 is using GPS"

#### Terminal 3: Run Detection System
```bash
source ~/catkin_ws/src/obj_detect/yolov5/venv/bin/activate
source ~/catkin_ws/devel/setup.bash
cd ~/catkin_ws/src/obj_detect/yolov5
python integrated_drone_detect_with_mot.py
```

### Expected Output

```
🚁 DRONE PERSON TRACKER WITH MOT
================================================================

Connecting to vehicle...
✅ Drone connected: ArduCopter V4.x.x
🎯 MOT Tracking Mode: Will lock onto ONE person and follow persistently

Arming motors...
Taking off!
 Altitude: 10.00 m
Reached target altitude

--- Starting Mission ---

Starting search: 24 positions, 3s pause
⚠️⚠️ TRACK ID 5 DETECTED! ⚠️⚠️
🎯 Starting CENTERING...
✓✓✓ TRACK ID 5 CENTERED! ✓✓✓
📐 PRECISE TRAJECTORY CALCULATED:
   🔹 FORWARD: 4.2m
   🔹 LATERAL: -0.3m
   🔹 VERTICAL: +0.1m
🚀 HYBRID SYSTEM: EXECUTING PRECISE TRAJECTORY
   Progress: 50% (2.1/4.2s)
✓ TRAJECTORY EXECUTION COMPLETE!
🎯 APPROACH COMPLETE! At 5.1m from target
⏱️ HOVERING at target (5m)... (5.2/10.0s)
✓✓✓ HOVER COMPLETE (10.0s)! ✓✓✓
🏠 RETURNING TO LAUNCH POSITION...
✓✓✓ RTL COMPLETE! ✓✓✓
Landing...
✅✅✅ MISSION SUCCESS ✅✅✅
```

---

## 🔄 Mission Workflow

### State Machine Flow

```
IDLE → SEARCHING → DETECTED → CENTERING → TRAJECTORY_EXEC → LOCKED → HOVERING → RTL → LAND
```

### Detailed States

| State | Description | Duration | Exit Condition |
|-------|-------------|----------|----------------|
| **SEARCHING** | 360° rotation to find person | Variable | Person detected |
| **DETECTED** | Target acquired | Instant | Transition to centering |
| **CENTERING** | Center target in frame | ~5 seconds | Target centered |
| **TRAJECTORY_EXEC** | Execute precise approach | 1-10 seconds | At target distance (5m) |
| **LOCKED** | Approach complete | Instant | Start hovering |
| **HOVERING** | Hold position at 5m | 10 seconds | Timer complete |
| **RTL** | Return to launch | Variable | At initial position |
| **LAND** | Descend to ground | Variable | Disarmed |

---

## ⚙️ Configuration

### Key Parameters (in `integrated_drone_detect_with_mot.py`)

#### Distance & Approach
```python
self.target_approach_distance = 5.0      # Stop at 5 meters
self.min_approach_distance = 0.5         # Minimum safe distance
self.max_approach_distance = 15.0        # Maximum detection range
```

#### Centering Thresholds
```python
self.centering_threshold_pixels = 100    # Centering tolerance (±100px)
self.centered_hold_time = 2.0            # Hold centered for 2 seconds
```

#### PID Gains (Yaw Control)
```python
self.pid_yaw_kp = 0.01    # Proportional gain
self.pid_yaw_ki = 0.0002  # Integral gain
self.pid_yaw_kd = 0.01    # Derivative gain
```

#### Hover Duration
```python
self.hover_duration = 10.0  # Hover for 10 seconds
```

#### Trajectory Execution
```python
movement_velocity = 0.5  # 0.5 m/s smooth approach
```

#### Emergency Override
```python
edge_padding_horizontal = 150  # 150px padding from edges
edge_padding_vertical = 150
emergency_lateral = 0.40       # 0.4 m/s emergency correction
emergency_vertical = 0.35      # 0.35 m/s emergency correction
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "No module named 'cv2'"
```bash
source ~/catkin_ws/src/obj_detect/yolov5/venv/bin/activate
pip install opencv-python
```

#### 2. "Connection refused" to drone
- Ensure ArduPilot SITL is running in Terminal 2
- Check MAVLink connection: `udp:127.0.0.1:14550`
- Wait for EKF initialization messages

#### 3. Camera feed not showing
- Verify Gazebo is running
- Check ROS topic: `rostopic echo /webcam/image_raw`
- Ensure `iq_sim` is properly launched

#### 4. Drone won't arm
- Check GPS lock in SITL console
- Ensure mode is GUIDED: `mode GUIDED`
- Check battery level (simulated)

#### 5. Target not detected
- Ensure person model is in Gazebo world
- Check YOLOv5 confidence threshold (default 0.20)
- Verify camera FOV and position

#### 6. Drone falls during hover
- System now uses zero-velocity commands (not ALT_HOLD)
- Check logs for "HOVERING at target (5m)"
- Ensure GUIDED mode is maintained

#### 7. Edge detection too sensitive
- Adjust `edge_padding_horizontal` and `edge_padding_vertical`
- Default is 150px padding from frame edges

---

## 🧠 Technical Details

### Architecture

#### 1. Object Detection
- **Model**: YOLOv5x (extra-large for accuracy)
- **Framework**: PyTorch
- **Input**: 640x640 RGB images
- **Output**: Bounding boxes [x1, y1, x2, y2, confidence]
- **Confidence**: 0.20 threshold
- **Classes**: Person (class 0) only

#### 2. Multi-Object Tracking (MOT)
- **Algorithm**: ByteTrack
- **Features**:
  - Kalman filter for bbox prediction
  - IoU-based association (Hungarian algorithm)
  - Persistent track IDs
  - Handles occlusions (max_age=60 frames)
- **Parameters**:
  - `max_age`: 60 frames (~2 seconds)
  - `min_hits`: 3 detections
  - `iou_threshold`: 0.25

#### 3. Person Re-Identification
- **Model**: ResNet50 (pretrained on ImageNet)
- **Feature Dimension**: 2048
- **Similarity**: Cosine similarity
- **Threshold**: 0.25 (very permissive)
- **Usage**: Hybrid approach (blind navigation < 3m)

#### 4. Visual Servoing
- **Control**: PID (Proportional-Integral-Derivative)
- **Axes**: Yaw (horizontal), Pitch (vertical)
- **Frequency**: 10 Hz command rate
- **Frame**: Body-frame to NED conversion

#### 5. Trajectory Planning
- **Method**: Pinhole camera model
- **Formula**: `real_offset_m = (pixel_offset × distance) / focal_length`
- **Execution**: Single coordinated movement
- **Velocity**: 0.5 m/s (smooth)
- **Safety**: Edge monitoring + emergency interrupt

#### 6. Distance Estimation
- **Method**: Monocular depth from bbox height
- **Formula**: `distance = (person_height × focal_length) / bbox_height`
- **Assumptions**:
  - Average person height: 1.7 meters
  - Camera focal length: 640 pixels

### Communication

#### ROS Topics
- `/webcam/image_raw` (sensor_msgs/Image) - Camera feed

#### MAVLink
- **Connection**: UDP 127.0.0.1:14550
- **Protocol**: MAVLink v2
- **Commands**:
  - `SET_POSITION_TARGET_LOCAL_NED` - Velocity control
  - `COMMAND_LONG` - Mode changes, arming
  - `MAV_CMD_CONDITION_YAW` - Rotation

### Coordinate Frames

#### NED Frame (North-East-Down)
- **North**: Forward
- **East**: Right
- **Down**: Downward (positive = descend)

#### Body Frame
- **X**: Forward (drone nose)
- **Y**: Right (drone right side)
- **Z**: Down

---

## 📄 License

- **Detection System**: MIT License (this project)
- **YOLOv5**: GPL-3.0 License (Ultralytics)
- **ArduPilot**: GPL-3.0 License

---

## 🙏 Acknowledgments

- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- [ArduPilot](https://ardupilot.org/) autonomous vehicle platform
- [MAVROS](https://github.com/mavlink/mavros) MAVLink to ROS gateway
- [ByteTrack](https://github.com/ifzhang/ByteTrack) multi-object tracking
- [Intelligent Quads](https://github.com/Intelligent-Quads) for iq_sim

---

## 📞 Support

For issues, questions, or contributions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review system logs in terminal outputs
3. Open an issue with detailed logs and error messages

---

## 🚁 Happy Flying!

**This system demonstrates autonomous person tracking with state-of-the-art computer vision and robust multi-object tracking. Safe flights!**
