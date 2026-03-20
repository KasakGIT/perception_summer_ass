# LiDAR Cone Detection & Classification | IITB Racing

A ROS2 perception pipeline that detects and classifies traffic cones 
(blue/yellow) using only LiDAR point cloud data, built for the IITB 
Racing Formula Student Driverless car.

## Demo

Blue cone (left boundary) and yellow cone (right boundary) correctly 
detected and classified in real-time at ~62ms per frame.

## Problem

A Formula Student driverless car needs to know where the track 
boundaries are. Blue cones mark the left boundary, yellow cones mark 
the right. Given only raw LiDAR point cloud data, detect all cones and 
output:
- (x, y) coordinates in meters
- Color: blue or yellow
- Real-time visualization in RViz

## Pipeline
```
LiDAR PointCloud2
          ↓
Multi-stage filtering:
  - Height range: -0.2m to 0.3m (cone geometry)
  - Distance: 1m to 150m
  - Lateral: ±5m
  - Forward only (x > 0)
          ↓
DBSCAN Clustering (eps=0.4, min_samples=6)
          ↓
Cluster validation:
  - Point count: 6-300
  - Shape check: std(x,y) < 0.4 (rejects non-cone shapes)
  - Minimum height: 0.05m
          ↓
Feature extraction:
  [mean, std, Q25, Q75] of intensity distribution
          ↓
ANN Classification → blue / yellow
          ↓
MarkerArray published to RViz
```

## ANN Architecture (PyTorch)
```
Input: 4 intensity statistics
  [mean, std deviation, 25th percentile, 75th percentile]
          ↓
Linear(4 → 16) + ReLU
          ↓
Linear(16 → 8) + ReLU
          ↓
Linear(8 → 2) → blue (0) or yellow (1)
```

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.01) |
| Loss | CrossEntropy |
| Train/test split | 80/20 |
| Epochs | 50 |
| Test Accuracy | **97%** |

## Key Design Decisions

**Why DBSCAN over KMeans?**
Cone count per frame is unknown. DBSCAN automatically determines 
cluster count and rejects noise points (label = -1). KMeans requires 
specifying k in advance.

**Why intensity statistics as features?**
Blue cones have white stripes → high intensity LiDAR reflection.
Yellow cones have black stripes → low intensity reflection.
[mean, std, Q1, Q3] captures the intensity distribution robustly 
without needing raw point sequences.

**Why multi-stage filtering before clustering?**
Reduces noise drastically — only points consistent with cone geometry 
are passed to DBSCAN. Improves both speed and accuracy significantly.

**Why shape validation after clustering?**
DBSCAN can merge nearby non-cone objects into one cluster. Checking 
std deviation of x,y rejects elongated or irregular clusters that 
aren't cones.

## Results
```
[INFO] [cone_detector]:  Cone: (1.24, -1.55), Color: yellow, Range: 1.99m
[INFO] [cone_detector]:  Cone: (1.00, 1.45), Color: blue, Range: 1.76m
[INFO] [cone_detector]:  Frame processed in 62.42 ms
```

- **97% classification accuracy** on test data
- **~62ms processing time** per frame (real-time capable)
- Correct spatial placement — blue left (y > 0), yellow right (y < 0)
- Built-in per-range accuracy tracking logged every 30 detections

## Repository Structure
```
perception_pkg/
├── perception_pkg/
│   └── cone_detector.py        # Main ROS2 node
├── ann_model/
│   ├── train_ann.py            # ANN training script
│   ├── extract_dataset.py      # Dataset extraction
│   ├── dataset.npy.npz         # Training data
│   └── model_ann.pth           # Saved model weights
└── package.xml
```

## How to Run
```bash
# Build
cd ~/your_ws
colcon build
source install/setup.bash

# Run detector (with your topic)
ros2 run perception_pkg cone_detector --ros-args -p lidar_topic:=/carmaker/points

# Play rosbag in second terminal
ros2 bag play <your_bag_file>

# Visualize in RViz
rviz2
# Fixed Frame: velodyne
# Add → By Topic → /detected_cones (MarkerArray)
```

## Dependencies

- ROS2 Humble
- PyTorch
- scikit-learn (DBSCAN)
- numpy
