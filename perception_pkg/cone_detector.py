import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from sklearn.cluster import DBSCAN
from std_msgs.msg import ColorRGBA
import time
from collections import defaultdict
import os
import torch
import torch.nn as nn
from ament_index_python.packages import get_package_share_directory


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ConeDetector(Node):
    def __init__(self):
        super().__init__('cone_detector')

        self.declare_parameter('lidar_topic', '/velodyne_points')
        self.lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value

        self.subscription = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.pointcloud_callback,
            10)

        self.marker_pub = self.create_publisher(MarkerArray, 'detected_cones', 10)
        self.frame_id = 'Lidar_F'

        self.min_points = 6
        self.max_points = 300
        self.height_range = (-0.2, 0.3)
        self.max_distance = 150.0
        self.min_distance = 1.0
        self.eps = 0.4
        self.min_samples = 6
        self.marker_lifetime = 0.2
        self.min_cone_height = 0.05

        self.range_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.total_cones_detected = 0

        # Load ANN model
        self.model = ANN() # make an empty shell of ann
        model_path = os.path.join(get_package_share_directory('perception_pkg'), 'ann_model', 'model_ann.pth') # load the learned wts during training
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # put them in the ann shell
        self.model.eval() # turn off dropout for consistency

    def pointcloud_callback(self, msg):
        try:
            self.get_logger().info("PointCloud2 callback received")

            start_time = time.time()
            cloud = pc2.read_points(msg, field_names=["x", "y", "z", "intensity"], skip_nans=True)
            cloud_array = np.array(list(cloud))

            if cloud_array.shape[0] == 0:
                self.get_logger().warn("⚠️ PointCloud is empty.")
                return

            xyz = np.vstack((cloud_array['x'], cloud_array['y'], cloud_array['z'])).T
            intensity = cloud_array['intensity']

            mask = (xyz[:, 2] > self.height_range[0]) & (xyz[:, 2] < self.height_range[1])
            mask &= (np.linalg.norm(xyz[:, :2], axis=1) < self.max_distance)
            mask &= (np.linalg.norm(xyz[:, :2], axis=1) > self.min_distance)
            mask &= (xyz[:, 1] > -5.0) & (xyz[:, 1] < 5.0)
            mask &= (xyz[:, 0] > 0.0)

            filtered_points = xyz[mask]
            filtered_intensity = np.array(intensity)[mask]

            if len(filtered_points) == 0:
                self.get_logger().warn("⚠️ No points after filtering.")
                return

            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(filtered_points)
            labels = dbscan.labels_

            marker_array = MarkerArray()
            id_counter = 0

            for label in set(labels):
                if label == -1:
                    continue

                cluster_mask = (labels == label)
                cluster_points = filtered_points[cluster_mask]
                cluster_intensity = filtered_intensity[cluster_mask]

                if len(cluster_points) < self.min_points:
                    continue

                if np.std(cluster_points[:, 0]) > 0.4 or np.std(cluster_points[:, 1]) > 0.4:
                    continue

                x_mean, y_mean = np.mean(cluster_points[:, :2], axis=0)
                z_min = np.min(cluster_points[:, 2])
                z_max = np.max(cluster_points[:, 2])

                if (z_max - z_min) < self.min_cone_height:
                    continue

                # Feature vector: [mean, std, Q1, Q3]
                intens = np.clip(cluster_intensity, 0, 255)
                features = np.array([
                    np.mean(intens),
                    np.std(intens),
                    np.percentile(intens, 25),
                    np.percentile(intens, 75)
                ], dtype=np.float32)

                with torch.no_grad():
                    input_tensor = torch.tensor(features).unsqueeze(0)
                    output = self.model(input_tensor)
                    pred = torch.argmax(output, dim=1).item()

                cone_color = 'blue' if pred == 0 else 'yellow'
                color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0) if cone_color == 'blue' else ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)

                marker = Marker()
                marker.header.frame_id = self.frame_id
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = id_counter
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                marker.pose.position.x = float(x_mean)
                marker.pose.position.y = float(y_mean)
                marker.pose.position.z = float(z_min + (z_max - z_min)/2 + 0.5/2)
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = z_max - z_min + 0.5
                marker.color = color
                marker.lifetime.sec = int(self.marker_lifetime)
                marker.lifetime.nanosec = int((self.marker_lifetime % 1) * 1e9)

                marker_array.markers.append(marker)
                id_counter += 1

                range_m = np.linalg.norm([x_mean, y_mean])
                r_bin = int(round(range_m))
                ground_truth = 'yellow' if y_mean < 0 else 'blue'

                self.range_stats[r_bin]['total'] += 1
                if cone_color == ground_truth:
                    self.range_stats[r_bin]['correct'] += 1

                self.total_cones_detected += 1
                self.get_logger().info(f" Cone: ({x_mean:.2f}, {y_mean:.2f}), Color: {cone_color}, Range: {range_m:.2f}m")

            if self.total_cones_detected % 30 == 0 and self.total_cones_detected > 0:
                self.get_logger().info(" Accuracy Report:")
                for r in sorted(self.range_stats):
                    stats = self.range_stats[r]
                    acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                    self.get_logger().info(f"Range {r}m: Accuracy = {acc:.2f} ({stats['correct']}/{stats['total']})")

            self.marker_pub.publish(marker_array)

            latency = (time.time() - start_time) * 1000
            self.get_logger().info(f" Frame processed in {latency:.2f} ms")

        except Exception as e:
            self.get_logger().error(f" Error in callback: {str(e)}", throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = ConeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



