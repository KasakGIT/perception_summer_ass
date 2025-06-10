import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import sklearn.cluster
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

class ConeDetector(Node):
    def __init__(self):
        super().__init__('cone_detector')
        
        self.declare_parameter('lidar_topic', '/carmaker/pointcloud')
        self.subscription = self.create_subscription(
            PointCloud,
            self.get_parameter('lidar_topic').get_parameter_value().string_value,
            self.pointcloud_callback,
            10)

        self.marker_pub = self.create_publisher(MarkerArray, 'detected_cones', 10)
        self.frame_id = 'Fr1A'

        # Tunable parameters
        self.min_points = 10
        self.max_points = 300
        self.height_range = (0.0, 0.4)
        self.max_distance = 15.0
        self.eps = 0.5
        self.min_samples = 70
        self.intensity_threshold = 2.0
        self.window_size_ratio = 0.2
        self.marker_lifetime = 1  # seconds (reduced flickering)

    def safe_mean(self, arr):
        """Safe mean calculation that handles empty arrays"""
        if len(arr) == 0:
            return np.nan
        return np.mean(arr)

    def pointcloud_callback(self, msg):
        try:
            # Extract points and intensity
            xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
            intensity = None
            for channel in msg.channels:
                if channel.name == "intensity":
                    intensity = np.array(channel.values)
                    break

            if intensity is None or len(intensity) != len(xyz):
                self.get_logger().warn("Intensity channel missing or size mismatch.", throttle_duration_sec=1.0)
                return

            # Apply transform from LiDAR to Fr1A frame
            xyz[:, 0] += 2.921  # x offset (forward)
            xyz[:, 2] += 0.1629  # z offset (height)

            # Filter points
            mask = (xyz[:, 2] > self.height_range[0]) & (xyz[:, 2] < self.height_range[1])
            mask &= (np.linalg.norm(xyz[:, :2], axis=1) < self.max_distance)
            filtered_points = xyz[mask]
            filtered_intensity = intensity[mask]

            if len(filtered_points) == 0:
                return

            # Clustering
            dbscan = sklearn.cluster.DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(filtered_points)
            labels = dbscan.labels_
            unique_labels = set(labels)

            marker_array = MarkerArray()
            id_counter = 0

            for label in unique_labels:
                if label == -1:  # Skip noise
                    continue

                cluster_mask = (labels == label)
                cluster_points = filtered_points[cluster_mask]
                cluster_intensity = filtered_intensity[cluster_mask]

                # Skip if not enough points
                if len(cluster_points) < self.min_points or len(cluster_points) > self.max_points:
                    continue

                # Get cone position
                x_mean = np.mean(cluster_points[:, 0])
                y_mean = np.mean(cluster_points[:, 1])
                z_min = np.min(cluster_points[:, 2])

                # Sort points by height
                height_order = np.argsort(cluster_points[:, 2])
                sorted_intensity = cluster_intensity[height_order]
                
                # Dynamic window size
                n = len(sorted_intensity)
                window_size = max(3, int(n * self.window_size_ratio))
                
                # Find most significant intensity transition
                max_diff = 0
                max_diff_pos = n//2  # Default to middle
                valid_transition_found = False
                
                for i in range(window_size, n-window_size):
                    lower = self.safe_mean(sorted_intensity[i-window_size:i])
                    upper = self.safe_mean(sorted_intensity[i:i+window_size])
                    
                    if np.isnan(lower) or np.isnan(upper):
                        continue
                    
                    current_diff = abs(upper - lower)
                    if current_diff > max_diff:
                        max_diff = current_diff
                        max_diff_pos = i
                        valid_transition_found = True

                if not valid_transition_found or max_diff < self.intensity_threshold:
                    continue

                # Calculate stripe and surrounding intensities
                stripe_window = max(1, window_size//2)
                stripe_start = max(0, max_diff_pos - stripe_window)
                stripe_end = min(n, max_diff_pos + stripe_window)
                stripe_value = self.safe_mean(sorted_intensity[stripe_start:stripe_end])
                
                lower_surrounding = self.safe_mean(sorted_intensity[:max_diff_pos-window_size])
                upper_surrounding = self.safe_mean(sorted_intensity[max_diff_pos+window_size:])
                
                if np.isnan(stripe_value) or np.isnan(lower_surrounding) or np.isnan(upper_surrounding):
                    continue
                
                surrounding_value = (lower_surrounding + upper_surrounding) / 2
                is_left = y_mean > 0  # Positive y is left in vehicle frame

                # Determine cone color (INVERTED from previous version)
                is_blue = (stripe_value > surrounding_value + self.intensity_threshold) and is_left
                is_yellow = (stripe_value < surrounding_value - self.intensity_threshold) and not is_left

                if not (is_blue or is_yellow):
                    continue

                color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0) if is_blue else ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)

                # Create marker with proper lifetime handling
                marker = Marker()
                marker.header.frame_id = self.frame_id
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = id_counter
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                marker.pose.position.x = x_mean
                marker.pose.position.y = y_mean
                marker.pose.position.z = z_min + 0.155
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.15
                marker.scale.y = 0.15
                marker.scale.z = 0.45
                marker.color = color
                marker.lifetime.sec = int(self.marker_lifetime)
                marker.lifetime.nanosec = int((self.marker_lifetime % 1) * 1e9)
                marker_array.markers.append(marker)
                id_counter += 1

            self.marker_pub.publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f"Error in callback: {str(e)}", throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = ConeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()







