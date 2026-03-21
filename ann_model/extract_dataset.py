# perception_pkg/ann_model/extract_dataset.py

import numpy as np
import os
import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
import sqlite3
import struct

def extract_clusters_from_rosbag(bag_path, topic_name="/velodyne_points"):
    conn = sqlite3.connect(os.path.join(bag_path, "rosbag2_2024_06_18-02_10_29_0.db3"))
    cursor = conn.cursor()

    # Get topic_id
    cursor.execute("SELECT id FROM topics WHERE name=?", (topic_name,))
    topic_id = cursor.fetchone()
    if topic_id is None:
        raise RuntimeError(f"Topic {topic_name} not found in rosbag.")
    topic_id = topic_id[0]

    # Get messages
    cursor.execute("SELECT data FROM messages WHERE topic_id=?", (topic_id,))
    rows = cursor.fetchall()

    all_features = []
    all_labels = []

    for row in rows:
        raw = row[0]
        msg = deserialize_message(raw, PointCloud2)

        # Extract points
        cloud = pc2.read_points(msg, field_names=["x", "y", "z", "intensity"], skip_nans=True)
        cloud_array = np.array(list(cloud))
        if cloud_array.shape[0] == 0:
            continue

        xyz = np.vstack((cloud_array['x'], cloud_array['y'], cloud_array['z'])).T
        intensity = cloud_array['intensity']

        # Filter
        mask = (xyz[:, 2] > -0.2) & (xyz[:, 2] < 0.3)
        mask &= (np.linalg.norm(xyz[:, :2], axis=1) > 1.0)
        mask &= (np.linalg.norm(xyz[:, :2], axis=1) < 30.0)
        filtered_points = xyz[mask]
        filtered_intensity = np.array(intensity)[mask]

        if len(filtered_points) == 0:
            continue

        db = DBSCAN(eps=0.4, min_samples=6).fit(filtered_points)
        labels = db.labels_

        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            cluster_mask = labels == cluster_id
            cluster_pts = filtered_points[cluster_mask]
            cluster_int = filtered_intensity[cluster_mask]

            if len(cluster_pts) < 6:
                continue

            y_mean = np.mean(cluster_pts[:, 1])
            z_min = np.min(cluster_pts[:, 2])
            z_max = np.max(cluster_pts[:, 2])
            if (z_max - z_min) < 0.05:
                continue

            # Auto label based on y position
            label = 0 if y_mean > 0 else 1  # 0 = blue, 1 = yellow

            # Features
            cluster_int = np.clip(cluster_int, 0, 255)
            mean = np.mean(cluster_int)
            std = np.std(cluster_int)
            q1 = np.percentile(cluster_int, 25)
            q3 = np.percentile(cluster_int, 75)

            all_features.append([mean, std, q1, q3])
            all_labels.append(label)

    print(f"✅ Total clusters: {len(all_features)}")
    conn.close()

    all_features = np.array(all_features)
    all_labels = np.array(all_labels).reshape(-1,1)
    data = np.hstack((all_features, all_labels))  # shape (N, 5)
    output_path = "src/perception_pkg/ann_model/dataset.npz"
    np.savez(output_path, data)
    print(f"✅ Saved dataset.npz at {output_path} with shape:", data.shape)


if __name__ == '__main__':
    bag_path = "/home/kasak/Downloads/rosbag2_2024_06_18-02_10_29"
    extract_clusters_from_rosbag(bag_path)
