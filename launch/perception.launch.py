from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='perception_pkg',
            executable='cone_detector',
            name='cone_detector',
            parameters=[{'lidar_topic': '/carmaker/pointcloud'}]
        )
    ])
