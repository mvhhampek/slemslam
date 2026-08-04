import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # bag topics
    input_odom_arg = DeclareLaunchArgument('input_odom_topic', default_value='/tesse/odom', description='Input dataset odometry topic')
    rgb_topic_arg = DeclareLaunchArgument('rgb_topic', default_value='/tesse/left_cam/rgb/image_raw', description='RGB camera image topic')
    depth_topic_arg = DeclareLaunchArgument('depth_topic', default_value='/tesse/depth_cam/mono/image_raw', description='Depth camera image topic')
    camera_info_arg = DeclareLaunchArgument('camera_info_topic', default_value='/tesse/left_cam/camera_info', description='Camera info topic')

    odom_corruptor = Node(
        package='slemslam',
        executable='odom_corruptor',
        name='odom_corruptor',
        parameters=[{
            'use_sim_time': True,
            'input_odom_topic': LaunchConfiguration('input_odom_topic')
        }]
    )

    semantic_detector = Node(
        package='slemslam',
        executable='semantic_detector',
        name='semantic_detector',
        parameters=[{
            'use_sim_time': True,
            'rgb_topic': LaunchConfiguration('rgb_topic'),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic')
        }]
    )

    semantic_backend = Node(
        package='slemslam',
        executable='semantic_backend',
        name='semantic_backend',
        parameters=[{
            'use_sim_time': True
        }]
    )

    return LaunchDescription([
        input_odom_arg,
        rgb_topic_arg,
        depth_topic_arg,
        camera_info_arg,
        odom_corruptor,
        semantic_detector,
        semantic_backend
    ])