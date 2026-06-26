import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    slemslam_pkg = get_package_share_directory('slemslam')
    slam_params_file = os.path.join(slemslam_pkg, 'config', 'slam_toolbox_config.yaml')

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_slam',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'autostart': True},
            {'node_names': ['/drifting/slam_toolbox', '/gt/slam_toolbox']},
            {'bond_timeout': 0.0}
        ]
    )

    odom_corruptor = Node(
        package='slemslam',
        executable='odom_corruptor',
        name='odom_corruptor',
        parameters=[{'use_sim_time': True}]
    )

    lidar_merger = Node(
        package='slemslam',
        executable='lidar_merger',
        name='lidar_merger',
        parameters=[{'use_sim_time': True}]
    )


    world_to_noisy_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_map_drifting',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0', 
            '--yaw', '0', '--pitch', '0', '--roll', '0', 
            '--frame-id', 'world', 
            '--child-frame-id', 'map_drifting_frame'
        ],
        parameters=[{'use_sim_time': True}]
    )


    slam_drifting = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        namespace='drifting',
        remappings=[
            ('/scan', '/scan_drifting'),
            ('/map', '/map_drifting'),
            ('/map_metadata', '/map_drifting_metadata')
        ],
        parameters=[
            slam_params_file,
            {'use_sim_time': True},
            {'odom_frame': 'noisy_odom'},
            {'base_frame': 'noisy_base_link'},
            {'map_frame': 'map_drifting_frame'},
            {'resolution': 0.05},
            {'max_laser_range': 20.0}
        ]
    )


    slam_gt = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        namespace='gt',
        remappings=[
            ('/scan', '/scan_gt'),
            ('/map', '/map_gt'),
            ('/map_metadata', '/map_gt_metadata')
        ],
        parameters=[
            slam_params_file,
            {'use_sim_time': True},
            {'odom_frame': 'world'},
            {'base_frame': 'base_link_gt'},
            {'map_frame': 'map_gt_frame'},
            {'resolution': 0.05},
            {'max_laser_range': 20.0}
        ]
    )


    return LaunchDescription([
        odom_corruptor,
        lidar_merger,
        world_to_noisy_odom,
        slam_drifting,
        slam_gt,
        lifecycle_manager
    ])