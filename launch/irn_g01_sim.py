import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    pkg_moving_aruco = get_package_share_directory('moving_aruco')
    pkg_turtlebot4 = get_package_share_directory('turtlebot4_ignition_bringup')

    # Include moving_aruco.launch.py (imposta il path)
    moving_aruco_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_moving_aruco, 'launch', 'moving_aruco.launch.py'])
        ])
    )
    turtlebot4_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_turtlebot4, 'launch', 'turtlebot4_ignition.launch.py'])
        ]),
        launch_arguments=[
            ('nav2', 'true'),
            ('slam', 'false'),
            ('localization', 'true'),
            ('rviz', 'true'),
        ]
    )

    # Nodo aruco_reader
    aruco_reader_node = Node(
        package='irn_g01',
        executable='aruco_reader',
        name='aruco_reader',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )
    
    core = Node(
        package='irn_g01',
        executable='core',
        name='core',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )

    ld = LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              choices=['true', 'false'],
                              description='use_sim_time'),
        DeclareLaunchArgument('world_name', default_value='square',
                              description='Ignition World Name'),
    ])
    ld.add_action(turtlebot4_launch)
    ld.add_action(moving_aruco_launch)
    ld.add_action(aruco_reader_node)
    ld.add_action(core)