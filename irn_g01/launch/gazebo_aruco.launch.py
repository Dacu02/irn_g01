"""Launch file for ArUco marker display in Gazebo Ignition."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description."""
    # Define package name
    pkg_name = 'irn_g01'
    pkg_share = FindPackageShare(pkg_name)

    # Declare arguments
    world_file_arg = DeclareLaunchArgument(
        'world',
        default_value=PathJoinSubstitution([
            pkg_share,
            'worlds',
            'aruco_world.world'
        ]),
        description='Path to the Gazebo world file'
    )

    # Gazebo server
    gz_server = Node(
        package='gz-sim',
        executable='gz',
        arguments=[
            'sim',
            '-r',
            '-s',
            LaunchConfiguration('world')
        ],
        output='screen'
    )

    # Gazebo GUI
    gz_gui = Node(
        package='gz-sim',
        executable='gz',
        arguments=[
            'sim',
            '-g',
            LaunchConfiguration('world')
        ],
        output='screen'
    )

    return LaunchDescription([
        world_file_arg,
        LogInfo(msg=['Launching Gazebo Ignition with world: ', LaunchConfiguration('world')]),
        gz_server,
        gz_gui
    ])
