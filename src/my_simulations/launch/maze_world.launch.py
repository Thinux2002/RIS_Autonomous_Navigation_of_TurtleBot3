#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, EnvironmentVariable, TextSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    world = PathJoinSubstitution([
        FindPackageShare('my_simulations'), 'worlds', 'maze_world.world'  # <-- your world file name
    ])

    # Add BOTH your models and TB3 models to Gazebo's search path (append, don't overwrite)
    my_models  = PathJoinSubstitution([FindPackageShare('my_simulations'), 'models'])
    tb3_models = PathJoinSubstitution([FindPackageShare('turtlebot3_gazebo'), 'models'])
    set_paths = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[
            EnvironmentVariable('GZ_SIM_RESOURCE_PATH', default_value=TextSubstitution(text='')),
            TextSubstitution(text=':'), my_models,
            TextSubstitution(text=':'), tb3_models
        ]
    )

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])
        ]),
        launch_arguments={'gz_args': world}.items()
    )

    spawn_tb3 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('turtlebot3_gazebo'), 'launch', 'spawn_turtlebot3.launch.py'])
        ]),
        launch_arguments={'x_pose': '0.0', 'y_pose': '0.0'}.items()
    )

    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('turtlebot3_gazebo'), 'launch', 'robot_state_publisher.launch.py'])
        ]),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    return LaunchDescription([set_paths, gz, spawn_tb3, rsp])

