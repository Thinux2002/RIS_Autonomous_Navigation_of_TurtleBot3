from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os

def generate_launch_description():
    return LaunchDescription([
        # 1. Launch Gazebo with maze world
        ExecuteProcess(
            cmd=['gz', 'sim', '-v', '4', os.path.expanduser('~/turtlebot3_ws/src/my_simulations/worlds/maze_world.world')],
            output='screen'
        ),

        # 2. Spawn TurtleBot3
        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', 'tb3',
                '-topic', 'robot_description',
                '-x', '0.0', '-y', '0.0', '-z', '0.1'
            ],
            output='screen'
        ),

        # 3. Robot state publisher (URDF)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'robot_description': open(
                    os.path.expanduser('~/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/urdf/turtlebot3_burger.urdf.xacro')
                ).read()
            }]
        ),

        # 4. DQN training node
        Node(
            package='turtlebot3_dqn',
            executable='dqn_node',
            name='dqn_trainer',
            output='screen'
        ),
    ])