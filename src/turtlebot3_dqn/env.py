import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import gym
from gym import spaces

class Turtlebot3MazeEnv(Node, gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        Node.__init__(self, 'turtlebot3_maze_env')
        gym.Env.__init__(self)

        # Action space: 3 discrete actions
        self.action_space = spaces.Discrete(3)  # 0: forward, 1: left, 2: right

        # Observation: 24 laser rays (downsampled to 8) + goal distance + angle
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(10,), dtype=np.float32)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        self.scan = None
        self.pose = np.zeros(3)  # x, y, theta
        self.goal = np.array([4.0, 4.0])   # maze exit
        self.max_linear = 0.22
        self.max_angular = 1.0
        self.step_count = 0
        self.max_steps = 1000

    def scan_cb(self, msg: LaserScan):
        self.scan = np.array(msg.ranges)
        self.scan = np.clip(self.scan, 0.0, 10.0)

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pose[0] = p.x
        self.pose[1] = p.y
        yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.pose[2] = yaw

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Reset robot via Gazebo service (you can also respawn via launch)
        self.get_logger().info("Resetting environment...")
        self.step_count = 0
        self._publish_cmd(0.0, 0.0)
        rclpy.spin_once(self, timeout_sec=1.0)
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        linear, angular = self._action_to_vel(action)
        self._publish_cmd(linear, angular)
        rclpy.spin_once(self, timeout_sec=0.1)

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        info = {}
        return obs, reward, done, False, info

    def _action_to_vel(self, a):
        if a == 0:   # forward
            return self.max_linear, 0.0
        elif a == 1: # turn left
            return 0.0, self.max_angular
        elif a == 2: # turn right
            return 0.0, -self.max_angular

    def _publish_cmd(self, lin, ang):
        msg = Twist()
        msg.linear.x = lin
        msg.angular.z = ang
        self.cmd_pub.publish(msg)

    def _get_obs(self):
        if self.scan is None:
            return np.zeros(10, dtype=np.float32)

        # Downsample 24 rays → 8
        rays = self.scan[::3][:8]
        rays = np.where(np.isinf(rays), 10.0, rays)

        # Goal distance & angle
        dx = self.goal[0] - self.pose[0]
        dy = self.goal[1] - self.pose[1]
        dist = np.hypot(dx, dy)
        angle_to_goal = np.arctan2(dy, dx) - self.pose[2]
        angle_to_goal = (angle_to_goal + np.pi) % (2*np.pi) - np.pi

        return np.concatenate([rays, [dist, angle_to_goal]]).astype(np.float32)

    def _compute_reward(self):
        dist = np.hypot(self.goal[0] - self.pose[0], self.goal[1] - self.pose[1])
        collision = np.any(self.scan < 0.25) if self.scan is not None else False

        reward = -0.01  # time penalty
        if collision:
            reward -= 5.0
        if dist < 0.3:
            reward += 100.0
        else:
            reward -= dist * 0.1
        return reward

    def _is_done(self):
        dist = np.hypot(self.goal[0] - self.pose[0], self.goal[1] - self.pose[1])
        collision = np.any(self.scan < 0.25) if self.scan is not None else False
        return dist < 0.3 or collision or self.step_count >= self.max_steps