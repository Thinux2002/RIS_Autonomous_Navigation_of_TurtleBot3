#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .env import Turtlebot3MazeEnv
from .utils import ReplayBuffer
import os

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def main(args=None):
    rclpy.init(args=args)
    env = Turtlebot3MazeEnv()
    rclpy.spin_once(env, timeout_sec=1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(10, 3).to(device)
    target_net = DQN(10, 3).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = ReplayBuffer(10000)

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 64
    target_update = 10
    episodes = 500

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            if np.random.rand() < epsilon:
                action = np.random.randint(3)
            else:
                action = policy_net(state_tensor).argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1

            if len(memory) > batch_size:
                s, a, r, ns, d = memory.sample(batch_size)
                s = torch.FloatTensor(s).to(device)
                a = torch.LongTensor(a).to(device).unsqueeze(1)
                r = torch.FloatTensor(r).to(device)
                ns = torch.FloatTensor(ns).to(device)
                d = torch.FloatTensor(d).to(device)

                q_values = policy_net(s).gather(1, a).squeeze(1)
                next_q = target_net(ns).max(1)[0]
                target = r + gamma * next_q * (1 - d)

                loss = nn.MSELoss()(q_values, target.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        env.get_logger().info(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

        # Save model every 50 episodes
        if (ep+1) % 50 == 0:
            path = os.path.join(os.path.expanduser("~"), "turtlebot3_dqn_models")
            os.makedirs(path, exist_ok=True)
            torch.save(policy_net.state_dict(), f"{path}/dqn_ep{ep+1}.pth")

    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()