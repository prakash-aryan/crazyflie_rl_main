#!/usr/bin/env python3
"""
Crazyflie 2 Reinforcement Learning - Extended Training Version
Conservative improvements for better long-term hovering
"""

import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import mujoco
from mujoco import viewer
import numpy as np
import time
import logging
import threading
import math
import random
from collections import namedtuple, deque
from itertools import count
import warnings

# Suppress additional warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import TensorBoard after setting environment variables
from torch.utils.tensorboard import SummaryWriter

# Global simulation lock for thread safety
sim_lock = threading.RLock()

# Configuration and Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Create necessary directories
os.makedirs('logs/crazyflie_hover_rl', exist_ok=True)
os.makedirs('models', exist_ok=True)

writer = SummaryWriter('logs/crazyflie_hover_rl')

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global simulation objects (shared between training and viewer)
model = None
data = None
viewer_handle = None

def initialize_simulation():
    """Initialize global simulation objects"""
    global model, data
    with sim_lock:
        try:
            # Try to load scene with DQN-specific settings, fallback to default
            model = mujoco.MjModel.from_xml_path('scene_dueling_dqn.xml')
        except:
            model = mujoco.MjModel.from_xml_path('scene.xml')
        
        data = mujoco.MjData(model)
        
        # Set physics parameters
        model.opt.timestep = 0.02
        model.opt.iterations = 10
        model.opt.tolerance = 1e-2
        
        reset_simulation()

def reset_simulation(randomize=True):
    """Reset simulation to initial state with optional randomization"""
    global model, data
    with sim_lock:
        try:
            mujoco.mj_resetData(model, data)
            
            if randomize:
                # Randomized starting position for better generalization
                data.qpos[0] = np.random.uniform(-0.15, 0.15)    # x
                data.qpos[1] = np.random.uniform(-0.15, 0.15)    # y  
                data.qpos[2] = np.random.uniform(0.85, 1.15)     # z
                
                # Small random orientation
                angle = np.random.uniform(-0.05, 0.05)
                data.qpos[3] = np.cos(angle/2)  # qw
                data.qpos[4] = np.sin(angle/2) * np.random.uniform(-0.1, 0.1)  # qx
                data.qpos[5] = np.sin(angle/2) * np.random.uniform(-0.1, 0.1)  # qy
                data.qpos[6] = 0.0  # qz
            else:
                # Fixed starting position
                data.qpos[0] = 0.0    # x
                data.qpos[1] = 0.0    # y  
                data.qpos[2] = 1.0    # z - at target height
                data.qpos[3] = 1.0    # qw (quaternion)
                data.qpos[4] = 0.0    # qx
                data.qpos[5] = 0.0    # qy
                data.qpos[6] = 0.0    # qz
            
            # Small random velocities if randomizing
            if randomize:
                data.qvel[:] = np.random.uniform(-0.1, 0.1, size=data.qvel.shape)
            else:
                data.qvel[:] = 0.0
            
            # Initial control with slight randomization
            data.ctrl[0] = 0.26 + (np.random.uniform(-0.01, 0.01) if randomize else 0)
            data.ctrl[1] = 0.0
            data.ctrl[2] = 0.0
            data.ctrl[3] = 0.0
            
            # Forward kinematics
            mujoco.mj_forward(model, data)
            
        except Exception as e:
            print(f"Error in reset_simulation: {e}")
            data.qpos[:] = 0.0
            data.qpos[2] = 1.0
            data.qpos[3] = 1.0
            data.qvel[:] = 0.0
            data.ctrl[0] = 0.26
            data.ctrl[1:] = 0.0

# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN Network
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Environment Functions
def get_state():
    """Get current state vector with safety checks"""
    global model, data
    with sim_lock:
        try:
            # Position (3)
            pos = data.qpos[0:3].copy()
            # Velocity (3)  
            vel = data.qvel[0:3].copy()
            # Orientation quaternion (4)
            quat = data.qpos[3:7].copy()
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 0:
                quat = quat / quat_norm
            else:
                quat = np.array([1, 0, 0, 0])
            
            # Target relative position (3)
            target_pos = np.array([0.0, 0.0, 1.0])
            rel_pos = target_pos - pos
            
            # Build state vector (13 dimensions total)
            state = np.concatenate([
                pos,           # 3: current position
                vel,           # 3: velocity
                quat,          # 4: orientation quaternion
                rel_pos        # 3: relative position to target
            ])
            
            return state.astype(np.float32)
            
        except Exception as e:
            print(f"Error in get_state: {e}")
            return np.zeros(13, dtype=np.float32)

def calculate_reward(step_count=0):
    """Calculate reward based on hovering performance"""
    global model, data
    with sim_lock:
        try:
            pos = data.qpos[0:3].copy()
            vel = data.qvel[0:3].copy()
            quat = data.qpos[3:7].copy()
            
            # Target position
            target = np.array([0.0, 0.0, 1.0])
            
            # Position error
            position_error = np.linalg.norm(pos - target)
            xy_error = np.linalg.norm(pos[:2] - target[:2])
            
            # Velocity magnitude
            velocity_magnitude = np.linalg.norm(vel)
            vertical_velocity = abs(vel[2])
            horizontal_velocity = np.linalg.norm(vel[:2])
            
            # Orientation penalty (deviation from upright)
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 0:
                quat = quat / quat_norm
            else:
                quat = np.array([1, 0, 0, 0])
            
            # Calculate "uprightness" - how close to vertical orientation
            upright = 2 * (quat[0]**2 + quat[3]**2) - 1  # This should be close to 1 for upright
            upright_penalty = max(0, 1 - upright)
            
            # Base reward: exponential in distance for precision
            reward = 50.0 * np.exp(-5 * position_error)
            
            # Velocity penalty (we want the drone to hover, not move)
            reward -= velocity_magnitude * 2.0
            reward -= vertical_velocity * 3.0  # Extra penalty for vertical movement
            reward -= horizontal_velocity * 2.0
            
            # Stability bonus - increased
            if velocity_magnitude < 0.1 and position_error < 0.2:
                reward += 20.0  # Increased bonus for stable hovering
            
            # Time bonus - small reward for surviving longer
            reward += min(step_count * 0.05, 5.0)  # Cap at 5.0
            
            # Orientation penalty
            reward -= upright_penalty * 5.0
            
            # Terminal penalties
            if pos[2] < 0.3 or pos[2] > 3.0:
                reward = -200.0
            elif xy_error > 2.0:
                reward = -150.0
            
            return reward
            
        except Exception as e:
            print(f"Error in calculate_reward: {e}")
            return 0.0

def is_done():
    """Check if episode should end"""
    global data
    
    pos = data.qpos[0:3]
    
    if not np.all(np.isfinite(pos)):
        return True
    
    # More lenient bounds to allow learning
    if pos[2] < 0.3 or pos[2] > 3.0:
        return True
    
    if abs(pos[0]) > 2.0 or abs(pos[1]) > 2.0:
        return True
    
    return False

def apply_action(action):
    """Apply action with smaller steps for finer control"""
    global model, data
    
    with sim_lock:
        try:
            current_thrust = data.ctrl[0]
            
            # Smaller action steps for finer control
            thrust_step = 0.005  # Reduced from 0.01
            moment_step = 0.01   # Reduced from 0.02
            
            if action == 0:  # Increase thrust
                data.ctrl[0] = np.clip(current_thrust + thrust_step, 0.20, 0.32)
            elif action == 1:  # Decrease thrust
                data.ctrl[0] = np.clip(current_thrust - thrust_step, 0.20, 0.32)
            elif action == 2:  # Pitch forward
                data.ctrl[1] = np.clip(data.ctrl[1] + moment_step, -0.05, 0.05)
            elif action == 3:  # Pitch backward
                data.ctrl[1] = np.clip(data.ctrl[1] - moment_step, -0.05, 0.05)
            elif action == 4:  # Roll left
                data.ctrl[2] = np.clip(data.ctrl[2] + moment_step, -0.05, 0.05)
            elif action == 5:  # Roll right
                data.ctrl[2] = np.clip(data.ctrl[2] - moment_step, -0.05, 0.05)
            elif action == 6:  # Stabilize (reduce all moments)
                data.ctrl[1] *= 0.7
                data.ctrl[2] *= 0.7
                data.ctrl[3] *= 0.7
            
            # Stronger moment decay for stability
            data.ctrl[1] = np.clip(data.ctrl[1] * 0.85, -0.05, 0.05)
            data.ctrl[2] = np.clip(data.ctrl[2] * 0.85, -0.05, 0.05)
            data.ctrl[3] = np.clip(data.ctrl[3] * 0.85, -0.05, 0.05)
            
            # Step simulation
            for _ in range(2):
                if np.any(np.abs(data.qvel) > 50):
                    print("High velocity detected, resetting...")
                    reset_simulation()
                    return
                    
                try:
                    mujoco.mj_step(model, data)
                except Exception as e:
                    print(f"Step error: {e}, resetting...")
                    reset_simulation()
                    return
            
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                print("Simulation became unstable, resetting...")
                reset_simulation()
                
        except Exception as e:
            print(f"Error in apply_action: {e}")
            reset_simulation()

# Training Agent
class CrazyflieAgent:
    def __init__(self):
        self.batch_size = 64
        self.gamma = 0.99      # Higher discount factor
        self.eps_start = 0.9
        self.eps_end = 0.08    # Slightly higher minimum exploration
        self.eps_decay = 1500  # Slower decay for 5k episodes
        self.tau = 0.005
        self.lr = 1e-4         # Lower learning rate
        
        self.n_actions = 7
        self.n_observations = 13
        
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayMemory(10000)
        
        self.steps_done = 0
        
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1.0 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.tensor([state], device=device, dtype=torch.float32)
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
            
        try:
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                        if s is not None])
            
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)
            
            next_state_values = torch.zeros(self.batch_size, device=device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            
            self.optimizer.zero_grad()
            loss.backward()
            
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            
            # Log metrics
            if self.steps_done % 100 == 0:
                writer.add_scalar('Loss', loss.item(), self.steps_done)
                current_eps = self.eps_end + (self.eps_start - self.eps_end) * \
                    math.exp(-1.0 * self.steps_done / self.eps_decay)
                writer.add_scalar('Epsilon', current_eps, self.steps_done)
                
        except Exception as e:
            print(f"Error in optimize_model: {e}")
    
    def soft_update_target_network(self):
        try:
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + \
                                           target_net_state_dict[key] * (1 - self.tau)
            self.target_net.load_state_dict(target_net_state_dict)
        except Exception as e:
            print(f"Error updating target network: {e}")

# Global flag for training status
training_active = True

def training_loop():
    """Main training loop for 5000 episodes"""
    global training_active
    
    agent = CrazyflieAgent()
    
    num_episodes = 5000  # Extended training
    max_steps_per_episode = 400  # Allow longer episodes
    
    best_reward = -float('inf')
    best_avg_reward = -float('inf')
    reward_history = deque(maxlen=100)
    
    for episode in range(num_episodes):
        if not training_active:
            break
            
        try:
            reset_simulation(randomize=True)
            time.sleep(0.05)
            
            state = get_state()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps_per_episode):
                if not training_active:
                    break
                    
                action_tensor = agent.select_action(state)
                action = action_tensor.item()
                
                apply_action(action)
                
                next_state = get_state()
                reward = calculate_reward(step_count=step)
                done = is_done()
                
                episode_reward += reward
                episode_steps += 1
                
                state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
                reward_tensor = torch.tensor([reward], device=device)
                
                if done:
                    next_state_tensor = None
                else:
                    next_state_tensor = torch.tensor([next_state], dtype=torch.float32, device=device)
                
                agent.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)
                
                state = next_state
                
                agent.optimize_model()
                agent.soft_update_target_network()
                
                if done:
                    break
                    
                time.sleep(0.01)
            
            # Episode completed
            reward_history.append(episode_reward)
            avg_reward = np.mean(reward_history) if len(reward_history) > 0 else episode_reward
            
            # Logging
            writer.add_scalar('Episode_Reward', episode_reward, episode)
            writer.add_scalar('Episode_Steps', episode_steps, episode)
            writer.add_scalar('Avg_Reward_100', avg_reward, episode)
            
            if episode % 50 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Steps: {episode_steps}, "
                      f"Avg100: {avg_reward:.2f}, Exploration: "
                      f"{(agent.eps_end + (agent.eps_start - agent.eps_end) * math.exp(-1.0 * agent.steps_done / agent.eps_decay)):.3f}")
            
            # Save best models
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save({
                    'episode': episode,
                    'policy_net_state_dict': agent.policy_net.state_dict(),
                    'target_net_state_dict': agent.target_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'reward': episode_reward
                }, 'models/crazyflie_hover_best.pth')
            
            if len(reward_history) >= 100 and avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save({
                    'episode': episode,
                    'policy_net_state_dict': agent.policy_net.state_dict(),
                    'target_net_state_dict': agent.target_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'reward': avg_reward
                }, 'models/crazyflie_hover_best_avg.pth')
            
            # Regular checkpoint
            if episode % 250 == 0 and episode > 0:
                torch.save({
                    'episode': episode,
                    'policy_net_state_dict': agent.policy_net.state_dict(),
                    'target_net_state_dict': agent.target_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'reward': episode_reward
                }, f'models/crazyflie_hover_checkpoint_{episode}.pth')
            
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            reset_simulation()
            time.sleep(1.0)
            continue
    
    training_active = False
    print("Training completed!")
    print(f"Best single episode reward: {best_reward:.2f}")
    print(f"Best average reward (100 episodes): {best_avg_reward:.2f}")
    writer.close()

def main():
    """Main function with training loop"""
    global training_active, viewer_handle

    import argparse
    parser = argparse.ArgumentParser(description='Train DQN Crazyflie')
    parser.add_argument('--headless', action='store_true', help='Run without viewer')
    args = parser.parse_args()

    try:
        print("="*60)
        print("Crazyflie Reinforcement Learning Training")
        print("="*60)
        print("Extended DQN Training for Stable Hovering")
        print("- More conservative action steps")
        print("- Enhanced reward structure")
        print("- Improved stability mechanisms")

        initialize_simulation()
        print("Simulation initialized successfully!")

        # Start training in a separate thread
        training_thread = threading.Thread(target=training_loop, daemon=True)
        training_thread.start()

        if args.headless:
            print("Starting extended training for 5000 episodes in headless mode...")
            print("Press Ctrl+C to stop.")
            try:
                training_thread.join()
            except KeyboardInterrupt:
                print("\nStopping training...")
        else:
            print("Starting extended training for 5000 episodes...")
            print("Close the viewer window to stop training.")
            with mujoco.viewer.launch_passive(model, data) as viewer_handle:
                while viewer_handle.is_running() and training_active:
                    with sim_lock:
                        viewer_handle.sync()
                    time.sleep(0.01)

        # Clean shutdown
        training_active = False
        training_thread.join(timeout=5.0)

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()