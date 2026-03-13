#!/usr/bin/env python3
"""
Crazyflie Dueling DQN Training
Separates state value and action advantage for better learning
"""

import os
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
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Global simulation lock
sim_lock = threading.RLock()

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
os.makedirs('logs/crazyflie_dueling_dqn', exist_ok=True)
os.makedirs('models', exist_ok=True)
writer = SummaryWriter('logs/crazyflie_dueling_dqn')

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global simulation objects
model = None
data = None
viewer_handle = None

def initialize_simulation():
    """Initialize MuJoCo simulation"""
    global model, data
    with sim_lock:
        try:
            model = mujoco.MjModel.from_xml_path('scene_dueling_dqn.xml')
        except:
            model = mujoco.MjModel.from_xml_path('scene.xml')
        
        data = mujoco.MjData(model)
        
        # Physics settings
        model.opt.timestep = 0.02
        model.opt.iterations = 10
        model.opt.tolerance = 1e-2
        
        reset_simulation()

def reset_simulation(randomize=True, difficulty=0.5):
    """Reset with curriculum learning"""
    global model, data
    with sim_lock:
        try:
            mujoco.mj_resetData(model, data)
            
            if randomize:
                # Progressive difficulty
                pos_range = 0.05 + 0.15 * difficulty
                vel_range = 0.05 + 0.15 * difficulty
                angle_range = 0.02 + 0.08 * difficulty
                
                data.qpos[0] = np.random.uniform(-pos_range, pos_range)
                data.qpos[1] = np.random.uniform(-pos_range, pos_range)
                data.qpos[2] = np.random.uniform(0.85, 1.15)
                
                # Small random orientation
                angle = np.random.uniform(-angle_range, angle_range)
                axis = np.random.randn(3)
                axis = axis / np.linalg.norm(axis)
                
                data.qpos[3] = np.cos(angle/2)
                data.qpos[4] = np.sin(angle/2) * axis[0]
                data.qpos[5] = np.sin(angle/2) * axis[1]
                data.qpos[6] = np.sin(angle/2) * axis[2]
                
                data.qvel[:] = np.random.uniform(-vel_range, vel_range, size=data.qvel.shape)
            else:
                data.qpos[:3] = [0.0, 0.0, 1.0]
                data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                data.qvel[:] = 0.0
            
            data.ctrl[0] = 0.26
            data.ctrl[1:] = 0.0
            
            mujoco.mj_forward(model, data)
            
        except Exception as e:
            print(f"Error in reset: {e}")

# Dueling DQN Network
class DuelingDQN(nn.Module):
    """Dueling DQN that separates state value and action advantage"""
    def __init__(self, n_observations, n_actions, hidden_dim=256):
        super(DuelingDQN, self).__init__()
        
        # Shared feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(n_observations, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Value stream - estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Advantage stream - estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, n_actions)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = self.feature_layer(x)
        
        # Compute state value V(s)
        value = self.value_stream(features)
        
        # Compute advantage A(s,a)
        advantage = self.advantage_stream(features)
        
        # Combine using dueling architecture formula:
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

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

def get_state():
    """Get current state vector"""
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
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            
            # Angular velocity (3)
            angvel = data.qvel[3:6].copy()
            
            # Combine all features (13 total)
            state = np.concatenate([pos, vel, quat, angvel])
            
            return state.astype(np.float32)
            
        except Exception as e:
            print(f"Error getting state: {e}")
            return np.zeros(13, dtype=np.float32)

def calculate_reward(step_count=0):
    """Dense reward function optimized for dueling architecture"""
    global data
    
    pos = data.qpos[0:3]
    vel = data.qvel[0:3]
    quat = data.qpos[3:7]
    
    # Target position
    target = np.array([0.0, 0.0, 1.0])
    
    # Distance metrics
    position_error = np.linalg.norm(pos - target)
    height_error = abs(pos[2] - 1.0)
    xy_error = np.linalg.norm(pos[0:2])
    
    # Velocity metrics
    velocity_magnitude = np.linalg.norm(vel)
    vertical_velocity = abs(vel[2])
    
    # Orientation metrics
    quat_norm = np.linalg.norm(quat)
    if quat_norm > 0:
        quat = quat / quat_norm
    w, x, y, z = quat
    upright_penalty = np.sqrt(x*x + y*y)
    
    # Dense reward calculation with clear state value hierarchy
    reward = 15.0  # Base state value
    
    # Position rewards (exponential for precision)
    if position_error < 0.05:
        reward += 40.0  # Excellent position
    elif position_error < 0.1:
        reward += 25.0  # Good position
    elif position_error < 0.2:
        reward += 15.0  # Decent position
    else:
        reward += max(0.0, 10.0 - position_error * 10.0)
    
    # Height priority (most important for hovering)
    if height_error < 0.02:
        reward += 25.0
    elif height_error < 0.05:
        reward += 15.0
    elif height_error < 0.1:
        reward += 8.0
    
    # Stability bonuses (action advantages)
    if velocity_magnitude < 0.1:
        reward += 10.0  # Stable
    elif velocity_magnitude < 0.2:
        reward += 5.0   # Somewhat stable
    
    # Combined stability bonus
    if position_error < 0.1 and velocity_magnitude < 0.1:
        reward += 30.0  # Perfect hovering state
    elif position_error < 0.2 and velocity_magnitude < 0.2:
        reward += 15.0  # Good hovering state
    
    # Penalties
    reward -= velocity_magnitude * 5.0
    reward -= vertical_velocity * 8.0
    reward -= upright_penalty * 10.0
    
    # Time bonus for staying alive
    reward += min(step_count * 0.1, 10.0)
    
    # Control effort penalty
    thrust_deviation = abs(data.ctrl[0] - 0.26)
    moment_magnitude = np.linalg.norm(data.ctrl[1:])
    reward -= thrust_deviation * 8.0
    reward -= moment_magnitude * 15.0
    
    # Terminal penalties (clear state value differences)
    if pos[2] < 0.2 or pos[2] > 2.5:
        reward = -200.0  # Very bad state
    elif xy_error > 1.5:
        reward = -100.0  # Bad state
    
    return reward

def is_done():
    """Check termination"""
    global data
    pos = data.qpos[0:3]
    
    if not np.all(np.isfinite(pos)):
        return True
    
    if pos[2] < 0.2 or pos[2] > 2.5:
        return True
    if abs(pos[0]) > 1.5 or abs(pos[1]) > 1.5:
        return True
    
    return False

def apply_action(action):
    """Apply discrete action"""
    global model, data
    
    with sim_lock:
        try:
            current_thrust = data.ctrl[0]
            
            # Refined action mapping for better control
            thrust_step = 0.008  # Slightly larger for more responsive control
            moment_step = 0.015
            
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
                data.ctrl[1] *= 0.6
                data.ctrl[2] *= 0.6
                data.ctrl[3] *= 0.6
            
            # Moment decay for natural stability
            data.ctrl[1] = np.clip(data.ctrl[1] * 0.9, -0.05, 0.05)
            data.ctrl[2] = np.clip(data.ctrl[2] * 0.9, -0.05, 0.05)
            data.ctrl[3] = np.clip(data.ctrl[3] * 0.9, -0.05, 0.05)
            
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

# Dueling DQN Agent
class DuelingDQNAgent:
    def __init__(self):
        self.batch_size = 64
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 2000  # Slower decay for better exploration
        self.tau = 0.005
        self.lr = 1e-4
        
        self.n_actions = 7
        self.n_observations = 13
        
        # Dueling DQN networks
        self.policy_net = DuelingDQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DuelingDQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayMemory(20000)  # Larger buffer
        
        self.steps_done = 0
        
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1).indices.item()
        else:
            return random.randrange(self.n_actions)
    
    def optimize_model(self):
        try:
            if len(self.memory) < self.batch_size:
                return
            
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            
            # Filter out done states
            non_final_mask = torch.tensor(
                tuple(map(lambda s: s is not None, batch.next_state)), 
                device=device, dtype=torch.bool
            )
            
            if non_final_mask.sum() == 0:
                return
                
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            
            # Current Q values
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)
            
            # Next state values using target network
            next_state_values = torch.zeros(self.batch_size, device=device)
            if len(non_final_next_states) > 0:
                with torch.no_grad():
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
            
            # Expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            
            # Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()
            
            # Logging
            if self.steps_done % 1000 == 0:
                writer.add_scalar('Loss/DQN', loss.item(), self.steps_done)
                current_eps = self.eps_end + (self.eps_start - self.eps_end) * \
                    math.exp(-1. * self.steps_done / self.eps_decay)
                writer.add_scalar('Epsilon', current_eps, self.steps_done)
                
                # Log value and advantage separation
                with torch.no_grad():
                    if len(state_batch) > 0:
                        features = self.policy_net.feature_layer(state_batch[:1])
                        value = self.policy_net.value_stream(features)
                        advantage = self.policy_net.advantage_stream(features)
                        writer.add_scalar('Value/StateValue', value.mean().item(), self.steps_done)
                        writer.add_scalar('Value/AdvantageRange', 
                                        (advantage.max() - advantage.min()).item(), self.steps_done)
                
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

# Global training flag
training_active = True

def training_loop():
    """Main Dueling DQN training loop"""
    global training_active
    
    agent = DuelingDQNAgent()
    
    num_episodes = 5000
    max_steps_per_episode = 500
    
    # Training metrics
    episode = 0
    best_reward = -float('inf')
    best_avg_reward = -float('inf')
    reward_history = deque(maxlen=100)
    
    # Curriculum learning stages
    curriculum_stages = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    current_stage = 0
    stage_threshold = 50.0  # Average reward to advance
    
    print(f"Starting Dueling DQN training with {num_episodes} episodes")
    
    while episode < num_episodes and training_active:
        try:
            # Reset environment
            stage_difficulty = curriculum_stages[min(current_stage, len(curriculum_stages) - 1)]
            reset_simulation(randomize=True, difficulty=stage_difficulty)
            
            state = get_state()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps_per_episode):
                if not training_active:
                    break
                
                # Select action
                action = agent.select_action(state)
                action_tensor = torch.tensor([[action]], device=device, dtype=torch.long)
                
                # Apply action
                apply_action(action)
                
                # Get next state and reward
                next_state = get_state()
                reward = calculate_reward(step_count=step)
                done = is_done()
                
                episode_reward += reward
                episode_steps += 1
                
                # Store transition
                state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
                reward_tensor = torch.tensor([reward], device=device)
                
                if done:
                    next_state_tensor = None
                else:
                    next_state_tensor = torch.tensor([next_state], dtype=torch.float32, device=device)
                
                agent.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)
                
                state = next_state
                
                # Optimize model
                if step % 4 == 0:
                    agent.optimize_model()
                
                # Update target network
                agent.soft_update_target_network()
                
                time.sleep(0.02)
                
                if done:
                    break
            
            reward_history.append(episode_reward)
            avg_reward = np.mean(list(reward_history))
            
            # Logging
            writer.add_scalar('Episode_Reward', episode_reward, episode)
            writer.add_scalar('Episode_Steps', episode_steps, episode)
            writer.add_scalar('Average_Reward_100', avg_reward, episode)
            writer.add_scalar('Curriculum_Stage', current_stage, episode)
            
            if episode % 25 == 0:
                current_eps = agent.eps_end + (agent.eps_start - agent.eps_end) * \
                    math.exp(-1. * agent.steps_done / agent.eps_decay)
                logging.info(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                           f"Steps: {episode_steps}, Avg100: {avg_reward:.2f}, "
                           f"Eps: {current_eps:.3f}, Stage: {current_stage}")
            
            # Curriculum progression
            if len(reward_history) >= 50 and avg_reward > stage_threshold and current_stage < len(curriculum_stages) - 1:
                current_stage += 1
                stage_threshold += 30.0
                print(f"Advanced to curriculum stage {current_stage}")
            
            # Save best models
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save({
                    'episode': episode,
                    'policy_net_state_dict': agent.policy_net.state_dict(),
                    'target_net_state_dict': agent.target_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'reward': episode_reward,
                    'curriculum_stage': current_stage
                }, 'models/crazyflie_dueling_dqn_best.pth')
            
            if len(reward_history) >= 100 and avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save({
                    'episode': episode,
                    'policy_net_state_dict': agent.policy_net.state_dict(),
                    'target_net_state_dict': agent.target_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'reward': avg_reward,
                    'curriculum_stage': current_stage
                }, 'models/crazyflie_dueling_dqn_best_avg.pth')
            
            # Regular checkpoint
            if episode % 100 == 0:
                torch.save({
                    'episode': episode,
                    'policy_net_state_dict': agent.policy_net.state_dict(),
                    'target_net_state_dict': agent.target_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'reward': episode_reward,
                    'avg_reward': avg_reward,
                    'curriculum_stage': current_stage
                }, f'models/crazyflie_dueling_dqn_checkpoint_{episode}.pth')
            
            episode += 1
        
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
    """Main function"""
    global training_active, viewer_handle

    import argparse
    parser = argparse.ArgumentParser(description='Train Dueling DQN Crazyflie')
    parser.add_argument('--headless', action='store_true', help='Run without viewer')
    args = parser.parse_args()

    try:
        print("Initializing Dueling DQN training...")
        print("Key improvements over vanilla DQN:")
        print("- Separates state value V(s) and action advantage A(s,a)")
        print("- Better learning for states with clear value hierarchy")
        print("- Improved convergence for hovering task")
        print("- Enhanced reward structure for value/advantage separation")

        initialize_simulation()
        print("Simulation initialized!")

        # Start training thread
        training_thread = threading.Thread(target=training_loop, daemon=True)
        training_thread.start()

        if args.headless:
            print("\nTraining started in headless mode. Press Ctrl+C to stop.")
            try:
                training_thread.join()
            except KeyboardInterrupt:
                print("\nStopping training...")
        else:
            print("\nTraining started! Close the viewer to stop.")
            with mujoco.viewer.launch_passive(model, data) as viewer_handle:
                while viewer_handle.is_running() and training_active:
                    with sim_lock:
                        viewer_handle.sync()
                    time.sleep(0.01)

        # Cleanup
        training_active = False
        training_thread.join(timeout=5.0)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()