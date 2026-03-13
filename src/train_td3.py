#!/usr/bin/env python3
"""
Crazyflie TD3 (Twin Delayed Deep Deterministic Policy Gradient) Training
Optimized for GPU and sample efficiency
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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import warnings

warnings.filterwarnings('ignore')

# Global simulation lock
sim_lock = threading.RLock()

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
os.makedirs('logs/crazyflie_td3', exist_ok=True)
os.makedirs('models', exist_ok=True)
writer = SummaryWriter('logs/crazyflie_td3')

# Device setup with GPU optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Global simulation objects
model = None
data = None
viewer_handle = None

def initialize_simulation():
    """Initialize MuJoCo simulation"""
    global model, data
    with sim_lock:
        try:
            model = mujoco.MjModel.from_xml_path('scene_td3.xml')
        except:
            model = mujoco.MjModel.from_xml_path('scene.xml')
        
        data = mujoco.MjData(model)
        
        # Fine-tuned physics for stability
        model.opt.timestep = 0.005
        model.opt.iterations = 50
        model.opt.tolerance = 1e-4
        
        reset_simulation()

def reset_simulation(randomize=True, difficulty=0.5):
    """Reset with progressive difficulty"""
    global model, data
    with sim_lock:
        try:
            mujoco.mj_resetData(model, data)
            
            if randomize:
                # Structured randomization based on difficulty
                pos_range = 0.02 + 0.18 * difficulty
                vel_range = 0.02 + 0.08 * difficulty
                angle_range = 0.01 + 0.09 * difficulty
                
                # Position with bias towards center
                data.qpos[0] = np.random.normal(0, pos_range/2)
                data.qpos[1] = np.random.normal(0, pos_range/2)
                data.qpos[2] = 1.0 + np.random.uniform(-0.05, 0.05)
                
                # Orientation with small perturbations
                angle = np.random.uniform(-angle_range, angle_range)
                axis = np.random.randn(3)
                axis = axis / (np.linalg.norm(axis) + 1e-8)
                
                data.qpos[3] = np.cos(angle/2)
                data.qpos[4] = np.sin(angle/2) * axis[0]
                data.qpos[5] = np.sin(angle/2) * axis[1]
                data.qpos[6] = np.sin(angle/2) * axis[2]
                
                # Small velocities
                data.qvel[:3] = np.random.normal(0, vel_range/3, size=3)
                data.qvel[3:6] = np.random.normal(0, vel_range/5, size=3)
            else:
                data.qpos[:3] = [0.0, 0.0, 1.0]
                data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                data.qvel[:] = 0.0
            
            # Initialize at hover thrust
            data.ctrl[0] = 0.26
            data.ctrl[1:] = 0.0
            
            mujoco.mj_forward(model, data)
            
        except Exception as e:
            print(f"Error in reset: {e}")

# TD3 Networks
class Actor(nn.Module):
    """Deterministic policy network for TD3"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    """Twin Q-network for TD3"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2
    
    def q1_forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)

# Optimized Replay Buffer
class ReplayBuffer:
    """Efficient replay buffer optimized for GPU usage"""
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        # Pre-allocate tensors on CPU for efficiency
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)
        
    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = torch.from_numpy(state).float()
        self.actions[self.position] = torch.from_numpy(action).float()
        self.rewards[self.position] = reward
        self.next_states[self.position] = torch.from_numpy(next_state).float()
        self.dones[self.position] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,))
        
        # Move to GPU in batches for efficiency
        states = self.states[indices].to(device, non_blocking=True)
        actions = self.actions[indices].to(device, non_blocking=True)
        rewards = self.rewards[indices].to(device, non_blocking=True)
        next_states = self.next_states[indices].to(device, non_blocking=True)
        dones = self.dones[indices].to(device, non_blocking=True)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return self.size

def get_observation():
    """Enhanced observation with more features"""
    global model, data
    with sim_lock:
        try:
            pos = data.qpos[0:3].copy()
            vel = data.qvel[0:3].copy()
            
            # Quaternion and derived features
            quat = data.qpos[3:7].copy()
            quat = quat / (np.linalg.norm(quat) + 1e-8)
            
            # Rotation matrix elements
            w, x, y, z = quat
            roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
            pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
            yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            
            # Angular velocity
            angvel = data.qvel[3:6].copy()
            
            # Target information
            target = np.array([0.0, 0.0, 1.0])
            target_error = target - pos
            target_dist = np.linalg.norm(target_error)
            
            # Previous controls for smoother transitions
            controls = data.ctrl.copy()
            
            # Observation vector (18 dimensions)
            obs = np.concatenate([
                target_error / 2.0,           # 3 - normalized target error
                vel / 2.0,                    # 3 - normalized velocity
                np.array([roll, pitch, yaw]), # 3 - euler angles
                angvel / 5.0,                 # 3 - normalized angular velocity
                controls[:1] / 0.35,          # 1 - normalized thrust
                controls[1:] / 0.1,           # 3 - normalized moments
                [target_dist / 2.0],          # 1 - normalized distance
                [np.tanh(pos[2] - 1.0)]       # 1 - height error feature
            ])
            
            return obs.astype(np.float32)
            
        except Exception as e:
            print(f"Error getting observation: {e}")
            return np.zeros(18, dtype=np.float32)

def calculate_reward():
    """Dense reward function with multiple components"""
    global data
    
    pos = data.qpos[0:3]
    vel = data.qvel[0:3]
    angvel = data.qvel[3:6]
    quat = data.qpos[3:7]
    
    target = np.array([0.0, 0.0, 1.0])
    
    # Distance components
    position_error = np.linalg.norm(pos - target)
    height_error = abs(pos[2] - 1.0)
    xy_error = np.linalg.norm(pos[0:2])
    
    # Velocity components
    linear_velocity = np.linalg.norm(vel)
    vertical_velocity = abs(vel[2])
    angular_velocity = np.linalg.norm(angvel)
    
    # Orientation stability
    w, x, y, z = quat
    orientation_error = np.sqrt(x**2 + y**2) * 2.0
    
    # Dense reward with multiple components
    reward = 0.0
    
    # Alive bonus
    reward += 3.0
    
    # Position reward (continuous, not discrete)
    position_reward = np.exp(-2.0 * position_error) * 20.0
    reward += position_reward
    
    # Height reward (most important)
    height_reward = np.exp(-5.0 * height_error) * 15.0
    reward += height_reward
    
    # Velocity penalty (encourage stillness)
    velocity_penalty = -linear_velocity * 2.0
    reward += velocity_penalty
    
    # Angular velocity penalty
    angular_penalty = -angular_velocity * 1.5
    reward += angular_penalty
    
    # Orientation penalty
    orientation_penalty = -orientation_error * 3.0
    reward += orientation_penalty
    
    # Stability bonus (near target and slow)
    if position_error < 0.1 and linear_velocity < 0.1:
        reward += 25.0
    elif position_error < 0.2 and linear_velocity < 0.2:
        reward += 10.0
    
    # Control effort penalty (for smooth control)
    thrust_effort = abs(data.ctrl[0] - 0.26) * 2.0
    moment_effort = np.linalg.norm(data.ctrl[1:]) * 5.0
    reward -= thrust_effort + moment_effort
    
    # Terminal penalties
    if pos[2] < 0.3 or pos[2] > 2.0:
        reward = -50.0
    elif xy_error > 1.0:
        reward = -30.0
    
    return reward

def is_done():
    """Check termination with safety margins"""
    global data
    pos = data.qpos[0:3]
    
    # Tighter bounds to encourage staying centered
    if pos[2] < 0.3 or pos[2] > 2.0:
        return True
    if abs(pos[0]) > 1.0 or abs(pos[1]) > 1.0:
        return True
    
    # Check for instability
    vel = data.qvel[0:3]
    if np.linalg.norm(vel) > 5.0:  # High velocity
        return True
    
    return False

def apply_continuous_action(action):
    """Apply continuous actions with safety limits"""
    global model, data
    
    with sim_lock:
        try:
            # Action is normalized [-1, 1] from tanh
            # Conservative action mapping
            
            # Thrust: map [-1, 1] to [0.24, 0.28] (tighter range)
            data.ctrl[0] = 0.26 + action[0] * 0.02
            
            # Moments: map [-1, 1] to [-0.02, 0.02] (very small)
            data.ctrl[1] = action[1] * 0.02
            data.ctrl[2] = action[2] * 0.02
            data.ctrl[3] = action[3] * 0.01  # Even smaller yaw
            
            # Additional safety: decay moments
            data.ctrl[1:] *= 0.98
            
            # Clip to safe ranges
            data.ctrl[0] = np.clip(data.ctrl[0], 0.20, 0.32)
            data.ctrl[1:] = np.clip(data.ctrl[1:], -0.05, 0.05)
            
            # Step simulation with smaller timestep
            for _ in range(10):  # 10 steps = 0.05s
                mujoco.mj_step(model, data)
                
        except Exception as e:
            print(f"Error in apply_action: {e}")
            reset_simulation()

# TD3 Agent
class TD3Agent:
    def __init__(self, state_dim=18, action_dim=4, hidden_dim=256):
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        
        # Target networks
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        
        # Copy parameters to targets
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers with learning rate scheduling
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=1000000, state_dim=state_dim, action_dim=action_dim)
        
        # Hyperparameters
        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        
        # Training counters
        self.total_it = 0
    
    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = self.actor(state_tensor).cpu().numpy()[0]
        return action
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return {'critic_loss': 0, 'actor_loss': 0, 'q_value': 0}
        
        self.total_it += 1
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Target policy smoothing
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        actor_loss = torch.tensor(0.0)
        
        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Actor loss
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # Soft update target networks
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': current_q1.mean().item()
        }
    
    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Training flag
training_active = True

def training_loop():
    """Main TD3 training loop with GPU optimization"""
    global training_active
    
    agent = TD3Agent()
    
    num_episodes = 5000
    max_steps_per_episode = 1000
    
    # Training metrics
    episode = 0
    total_steps = 0
    best_reward = -float('inf')
    reward_history = deque(maxlen=100)
    
    # Curriculum learning
    difficulty = 0.0
    difficulty_increment = 0.1
    difficulty_threshold = 300.0
    
    # Training parameters
    start_timesteps = 10000  # Random actions before training
    expl_noise = 0.1  # Exploration noise
    batch_size = 256
    eval_freq = 25
    
    print(f"Starting TD3 training with {num_episodes} episodes")
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e6:.1f}MB allocated")
    
    while episode < num_episodes and training_active:
        try:
            # Reset environment
            reset_simulation(randomize=True, difficulty=difficulty)
            
            # Initial observation
            state = get_observation()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps_per_episode):
                if not training_active:
                    break
                
                # Select action
                if total_steps < start_timesteps:
                    # Random exploration
                    action = np.random.uniform(-1, 1, size=4)
                else:
                    # Policy with exploration noise
                    action = agent.select_action(state)
                    if expl_noise != 0:
                        action = action + np.random.normal(0, expl_noise, size=action.shape)
                        action = np.clip(action, -1, 1)
                
                # Execute action
                apply_continuous_action(action)
                
                # Get next state and reward
                next_state = get_observation()
                reward = calculate_reward()
                done = is_done()
                
                # Store transition
                agent.memory.push(state, action, reward, next_state, done)
                
                # Update agent
                if total_steps >= start_timesteps:
                    update_info = agent.update()
                    
                    # Log training info
                    if total_steps % 1000 == 0:
                        writer.add_scalar('Loss/Critic', update_info['critic_loss'], total_steps)
                        writer.add_scalar('Loss/Actor', update_info['actor_loss'], total_steps)
                        writer.add_scalar('Values/Q', update_info['q_value'], total_steps)
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                if done:
                    break
            
            # Episode finished
            episode += 1
            reward_history.append(episode_reward)
            avg_reward = sum(reward_history) / len(reward_history)
            
            # Log episode info
            writer.add_scalar('Episode/Reward', episode_reward, episode)
            writer.add_scalar('Episode/Steps', episode_steps, episode)
            writer.add_scalar('Episode/AvgReward', avg_reward, episode)
            writer.add_scalar('Episode/Difficulty', difficulty, episode)
            writer.add_scalar('Episode/ExplNoise', expl_noise, episode)
            
            if episode % 10 == 0:
                gpu_memory = torch.cuda.memory_allocated() / 1e6
                logging.info(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                           f"Avg100: {avg_reward:.2f}, Steps: {episode_steps}, "
                           f"Total: {total_steps}, Difficulty: {difficulty:.2f}, "
                           f"Noise: {expl_noise:.3f}")
                if episode % 100 == 0:
                    logging.info(f"GPU Memory: {gpu_memory:.1f}MB allocated")
            
            # Curriculum progression
            if len(reward_history) >= 50 and avg_reward > difficulty_threshold and difficulty < 1.0:
                difficulty = min(1.0, difficulty + difficulty_increment)
                print(f"Increasing difficulty to {difficulty:.2f}")
            
            # Decay exploration noise
            if episode > 1000:
                expl_noise = max(0.05, expl_noise * 0.9998)
            
            # Save models
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save({
                    'episode': episode,
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'actor_target_state_dict': agent.actor_target.state_dict(),
                    'critic_target_state_dict': agent.critic_target.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                    'reward': episode_reward,
                    'difficulty': difficulty,
                    'total_steps': total_steps
                }, 'models/crazyflie_td3_best.pth')
            
            # Regular checkpoint
            if episode % 100 == 0:
                torch.save({
                    'episode': episode,
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'actor_target_state_dict': agent.actor_target.state_dict(),
                    'critic_target_state_dict': agent.critic_target.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                    'reward': episode_reward,
                    'avg_reward': avg_reward,
                    'difficulty': difficulty,
                    'total_steps': total_steps
                }, f'models/crazyflie_td3_checkpoint_{episode}.pth')
                
        except Exception as e:
            print(f"Error in training: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1.0)
            continue
    
    training_active = False
    print("Training completed!")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Total timesteps: {total_steps}")
    writer.close()

def main():
    """Main function"""
    global training_active, viewer_handle

    import argparse
    parser = argparse.ArgumentParser(description='Train TD3 Crazyflie')
    parser.add_argument('--headless', action='store_true', help='Run without viewer')
    args = parser.parse_args()

    try:
        print("Initializing TD3 training...")
        print("Key features:")
        print("- Twin Delayed Deep Deterministic Policy Gradient")
        print("- Target policy smoothing for robustness")
        print("- Delayed policy updates to reduce overestimation")
        print("- Optimized for GPU training")
        print("- Curriculum learning with progressive difficulty")
        print("- Conservative action space for drone stability")

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