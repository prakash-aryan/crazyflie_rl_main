#!/usr/bin/env python3
"""
Crazyflie REDQ (Randomized Ensembled Double Q-learning) Training
State-of-the-art continuous control with ensemble Q-networks and high UTD ratio
Optimized for GPU training
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
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import warnings
import math

warnings.filterwarnings('ignore')

# Global simulation lock
sim_lock = threading.RLock()

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
os.makedirs('logs/crazyflie_redq', exist_ok=True)
os.makedirs('models', exist_ok=True)
writer = SummaryWriter('logs/crazyflie_redq')

# Device setup with GPU optimization for RTX 4070
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.empty_cache()  # Clear cache for optimal memory usage
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1e6:.1f}MB allocated")

# Global simulation objects
model = None
data = None
viewer_handle = None

def initialize_simulation():
    """Initialize MuJoCo simulation"""
    global model, data
    with sim_lock:
        try:
            model = mujoco.MjModel.from_xml_path('scene_redq.xml')
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

# REDQ Networks - GPU Optimized
class QNetwork(nn.Module):
    """Single Q-network for ensemble"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        self.net = nn.Sequential(
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
        
        # Initialize weights for better training
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class QEnsemble(nn.Module):
    """Ensemble of Q-networks for REDQ"""
    def __init__(self, state_dim, action_dim, num_q=10, hidden_dim=256):
        super(QEnsemble, self).__init__()
        self.num_q = num_q
        
        # Create ensemble of Q-networks
        self.q_networks = nn.ModuleList([
            QNetwork(state_dim, action_dim, hidden_dim) 
            for _ in range(num_q)
        ])
        
    def forward(self, state, action, return_all=False):
        """Forward pass through all Q-networks"""
        q_values = []
        for q_net in self.q_networks:
            q_values.append(q_net(state, action))
        
        q_values = torch.stack(q_values, dim=0)  # [num_q, batch_size, 1]
        
        if return_all:
            return q_values
        else:
            return q_values.mean(dim=0)  # Return mean
    
    def forward_subset(self, state, action, subset_indices):
        """Forward pass through subset of Q-networks"""
        q_values = []
        for idx in subset_indices:
            q_values.append(self.q_networks[idx](state, action))
        
        return torch.stack(q_values, dim=0)  # [subset_size, batch_size, 1]

class Actor(nn.Module):
    """Stochastic policy network (SAC-style)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        features = self.shared(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        
        # Compute log probability with correction for tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Also return mean for deterministic evaluation
        mean = torch.tanh(mean)
        
        return action, log_prob, mean

# GPU-Optimized Replay Buffer
class REDQReplayBuffer:
    """Replay buffer optimized for GPU usage with pre-allocation"""
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        # Pre-allocate tensors on CPU for memory efficiency
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
        
        # Move to GPU efficiently
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

# REDQ Agent
class REDQAgent:
    def __init__(self, state_dim=18, action_dim=4, hidden_dim=256):
        # REDQ Hyperparameters
        self.num_q = 10          # Number of Q-networks in ensemble
        self.num_min = 2         # Number of Q-networks to sample for min
        self.utd_ratio = 20      # Update-to-data ratio (key REDQ feature)
        self.gamma = 0.99
        self.tau = 0.005
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.q_ensemble = QEnsemble(state_dim, action_dim, self.num_q, hidden_dim).to(device)
        self.q_ensemble_target = QEnsemble(state_dim, action_dim, self.num_q, hidden_dim).to(device)
        
        # Copy parameters to target
        self.q_ensemble_target.load_state_dict(self.q_ensemble.state_dict())
        
        # Optimizers with different learning rates
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q_optimizer = optim.Adam(self.q_ensemble.parameters(), lr=3e-4)
        
        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        # Training stats
        self.updates = 0
        
        # Replay buffer
        self.memory = REDQReplayBuffer(capacity=1000000, state_dim=state_dim, action_dim=action_dim)
        
    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device).unsqueeze(0)
            
            if evaluate:
                _, _, action = self.actor.sample(state)
                action = action.cpu().numpy()[0]
            else:
                action, _, _ = self.actor.sample(state)
                action = action.cpu().numpy()[0]
                
            return action
    
    def update_parameters(self, batch_size=256):
        if len(self.memory) < batch_size:
            return {}
        
        # REDQ: Multiple updates per environment step
        actor_losses = []
        critic_losses = []
        alpha_losses = []
        
        for _ in range(self.utd_ratio):
            # Sample batch
            states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
            
            # Update Q-networks
            with torch.no_grad():
                next_actions, next_log_probs, _ = self.actor.sample(next_states)
                
                # REDQ: Random subset of target Q-networks
                target_indices = random.sample(range(self.num_q), self.num_min)
                target_q_values = self.q_ensemble_target.forward_subset(next_states, next_actions, target_indices)
                target_q = torch.min(target_q_values, dim=0)[0]  # Min over subset
                
                # SAC-style entropy regularization
                target_q = target_q - self.log_alpha.exp() * next_log_probs
                target_value = rewards + (1 - dones) * self.gamma * target_q
            
            # Current Q-values from all networks
            current_q_values = self.q_ensemble(states, actions, return_all=True)  # [num_q, batch_size, 1]
            
            # Q-loss: MSE between each Q-network and target
            q_losses = []
            for i in range(self.num_q):
                q_loss = F.mse_loss(current_q_values[i], target_value)
                q_losses.append(q_loss)
            
            total_q_loss = torch.stack(q_losses).mean()
            
            # Update Q-networks
            self.q_optimizer.zero_grad()
            total_q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_ensemble.parameters(), 1.0)
            self.q_optimizer.step()
            
            critic_losses.append(total_q_loss.item())
            
            # Update actor (less frequently to save computation)
            if self.updates % 2 == 0:
                # Sample new actions
                new_actions, log_probs, _ = self.actor.sample(states)
                
                # REDQ: Random subset for actor update
                q_indices = random.sample(range(self.num_q), self.num_min)
                q_values = self.q_ensemble.forward_subset(states, new_actions, q_indices)
                q_new = torch.min(q_values, dim=0)[0]  # Min over subset
                
                actor_loss = (self.log_alpha.exp() * log_probs - q_new).mean()
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_optimizer.step()
                
                actor_losses.append(actor_loss.item())
                
                # Update alpha (temperature)
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                alpha_losses.append(alpha_loss.item())
            
            # Soft update target networks (less frequently)
            if self.updates % 2 == 0:
                self.soft_update(self.q_ensemble, self.q_ensemble_target)
        
        self.updates += 1
        
        # Logging
        if self.updates % 100 == 0:
            if actor_losses:
                writer.add_scalar('Loss/Actor', np.mean(actor_losses), self.updates)
            writer.add_scalar('Loss/Critic', np.mean(critic_losses), self.updates)
            if alpha_losses:
                writer.add_scalar('Loss/Alpha', np.mean(alpha_losses), self.updates)
            writer.add_scalar('Values/Alpha', self.log_alpha.exp().item(), self.updates)
            writer.add_scalar('Values/Q', current_q_values.mean().item(), self.updates)
            
            # GPU memory monitoring
            if device.type == 'cuda':
                writer.add_scalar('GPU/Memory_MB', torch.cuda.memory_allocated()/1e6, self.updates)
        
        return {
            'actor_loss': np.mean(actor_losses) if actor_losses else 0.0,
            'critic_loss': np.mean(critic_losses),
            'alpha_loss': np.mean(alpha_losses) if alpha_losses else 0.0,
            'alpha': self.log_alpha.exp().item(),
            'q_value': current_q_values.mean().item()
        }
    
    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Training flag
training_active = True

def training_loop():
    """Main REDQ training loop optimized for GPU"""
    global training_active
    
    agent = REDQAgent()
    
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
    start_timesteps = 5000   # Reduced due to high UTD ratio
    batch_size = 256
    
    print(f"Starting REDQ training with {num_episodes} episodes")
    print(f"Ensemble size: {agent.num_q}, UTD ratio: {agent.utd_ratio}")
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
                    # Policy action
                    action = agent.select_action(state)
                
                # Execute action
                apply_continuous_action(action)
                
                # Get next state and reward
                next_state = get_observation()
                reward = calculate_reward()
                done = is_done()
                
                # Store transition
                agent.memory.push(state, action, reward, next_state, done)
                
                # Update agent (REDQ performs many updates per step)
                if total_steps >= start_timesteps:
                    update_info = agent.update_parameters(batch_size)
                    
                    # Log training info periodically
                    if total_steps % 1000 == 0 and update_info:
                        for key, value in update_info.items():
                            writer.add_scalar(f'Training/{key}', value, total_steps)
                
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
            writer.add_scalar('Episode/Alpha', agent.log_alpha.exp().item(), episode)
            
            if episode % 10 == 0:
                logging.info(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                           f"Steps: {episode_steps}, Avg100: {avg_reward:.2f}, "
                           f"Total: {total_steps}, Difficulty: {difficulty:.2f}, "
                           f"Alpha: {agent.log_alpha.exp().item():.4f}")
                if episode % 100 == 0:
                    gpu_memory = torch.cuda.memory_allocated() / 1e6
                    logging.info(f"GPU Memory: {gpu_memory:.1f}MB allocated")
            
            # Curriculum progression
            if len(reward_history) >= 50 and avg_reward > difficulty_threshold and difficulty < 1.0:
                difficulty = min(1.0, difficulty + difficulty_increment)
                print(f"Increasing difficulty to {difficulty:.2f}")
            
            # Save models
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save({
                    'episode': episode,
                    'actor_state_dict': agent.actor.state_dict(),
                    'q_ensemble_state_dict': agent.q_ensemble.state_dict(),
                    'q_ensemble_target_state_dict': agent.q_ensemble_target.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    'q_optimizer_state_dict': agent.q_optimizer.state_dict(),
                    'log_alpha': agent.log_alpha,
                    'reward': episode_reward,
                    'difficulty': difficulty,
                    'total_steps': total_steps,
                    'num_q': agent.num_q,
                    'utd_ratio': agent.utd_ratio
                }, 'models/crazyflie_redq_best.pth')
            
            # Regular checkpoint
            if episode % 100 == 0:
                torch.save({
                    'episode': episode,
                    'actor_state_dict': agent.actor.state_dict(),
                    'q_ensemble_state_dict': agent.q_ensemble.state_dict(),
                    'q_ensemble_target_state_dict': agent.q_ensemble_target.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    'q_optimizer_state_dict': agent.q_optimizer.state_dict(),
                    'log_alpha': agent.log_alpha,
                    'reward': episode_reward,
                    'avg_reward': avg_reward,
                    'difficulty': difficulty,
                    'total_steps': total_steps,
                    'num_q': agent.num_q,
                    'utd_ratio': agent.utd_ratio
                }, f'models/crazyflie_redq_checkpoint_{episode}.pth')
                
                # Periodic GPU memory cleanup
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
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
    print(f"Final GPU Memory: {torch.cuda.memory_allocated()/1e6:.1f}MB allocated")
    writer.close()

def main():
    """Main function"""
    global training_active, viewer_handle
    
    try:
        print("Initializing REDQ training...")
        print("Key features:")
        print("- Randomized Ensembled Double Q-learning")
        print("- Ensemble of 10 Q-networks with random subset sampling")
        print("- High Update-to-Data ratio (20:1) for sample efficiency")
        print("- GPU optimized for RTX 4070")
        print("- Reduced overestimation bias")
        print("- Conservative action space for drone stability")
        
        initialize_simulation()
        print("Simulation initialized!")
        
        # Start training thread
        training_thread = threading.Thread(target=training_loop, daemon=True)
        training_thread.start()
        
        print("\nTraining started! Close the viewer to stop.")
        
        # Launch viewer
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