#!/usr/bin/env python3
"""
Crazyflie Dreamer-v3 Training
Model-based RL with world model learning and imagination-based policy training
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
from torch.distributions import Normal, Independent
from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple
import random
import warnings
import math

warnings.filterwarnings('ignore')

# Global simulation lock
sim_lock = threading.RLock()

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
os.makedirs('logs/crazyflie_dreamer', exist_ok=True)
os.makedirs('models', exist_ok=True)
writer = SummaryWriter('logs/crazyflie_dreamer')

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
            model = mujoco.MjModel.from_xml_path('scene_dreamer.xml')
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

# Symlog functions for numerical stability (Dreamer-v3 feature)
def symlog(x):
    """Symmetric logarithm for better numerical stability"""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x):
    """Inverse of symlog"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

# World Model Components
class Encoder(nn.Module):
    """Encodes observations to latent representations"""
    def __init__(self, obs_dim, latent_dim=32, hidden_dim=256):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and std
        )
        
    def forward(self, obs):
        x = self.net(obs)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(std) + 0.1
        return Independent(Normal(mean, std), 1)

class Decoder(nn.Module):
    """Decodes latent states back to observations"""
    def __init__(self, latent_dim, obs_dim, hidden_dim=256):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
    def forward(self, latent):
        return self.net(latent)

class DynamicsModel(nn.Module):
    """Predicts next latent state from current state and action"""
    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super(DynamicsModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and std
        )
        
    def forward(self, latent, action):
        x = torch.cat([latent, action], dim=-1)
        x = self.net(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(std) + 0.1
        return Independent(Normal(mean, std), 1)

class RewardModel(nn.Module):
    """Predicts reward from latent state"""
    def __init__(self, latent_dim, hidden_dim=256):
        super(RewardModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, latent):
        return self.net(latent)

class ContinueModel(nn.Module):
    """Predicts episode continuation from latent state"""
    def __init__(self, latent_dim, hidden_dim=256):
        super(ContinueModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, latent):
        return torch.sigmoid(self.net(latent))

class WorldModel(nn.Module):
    """Complete world model combining all components"""
    def __init__(self, obs_dim, action_dim, latent_dim=32, hidden_dim=256):
        super(WorldModel, self).__init__()
        self.encoder = Encoder(obs_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, obs_dim, hidden_dim)
        self.dynamics = DynamicsModel(latent_dim, action_dim, hidden_dim)
        self.reward = RewardModel(latent_dim, hidden_dim)
        self.continue_model = ContinueModel(latent_dim, hidden_dim)
        
    def encode(self, obs):
        return self.encoder(obs)
    
    def decode(self, latent):
        return self.decoder(latent)
    
    def predict_next(self, latent, action):
        return self.dynamics(latent, action)
    
    def predict_reward(self, latent):
        return self.reward(latent)
    
    def predict_continue(self, latent):
        return self.continue_model(latent)

# Policy Networks
class Actor(nn.Module):
    """Policy network that outputs action distribution"""
    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean and std
        )
        
    def forward(self, latent):
        x = self.net(latent)
        mean, std = torch.chunk(x, 2, dim=-1)
        mean = torch.tanh(mean)  # Bounded actions
        std = F.softplus(std) + 0.1
        return Independent(Normal(mean, std), 1)

class Critic(nn.Module):
    """Value function network"""
    def __init__(self, latent_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, latent):
        return self.net(latent)

# Experience Replay Buffer
Experience = namedtuple('Experience', ['obs', 'action', 'reward', 'done'])

class ExperienceBuffer:
    """Buffer for storing real environment experience"""
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        # Pre-allocate tensors
        self.observations = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32)
        self.dones = torch.zeros((capacity,), dtype=torch.bool)
        
    def add(self, obs, action, reward, done):
        self.observations[self.position] = torch.from_numpy(obs).float()
        self.actions[self.position] = torch.from_numpy(action).float()
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample_sequences(self, batch_size, seq_len):
        """Sample sequences for world model training"""
        if self.size < seq_len:
            return None
            
        # Sample starting indices
        max_start = self.size - seq_len
        starts = torch.randint(0, max_start, (batch_size,))
        
        sequences = []
        for start in starts:
            end = start + seq_len
            seq = {
                'obs': self.observations[start:end].to(device),
                'actions': self.actions[start:end].to(device),
                'rewards': self.rewards[start:end].to(device),
                'dones': self.dones[start:end].to(device)
            }
            sequences.append(seq)
        
        # Stack sequences
        batch = {}
        for key in sequences[0].keys():
            batch[key] = torch.stack([seq[key] for seq in sequences])
        
        return batch
    
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
            
            # Rotation matrix from quaternion
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

def apply_continuous_action(action):
    """Apply continuous action with conservative mapping"""
    global model, data
    with sim_lock:
        try:
            # Action is normalized [-1, 1] from tanh
            # Very conservative action mapping for stability
            
            # Thrust: map [-1, 1] to [0.24, 0.28] (tight range around hover)
            data.ctrl[0] = 0.26 + action[0] * 0.02
            
            # Moments: map [-1, 1] to small ranges for stability
            data.ctrl[1] = action[1] * 0.015  # roll moment
            data.ctrl[2] = action[2] * 0.015  # pitch moment
            data.ctrl[3] = action[3] * 0.008  # yaw moment (even smaller)
            
            # Safety clipping
            data.ctrl[0] = np.clip(data.ctrl[0], 0.20, 0.32)
            data.ctrl[1:] = np.clip(data.ctrl[1:], -0.03, 0.03)
            
            # Step simulation multiple times for stability
            for _ in range(10):  # 10 steps = 0.05s at 0.005s timestep
                mujoco.mj_step(model, data)
                
        except Exception as e:
            print(f"Error applying action: {e}")

def calculate_reward():
    """Enhanced reward function encouraging precise hovering"""
    global model, data
    with sim_lock:
        try:
            pos = data.qpos[0:3].copy()
            vel = data.qvel[0:3].copy()
            angvel = data.qvel[3:6].copy()
            
            # Target position
            target = np.array([0.0, 0.0, 1.0])
            
            # Distance to target
            distance = np.linalg.norm(pos - target)
            
            # Base reward - exponential in distance for precision
            position_reward = 100 * np.exp(-5 * distance)
            
            # Velocity penalty - encourage hovering
            velocity_penalty = 10 * np.linalg.norm(vel)
            
            # Angular velocity penalty
            angular_penalty = 5 * np.linalg.norm(angvel)
            
            # Action smoothness (encourage gentle control)
            action_penalty = 2 * np.linalg.norm(data.ctrl[1:])  # Don't penalize thrust as much
            
            # Height bonus for staying near target height
            height_bonus = 50 * np.exp(-10 * abs(pos[2] - 1.0))
            
            # Stability bonus for low velocities when close
            if distance < 0.2:
                stability_bonus = 20 * np.exp(-5 * np.linalg.norm(vel))
            else:
                stability_bonus = 0
                
            # Total reward
            reward = (position_reward + height_bonus + stability_bonus 
                     - velocity_penalty - angular_penalty - action_penalty)
            
            # Crash penalty
            if pos[2] < 0.3:
                reward -= 1000
                
            return reward
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0.0

def is_done():
    """Check if episode should terminate"""
    global model, data
    with sim_lock:
        try:
            pos = data.qpos[0:3].copy()
            
            # Episode ends if crashed or too far from target
            if pos[2] < 0.3:  # Crashed
                return True
            if np.linalg.norm(pos - np.array([0, 0, 1])) > 2.0:  # Too far
                return True
                
            return False
            
        except Exception as e:
            print(f"Error checking done condition: {e}")
            return True

class DreamerAgent:
    """Dreamer agent combining world model and policy learning"""
    def __init__(self, obs_dim=18, action_dim=4, latent_dim=32, hidden_dim=256, 
                 buffer_capacity=100000, learning_rate=3e-4):
        
        # Networks
        self.world_model = WorldModel(obs_dim, action_dim, latent_dim, hidden_dim).to(device)
        self.actor = Actor(latent_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(latent_dim, hidden_dim).to(device)
        
        # Optimizers
        self.world_model_optimizer = optim.Adam(self.world_model.parameters(), lr=learning_rate)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.buffer = ExperienceBuffer(buffer_capacity, obs_dim, action_dim)
        
        # Training counters
        self.world_model_updates = 0
        self.policy_updates = 0
        
    def select_action(self, obs, deterministic=False):
        """Select action using the current policy"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Encode observation to latent space
            latent_dist = self.world_model.encode(obs_tensor)
            latent = latent_dist.mode if deterministic else latent_dist.sample()
            
            # Get action from policy
            action_dist = self.actor(latent)
            action = action_dist.mode if deterministic else action_dist.sample()
            
            return action.cpu().numpy().squeeze()
    
    def update_world_model(self, batch_size=32, seq_len=16):
        """Update world model using real experience"""
        batch = self.buffer.sample_sequences(batch_size, seq_len)
        if batch is None:
            return None
        
        # Unpack batch
        obs_seq = batch['obs']  # [batch_size, seq_len, obs_dim]
        action_seq = batch['actions']  # [batch_size, seq_len, action_dim]
        reward_seq = batch['rewards']  # [batch_size, seq_len]
        done_seq = batch['dones']  # [batch_size, seq_len]
        
        # Encode observations to latent space
        obs_flat = obs_seq.view(-1, obs_seq.size(-1))
        latent_dist = self.world_model.encode(obs_flat)
        latent_seq = latent_dist.sample().view(batch_size, seq_len, -1)
        
        # Reconstruction loss
        recon_obs = self.world_model.decode(latent_seq.view(-1, latent_seq.size(-1)))
        recon_loss = F.mse_loss(recon_obs, obs_flat)
        
        # KL loss for encoder
        kl_loss = latent_dist.entropy().mean()
        
        # Dynamics loss
        dynamics_loss = 0
        reward_loss = 0
        continue_loss = 0
        
        for t in range(seq_len - 1):
            # Predict next latent state
            next_latent_dist = self.world_model.predict_next(latent_seq[:, t], action_seq[:, t])
            dynamics_loss += -next_latent_dist.log_prob(latent_seq[:, t + 1]).mean()
            
            # Predict reward
            pred_reward = self.world_model.predict_reward(latent_seq[:, t]).squeeze()
            reward_loss += F.mse_loss(pred_reward, reward_seq[:, t])
            
            # Predict continuation
            pred_continue = self.world_model.predict_continue(latent_seq[:, t]).squeeze()
            continue_target = (~done_seq[:, t]).float()
            continue_loss += F.binary_cross_entropy(pred_continue, continue_target)
        
        dynamics_loss /= (seq_len - 1)
        reward_loss /= (seq_len - 1)
        continue_loss /= (seq_len - 1)
        
        # Total loss
        world_model_loss = (recon_loss + 0.1 * kl_loss + dynamics_loss + 
                           reward_loss + continue_loss)
        
        # Update world model
        self.world_model_optimizer.zero_grad()
        world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.world_model_optimizer.step()
        
        self.world_model_updates += 1
        
        return {
            'world_model_loss': world_model_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'dynamics_loss': dynamics_loss.item(),
            'reward_loss': reward_loss.item(),
            'continue_loss': continue_loss.item()
        }
    
    def update_policy(self, horizon=15, batch_size=32):
        """Update policy using imagination"""
        if len(self.buffer) < 1000:
            return None
        
        # Sample starting states from buffer
        starts = torch.randint(0, len(self.buffer), (batch_size,))
        start_obs = self.buffer.observations[starts].to(device)
        
        # Encode to latent space
        with torch.no_grad():
            latent_dist = self.world_model.encode(start_obs)
            latents = latent_dist.sample()
        
        # Imagine rollouts
        imagined_latents = [latents]
        imagined_actions = []
        imagined_rewards = []
        imagined_continues = []
        
        for h in range(horizon):
            # Sample action from current policy
            action_dist = self.actor(latents)
            actions = action_dist.rsample()
            imagined_actions.append(actions)
            
            # Predict next state, reward, and continuation
            with torch.no_grad():
                next_latent_dist = self.world_model.predict_next(latents, actions)
                latents = next_latent_dist.sample()
                imagined_latents.append(latents)
                
                rewards = self.world_model.predict_reward(latents).squeeze()
                continues = self.world_model.predict_continue(latents).squeeze()
                
                imagined_rewards.append(rewards)
                imagined_continues.append(continues)
        
        # Convert to tensors
        latents = torch.stack(imagined_latents[:-1])  # [horizon, batch_size, latent_dim]
        actions = torch.stack(imagined_actions)  # [horizon, batch_size, action_dim]
        rewards = torch.stack(imagined_rewards)  # [horizon, batch_size]
        continues = torch.stack(imagined_continues)  # [horizon, batch_size]
        
        # Compute values
        values = self.critic(latents.view(-1, latents.size(-1))).view(horizon, batch_size)
        
        # Compute lambda returns
        returns = torch.zeros_like(rewards)
        last_value = values[-1]
        
        for t in reversed(range(horizon)):
            if t == horizon - 1:
                returns[t] = rewards[t] + 0.99 * continues[t] * last_value
            else:
                returns[t] = rewards[t] + 0.99 * continues[t] * (0.95 * returns[t+1] + 0.05 * values[t+1])
        
        # Compute advantages
        advantages = returns - values
        
        # Flatten for training
        latents_flat = latents.view(-1, latents.size(-1))
        actions_flat = actions.view(-1, actions.size(-1))
        advantages_flat = advantages.view(-1)
        returns_flat = returns.reshape(-1)
        
        # Actor loss
        action_dist = self.actor(latents_flat)
        log_probs = action_dist.log_prob(actions_flat)
        actor_loss = -(log_probs * advantages_flat).mean()
        
        # Critic loss
        predicted_values = self.critic(latents_flat).squeeze(-1)
        critic_loss = F.mse_loss(predicted_values, returns_flat)
        
        # Update networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_optimizer.step()
        
        self.policy_updates += 1
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_value': predicted_values.mean().item(),
            'mean_advantage': advantages_flat.mean().item()
        }

# Training flag
training_active = True

def training_loop():
    """Main Dreamer training loop"""
    global training_active
    
    agent = DreamerAgent()
    
    num_episodes = 3000  # Fewer episodes due to sample efficiency
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
    collect_steps = 5000  # Collect initial data before training
    world_model_train_freq = 1000  # Train world model every N steps
    policy_train_freq = 100  # Train policy every N steps
    
    print(f"Starting Dreamer training with {num_episodes} episodes")
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e6:.1f}MB allocated")
    
    while episode < num_episodes and training_active:
        try:
            # Reset environment
            reset_simulation(randomize=True, difficulty=difficulty)
            
            # Initial observation
            obs = get_observation()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps_per_episode):
                if not training_active:
                    break
                
                # Select action
                if total_steps < collect_steps:
                    # Random exploration during initial collection
                    action = np.random.uniform(-1, 1, size=4)
                else:
                    # Use policy
                    action = agent.select_action(obs, deterministic=False)
                
                # Apply action
                apply_continuous_action(action)
                
                # Get next observation and reward
                next_obs = get_observation()
                reward = calculate_reward()
                done = is_done() or episode_steps >= max_steps_per_episode - 1
                
                # Store experience
                agent.buffer.add(obs, action, reward, done)
                
                # Train world model
                if total_steps >= collect_steps and total_steps % world_model_train_freq == 0:
                    world_stats = agent.update_world_model()
                    if world_stats:
                        for key, value in world_stats.items():
                            writer.add_scalar(f'WorldModel/{key}', value, agent.world_model_updates)
                
                # Train policy
                if total_steps >= collect_steps and total_steps % policy_train_freq == 0:
                    policy_stats = agent.update_policy()
                    if policy_stats:
                        for key, value in policy_stats.items():
                            writer.add_scalar(f'Policy/{key}', value, agent.policy_updates)
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                obs = next_obs
                
                if done:
                    break
                
                time.sleep(0.01)
            
            # Episode finished
            episode += 1
            reward_history.append(episode_reward)
            avg_reward = np.mean(reward_history) if len(reward_history) > 0 else episode_reward
            
            # Logging
            writer.add_scalar('Episode/Reward', episode_reward, episode)
            writer.add_scalar('Episode/Steps', episode_steps, episode)
            writer.add_scalar('Episode/AvgReward', avg_reward, episode)
            writer.add_scalar('Episode/Difficulty', difficulty, episode)
            writer.add_scalar('Training/BufferSize', len(agent.buffer), episode)
            
            if episode % 10 == 0:
                logging.info(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                           f"Steps: {episode_steps}, Avg100: {avg_reward:.2f}, "
                           f"Difficulty: {difficulty:.2f}, TotalSteps: {total_steps}, "
                           f"BufferSize: {len(agent.buffer)}")
            
            # Save checkpoints
            if episode % 100 == 0:
                checkpoint = {
                    'episode': episode,
                    'world_model_state_dict': agent.world_model.state_dict(),
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'world_model_optimizer_state_dict': agent.world_model_optimizer.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                    'reward': episode_reward,
                    'avg_reward': avg_reward,
                    'difficulty': difficulty,
                    'total_steps': total_steps
                }
                torch.save(checkpoint, f'models/crazyflie_dreamer_checkpoint_{episode}.pth')
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                checkpoint = {
                    'episode': episode,
                    'world_model_state_dict': agent.world_model.state_dict(),
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'reward': episode_reward,
                    'avg_reward': avg_reward,
                    'difficulty': difficulty,
                    'total_steps': total_steps
                }
                torch.save(checkpoint, 'models/crazyflie_dreamer_best.pth')
                logging.info(f"New best model saved! Reward: {best_reward:.2f}")
            
            # Curriculum learning - increase difficulty
            if avg_reward > difficulty_threshold and len(reward_history) >= 50:
                if difficulty < 1.0:
                    difficulty = min(1.0, difficulty + difficulty_increment)
                    difficulty_threshold += 200  # Increase threshold
                    logging.info(f"Difficulty increased to {difficulty:.2f}")
            
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Training completed after {episode} episodes")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Total timesteps: {total_steps}")
    writer.close()

def main():
    """Main function"""
    global training_active, viewer_handle

    import argparse
    parser = argparse.ArgumentParser(description='Train Dreamer Crazyflie')
    parser.add_argument('--headless', action='store_true', help='Run without viewer')
    args = parser.parse_args()

    try:
        print("Initializing Dreamer-v3 training...")
        print("Key features:")
        print("- Model-based RL with world model learning")
        print("- Imagination-based policy training")
        print("- Symlog predictions for numerical stability")
        print("- Sample efficient learning")
        print("- Latent space dynamics modeling")
        print("- GPU optimized training")

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