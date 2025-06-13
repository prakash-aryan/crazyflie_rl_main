#!/usr/bin/env python3
"""
Crazyflie SAC (Soft Actor-Critic) Training
Maximum entropy RL for better exploration and robustness
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

warnings.filterwarnings('ignore')

# Global simulation lock
sim_lock = threading.RLock()

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
os.makedirs('logs/crazyflie_sac', exist_ok=True)
os.makedirs('models', exist_ok=True)
writer = SummaryWriter('logs/crazyflie_sac')

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
            model = mujoco.MjModel.from_xml_path('scene_sac.xml')
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
                # More structured randomization
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

# Neural Networks for SAC
class SoftQNetwork(nn.Module):
    """Soft Q-network for SAC"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SoftQNetwork, self).__init__()
        
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
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class PolicyNetwork(nn.Module):
    """Stochastic policy network for SAC"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
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

# Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    """Memory buffer with prioritized sampling"""
    def __init__(self, capacity=1000000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = self.max_priority
        
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = list(zip(*samples))
        states = torch.FloatTensor(batch[0]).to(device)
        actions = torch.FloatTensor(batch[1]).to(device)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(batch[3]).to(device)
        dones = torch.BoolTensor(batch[4]).unsqueeze(1).to(device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

# State representation with history
class StateHistory:
    """Maintain state history for better observability"""
    def __init__(self, history_length=3):
        self.history_length = history_length
        self.history = deque(maxlen=history_length)
        
    def reset(self):
        self.history.clear()
        
    def add(self, state):
        self.history.append(state)
        
    def get(self):
        # Pad with zeros if not enough history
        if len(self.history) < self.history_length:
            padding = [np.zeros_like(self.history[0]) for _ in range(self.history_length - len(self.history))]
            return np.concatenate(padding + list(self.history))
        return np.concatenate(list(self.history))

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
            
            # Acceleration estimate (finite difference)
            acc_estimate = np.zeros(3)  # Would need previous velocity for this
            
            # Observation vector
            obs = np.concatenate([
                target_error / 2.0,           # 3 - normalized target error
                vel / 2.0,                    # 3 - normalized velocity
                np.array([roll, pitch, yaw]), # 3 - euler angles
                angvel / 5.0,                 # 3 - normalized angular velocity
                controls[:1] / 0.35,          # 1 - normalized thrust
                controls[1:] / 0.05,          # 3 - normalized moments
                np.array([target_dist]) / 2.0 # 1 - normalized distance to target
            ])
            
            return obs.astype(np.float32)
            
        except Exception as e:
            print(f"Error in observation: {e}")
            return np.zeros(18, dtype=np.float32)

def compute_reward():
    """Dense reward function encouraging hovering"""
    global data
    pos = data.qpos[0:3]
    vel = data.qvel[0:3]
    
    # Target position (hover at 1m height)
    target = np.array([0.0, 0.0, 1.0])
    
    # Distance to target
    position_error = np.linalg.norm(pos - target)
    height_error = abs(pos[2] - 1.0)
    xy_error = np.linalg.norm(pos[0:2])
    
    # Velocity penalties
    linear_velocity = np.linalg.norm(vel)
    
    # Dense reward calculation
    reward = 3.0  # Base alive bonus
    
    # Position rewards (exponential to encourage precision)
    reward += np.exp(-2.0 * position_error) * 20.0  # Position reward
    reward += np.exp(-3.0 * height_error) * 15.0    # Height priority
    reward += np.exp(-2.0 * xy_error) * 10.0        # XY position
    
    # Velocity penalties (encourage stillness)
    reward -= linear_velocity * 2.0
    
    # Attitude rewards (prefer upright)
    quat = data.qpos[3:7]
    quat = quat / np.linalg.norm(quat)
    upright_bonus = quat[0] ** 2  # w component for upright
    reward += upright_bonus * 5.0
    
    # Control smoothness (penalize large control inputs)
    control_penalty = np.sum(np.abs(data.ctrl[1:])) * 0.5
    reward -= control_penalty
    
    # Bonus for being very close to target
    if position_error < 0.1:
        reward += 10.0
    if position_error < 0.05:
        reward += 20.0
    
    return reward

def is_terminal():
    """Check if episode should terminate"""
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
            # More conservative action mapping
            
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

# SAC Agent
class SACAgent:
    def __init__(self, state_dim=18, action_dim=4, hidden_dim=256):
        # Networks
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # Twin Q-networks
        self.critic1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # Copy parameters to targets
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)
        
        # Temperature parameter
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        # Memory
        self.memory = PrioritizedReplayBuffer(capacity=1000000)
        
        # Training parameters
        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 0.005
        self.updates = 0
    
    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            if evaluate:
                _, _, action = self.actor.sample(state_tensor)
            else:
                action, _, _ = self.actor.sample(state_tensor)
            return action.cpu().numpy()[0]
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        
        # Compute target Q-values
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q1_target = self.critic1_target(next_states, next_actions)
            q2_target = self.critic2_target(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target) - self.log_alpha.exp() * next_log_probs
            target_value = rewards + (1 - dones.float()) * self.gamma * min_q_target
        
        # Current Q-values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_value)
        critic2_loss = F.mse_loss(current_q2, target_value)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update priorities
        td_errors = torch.abs(current_q1 - target_value).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        # Update actor
        new_actions, log_probs, _ = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.log_alpha.exp() * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (temperature)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        self.updates += 1
        
        # Logging
        if self.updates % 100 == 0:
            writer.add_scalar('Loss/Critic1', critic1_loss.item(), self.updates)
            writer.add_scalar('Loss/Critic2', critic2_loss.item(), self.updates)
            writer.add_scalar('Loss/Actor', actor_loss.item(), self.updates)
            writer.add_scalar('Loss/Alpha', alpha_loss.item(), self.updates)
            writer.add_scalar('Values/Alpha', self.log_alpha.exp().item(), self.updates)
            writer.add_scalar('Values/Q', q_new.mean().item(), self.updates)
    
    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Training flag
training_active = True

def training_loop():
    """Main SAC training loop"""
    global training_active
    
    agent = SACAgent()
    state_history = StateHistory(history_length=1)  # Start with no history
    
    num_episodes = 5000
    max_steps_per_episode = 1000
    
    # Training metrics
    episode = 0
    total_steps = 0
    best_reward = -float('inf')
    reward_history = deque(maxlen=100)
    
    # Curriculum learning
    difficulty = 0.3
    difficulty_threshold = 50.0
    difficulty_increment = 0.1
    
    while training_active and episode < num_episodes:
        try:
            reset_simulation(randomize=True, difficulty=difficulty)
            state = get_observation()
            state_history.reset()
            state_history.add(state)
            
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps_per_episode):
                if not training_active:
                    break
                
                # Select action
                action = agent.select_action(state)
                
                # Apply action
                apply_continuous_action(action)
                
                # Get new state and reward
                next_state = get_observation()
                reward = compute_reward()
                done = is_terminal()
                
                # Store transition
                agent.memory.push(state, action, reward, next_state, done)
                
                # Update SAC
                if len(agent.memory) > agent.batch_size:
                    agent.update()
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                if done:
                    break
                
                state = next_state
                state_history.add(state)
            
            # Episode finished
            episode += 1
            reward_history.append(episode_reward)
            avg_reward = sum(reward_history) / len(reward_history)
            
            # Log episode results
            writer.add_scalar('Episode/Reward', episode_reward, episode)
            writer.add_scalar('Episode/Steps', episode_steps, episode)
            writer.add_scalar('Episode/AvgReward', avg_reward, episode)
            writer.add_scalar('Episode/Difficulty', difficulty, episode)
            writer.add_scalar('Episode/Alpha', agent.log_alpha.exp().item(), episode)
            
            if episode % 10 == 0:
                logging.info(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                           f"Steps: {episode_steps}, Avg100: {avg_reward:.2f}, "
                           f"Difficulty: {difficulty:.2f}, Alpha: {agent.log_alpha.exp().item():.4f}")
            
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
                    'critic1_state_dict': agent.critic1.state_dict(),
                    'critic2_state_dict': agent.critic2.state_dict(),
                    'log_alpha': agent.log_alpha,
                    'reward': episode_reward,
                    'difficulty': difficulty
                }, 'models/crazyflie_sac_best.pth')
            
            # Regular checkpoint
            if episode % 100 == 0:
                torch.save({
                    'episode': episode,
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic1_state_dict': agent.critic1.state_dict(),
                    'critic2_state_dict': agent.critic2.state_dict(),
                    'log_alpha': agent.log_alpha,
                    'reward': episode_reward,
                    'avg_reward': avg_reward,
                    'difficulty': difficulty
                }, f'models/crazyflie_sac_checkpoint_{episode}.pth')
                
        except Exception as e:
            print(f"Error in training: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1.0)
            continue
    
    training_active = False
    print("Training completed!")
    print(f"Best reward: {best_reward:.2f}")
    writer.close()

def main():
    """Main function"""
    global training_active, viewer_handle
    
    try:
        print("Initializing SAC training...")
        print("Key features:")
        print("- Maximum entropy framework for exploration")
        print("- Automatic temperature tuning")
        print("- Twin Q-networks to reduce overestimation")
        print("- Prioritized experience replay")
        print("- Dense reward shaping")
        print("- Conservative action space")
        
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