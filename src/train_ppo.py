#!/usr/bin/env python3
"""
Crazyflie PPO (Proximal Policy Optimization) Training
Addresses overfitting and exploration issues from DQN
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
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Global simulation lock
sim_lock = threading.RLock()

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
os.makedirs('logs/crazyflie_ppo', exist_ok=True)
os.makedirs('models', exist_ok=True)
writer = SummaryWriter('logs/crazyflie_ppo')

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
            model = mujoco.MjModel.from_xml_path('scene_ppo.xml')
        except:
            model = mujoco.MjModel.from_xml_path('scene.xml')
        
        data = mujoco.MjData(model)
        
        # Better physics settings for smoother control
        model.opt.timestep = 0.01
        model.opt.iterations = 20
        model.opt.tolerance = 1e-3
        
        reset_simulation()

def reset_simulation(randomize=True, difficulty=0.5):
    """Reset with curriculum learning - difficulty increases over time"""
    global model, data
    with sim_lock:
        try:
            mujoco.mj_resetData(model, data)
            
            if randomize:
                # Curriculum learning: start easy, increase difficulty
                pos_range = 0.05 + 0.15 * difficulty
                vel_range = 0.05 + 0.15 * difficulty
                angle_range = 0.02 + 0.08 * difficulty
                
                # Position randomization
                data.qpos[0] = np.random.uniform(-pos_range, pos_range)
                data.qpos[1] = np.random.uniform(-pos_range, pos_range)
                data.qpos[2] = np.random.uniform(0.95, 1.05)
                
                # Small random orientation
                angle = np.random.uniform(-angle_range, angle_range)
                axis = np.random.randn(3)
                axis = axis / np.linalg.norm(axis)
                
                data.qpos[3] = np.cos(angle/2)
                data.qpos[4] = np.sin(angle/2) * axis[0]
                data.qpos[5] = np.sin(angle/2) * axis[1]
                data.qpos[6] = np.sin(angle/2) * axis[2]
                
                # Velocity randomization
                data.qvel[:] = np.random.uniform(-vel_range, vel_range, size=data.qvel.shape)
            else:
                # Fixed start
                data.qpos[:3] = [0.0, 0.0, 1.0]
                data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                data.qvel[:] = 0.0
            
            # Start near hover thrust
            data.ctrl[0] = 0.26
            data.ctrl[1:] = 0.0
            
            mujoco.mj_forward(model, data)
            
        except Exception as e:
            print(f"Error in reset: {e}")

# Actor-Critic Network for PPO
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Actor (policy) network
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Learnable standard deviation for actions
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        """Extract features from state"""
        features = self.features(state)
        return features
    
    def act(self, state, deterministic=False):
        """Sample an action from the policy"""
        features = self.forward(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)
        
        if deterministic:
            action = action_mean
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
        
        # Apply tanh to bound actions to [-1, 1]
        action = torch.tanh(action)
        return action
    
    def evaluate(self, state, action):
        """Evaluate states and actions for PPO updates"""
        features = self.forward(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)
        
        # Create distribution
        dist = Normal(action_mean, action_std)
        
        # Handle tanh transformation
        action_logits = torch.atanh(torch.clamp(action, -0.999, 0.999))
        log_prob = dist.log_prob(action_logits).sum(dim=-1)
        
        # Entropy for exploration bonus
        entropy = dist.entropy().sum(dim=-1)
        
        # State value
        value = self.critic(features).squeeze(-1)
        
        return log_prob, value, entropy

def get_observation():
    """Get 18-dimensional observation vector"""
    global model, data
    with sim_lock:
        try:
            # Basic state
            pos = data.qpos[0:3].copy()
            vel = data.qvel[0:3].copy()
            quat = data.qpos[3:7].copy()
            angvel = data.qvel[3:6].copy()
            
            # Normalize quaternion
            quat = quat / (np.linalg.norm(quat) + 1e-8)
            
            # Extract Euler angles from quaternion
            w, x, y, z = quat
            roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
            pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
            yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            
            # Target position
            target = np.array([0.0, 0.0, 1.0])
            position_error = target - pos
            distance_to_target = np.linalg.norm(position_error)
            
            # Previous control inputs
            controls = data.ctrl.copy()
            
            # Build observation vector (18 dimensions)
            obs = np.concatenate([
                position_error / 2.0,                    # 3 - normalized position error
                vel / 2.0,                               # 3 - normalized velocity
                np.array([roll, pitch, yaw]),            # 3 - orientation (Euler angles)
                angvel / 5.0,                            # 3 - normalized angular velocity
                controls[:1] / 0.35,                     # 1 - normalized thrust
                controls[1:] / 0.1,                      # 3 - normalized moments
                [distance_to_target / 2.0],              # 1 - normalized distance to target
                [np.tanh(pos[2] - 1.0)]                  # 1 - height error feature
            ])
            
            return obs.astype(np.float32)
            
        except Exception as e:
            print(f"Error getting observation: {e}")
            return np.zeros(18, dtype=np.float32)

def apply_continuous_action(action):
    """Apply continuous control action"""
    global model, data
    with sim_lock:
        try:
            # Conservative action mapping for stability
            # Action bounds: [-1, 1] from tanh activation
            
            # Thrust: map to reasonable range around hover
            data.ctrl[0] = 0.26 + action[0] * 0.04  # [0.22, 0.30]
            
            # Moments: small control authority for stability  
            data.ctrl[1] = action[1] * 0.03  # roll moment
            data.ctrl[2] = action[2] * 0.03  # pitch moment  
            data.ctrl[3] = action[3] * 0.02  # yaw moment (smaller)
            
            # Apply momentum decay for smoother control
            data.ctrl[1:] *= 0.95
            
            # Step simulation multiple times for stability
            for _ in range(4):  # 4 steps = 0.04s at 0.01s timestep
                mujoco.mj_step(model, data)
                
        except Exception as e:
            print(f"Error in apply_action: {e}")
            reset_simulation()

def calculate_reward():
    """Calculate reward for current state"""
    global model, data
    with sim_lock:
        try:
            pos = data.qpos[0:3].copy()
            vel = data.qvel[0:3].copy()
            angvel = data.qvel[3:6].copy()
            
            # Target position
            target = np.array([0.0, 0.0, 1.0])
            position_error = np.linalg.norm(pos - target)
            
            # Basic reward components
            position_reward = 5.0 - position_error * 10.0
            velocity_penalty = np.linalg.norm(vel) * 3.0
            angular_penalty = np.linalg.norm(angvel) * 2.0
            
            # Bonus rewards for good hovering
            if position_error < 0.1:
                position_reward += 20.0
            elif position_error < 0.2:
                position_reward += 10.0
            
            # Smooth control bonus
            control_penalty = np.linalg.norm(data.ctrl[1:]) * 1.0
            
            # Total reward
            reward = position_reward - velocity_penalty - angular_penalty - control_penalty
            
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
            
            # Termination conditions
            if pos[2] < 0.2:  # Too low (crash)
                return True
            if pos[2] > 2.5:  # Too high
                return True
            if abs(pos[0]) > 1.5 or abs(pos[1]) > 1.5:  # Too far horizontally
                return True
                
            return False
            
        except Exception as e:
            print(f"Error checking done condition: {e}")
            return True

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim=18, action_dim=4):
        # PPO hyperparameters
        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.clip_epsilon = 0.2
        self.c1 = 1.0  # Value loss coefficient
        self.c2 = 0.01  # Entropy coefficient
        self.lr = 3e-4
        self.max_grad_norm = 0.5
        
        # Training parameters
        self.n_steps = 2048
        self.n_epochs = 10
        self.batch_size = 64
        
        # Networks
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Buffers
        self.reset_buffers()
        
    def reset_buffers(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def store_transition(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def compute_returns_and_advantages(self, last_value):
        """Compute returns and GAE advantages"""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)
        
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        last_return = last_value
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma * last_return * (1 - dones[t])
            
            td_error = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = td_error + self.gamma * self.lambda_gae * last_advantage * (1 - dones[t])
            
            last_return = returns[t]
            last_advantage = advantages[t]
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
        
    def update(self, last_value):
        """PPO update using collected experience"""
        if len(self.states) < self.n_steps:
            return {}
            
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(last_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.FloatTensor(np.array(self.actions)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        
        # PPO epochs
        for epoch in range(self.n_epochs):
            # Mini-batch updates
            batch_size = min(self.batch_size, len(states))
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                
                batch_states = states[start:end]
                batch_actions = actions[start:end]
                batch_old_log_probs = old_log_probs[start:end]
                batch_returns = returns[start:end]
                batch_advantages = advantages[start:end]
                
                # Evaluate current policy
                log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)
                
                # Compute ratios and clipped objective
                ratios = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns)
                entropy_bonus = entropy.mean()
                
                # Total loss
                loss = policy_loss + self.c1 * value_loss - self.c2 * entropy_bonus
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # Final evaluation for logging
        with torch.no_grad():
            log_probs, values, entropy = self.policy.evaluate(states, actions)
            final_ratio = torch.exp(log_probs - old_log_probs)
            
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'mean_ratio': final_ratio.mean().item(),
            'max_ratio': final_ratio.max().item(),
            'min_ratio': final_ratio.min().item()
        }

# Global training flag
training_active = True

def training_loop():
    """Main PPO training loop"""
    global training_active
    
    agent = PPOAgent()
    
    num_episodes = 5000
    max_steps_per_episode = 1000
    episode = 0
    total_steps = 0
    best_reward = -float('inf')
    reward_history = []
    
    # Curriculum learning
    curriculum_stages = [0.0, 0.3, 0.5, 0.7, 1.0]
    current_stage = 0
    stage_threshold = 300.0  # Average reward to advance stage
    
    while episode < num_episodes and training_active:
        try:
            # Check curriculum progression
            if len(reward_history) >= 100:
                avg_reward = np.mean(reward_history[-100:])
                if avg_reward > stage_threshold and current_stage < len(curriculum_stages) - 1:
                    current_stage += 1
                    print(f"Advancing to curriculum stage {current_stage} (difficulty: {curriculum_stages[current_stage]})")
            
            # Reset environment
            reset_simulation(randomize=True, difficulty=curriculum_stages[current_stage])
            state = get_observation()
            episode_reward = 0
            episode_steps = 0
            
            # Collect trajectory
            for step in range(agent.n_steps):
                if not training_active:
                    break
                
                # Get action from policy
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = agent.policy.act(state_tensor, deterministic=False)
                    
                    # Also get value and log prob for PPO
                    features = agent.policy.forward(state_tensor)
                    value = agent.policy.critic(features).item()
                    
                    # Calculate log prob
                    action_mean = agent.policy.actor_mean(features)
                    action_std = torch.exp(agent.policy.actor_log_std)
                    dist = Normal(action_mean, action_std)
                    action_logits = torch.atanh(torch.clamp(action, -0.999, 0.999))
                    log_prob = dist.log_prob(action_logits).sum().item()
                
                action_np = action.cpu().numpy()[0]
                
                # Apply action and get response
                apply_continuous_action(action_np)
                next_state = get_observation()
                reward = calculate_reward()
                done = is_done()
                
                # Store transition
                agent.store_transition(state, action_np, reward, value, log_prob, done)
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                state = next_state
                
                # Early termination conditions
                if done or episode_steps >= max_steps_per_episode:
                    episode += 1
                    reward_history.append(episode_reward)
                    avg_reward = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else episode_reward
                    
                    # Logging
                    writer.add_scalar('Episode/Reward', episode_reward, episode)
                    writer.add_scalar('Episode/Length', episode_steps, episode)
                    writer.add_scalar('Episode/AvgReward100', avg_reward, episode)
                    
                    if episode % 10 == 0:
                        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Steps: {episode_steps}, "
                              f"Avg100: {avg_reward:.2f}, Stage: {current_stage}")
                    
                    # Save best model
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        torch.save({
                            'episode': episode,
                            'model_state_dict': agent.policy.state_dict(),
                            'optimizer_state_dict': agent.optimizer.state_dict(),
                            'reward': episode_reward,
                            'avg_reward': avg_reward,
                            'curriculum_stage': current_stage
                        }, 'models/crazyflie_ppo_best.pth')
                    
                    # Regular checkpoint
                    if episode % 100 == 0:
                        torch.save({
                            'episode': episode,
                            'model_state_dict': agent.policy.state_dict(),
                            'optimizer_state_dict': agent.optimizer.state_dict(),
                            'reward': episode_reward,
                            'avg_reward': avg_reward,
                            'curriculum_stage': current_stage
                        }, f'models/crazyflie_ppo_checkpoint_{episode}.pth')
                    
                    # Reset for next episode
                    reset_simulation(randomize=True, difficulty=curriculum_stages[current_stage])
                    state = get_observation()
                    episode_reward = 0
                    episode_steps = 0
                
                # Update policy when buffer is full
                if len(agent.states) >= agent.n_steps:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                        features = agent.policy.forward(state_tensor)
                        last_value = agent.policy.critic(features).item()
                    
                    # PPO update
                    update_stats = agent.update(last_value)
                    
                    # Log training stats
                    writer.add_scalar('Policy_Loss', update_stats['policy_loss'], total_steps)
                    writer.add_scalar('Value_Loss', update_stats['value_loss'], total_steps)
                    writer.add_scalar('Entropy', update_stats['entropy'], total_steps)
                    writer.add_scalar('Mean_Ratio', update_stats['mean_ratio'], total_steps)
                    
                    # Reset buffers
                    agent.reset_buffers()
                
                time.sleep(0.01)  # Real-time factor
                
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
        print("Initializing PPO training...")
        print("Key improvements over DQN:")
        print("- Continuous action space for smoother control")
        print("- Stochastic policy for better exploration") 
        print("- Curriculum learning (progressive difficulty)")
        print("- Entropy bonus to maintain exploration")
        print("- More stable learning with PPO clipping")
        
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