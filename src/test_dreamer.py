#!/usr/bin/env python3
"""
Test script for Dreamer-trained Crazyflie model
Comprehensive evaluation with multiple metrics
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import mujoco
from mujoco import viewer
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import threading
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global variables
model = None
data = None
viewer_handle = None
sim_lock = threading.RLock()

# Dreamer Network Components (same as training)
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

# Symlog functions for numerical stability (Dreamer-v3 feature)
def symlog(x):
    """Symmetric logarithm for better numerical stability"""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x):
    """Inverse of symlog"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def initialize_simulation():
    """Initialize MuJoCo simulation"""
    global model, data
    
    with sim_lock:
        try:
            model = mujoco.MjModel.from_xml_path('scene_dreamer.xml')
        except:
            model = mujoco.MjModel.from_xml_path('scene.xml')
        
        data = mujoco.MjData(model)
        
        # Set same physics parameters as training
        model.opt.timestep = 0.005
        model.opt.iterations = 50
        model.opt.tolerance = 1e-4
        
        reset_simulation()

def reset_simulation(randomize=True, difficulty=0.5):
    """Reset simulation with specified randomization"""
    global model, data
    with sim_lock:
        try:
            mujoco.mj_resetData(model, data)
            
            if randomize:
                # Controlled randomization based on difficulty
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

def get_observation():
    """Get observation vector (same as training)"""
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
    """Apply continuous action (same as training)"""
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
            print(f"Error applying action: {e}")

def load_model(checkpoint_path):
    """Load trained Dreamer model"""
    world_model = WorldModel(obs_dim=18, action_dim=4, latent_dim=32, hidden_dim=256).to(device)
    actor = Actor(latent_dim=32, action_dim=4, hidden_dim=256).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        world_model.load_state_dict(checkpoint['world_model_state_dict'])
        actor.load_state_dict(checkpoint['actor_state_dict'])
        
        world_model.eval()
        actor.eval()
        
        print(f"Loaded model from episode {checkpoint['episode']}")
        if 'reward' in checkpoint:
            print(f"Training reward: {checkpoint['reward']:.2f}")
        if 'difficulty' in checkpoint:
            print(f"Training difficulty: {checkpoint['difficulty']}")
        if 'total_steps' in checkpoint:
            print(f"Total training steps: {checkpoint['total_steps']}")
        
        return world_model, actor
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Global test flag
test_active = True

def test_model(world_model, actor, num_episodes=20, max_steps=1000, visualize=True):
    """Comprehensive model testing"""
    global test_active
    
    print(f"\n{'='*70}")
    print("DREAMER MODEL EVALUATION")
    print(f"{'='*70}")
    
    # Test configurations
    test_configs = [
        {"name": "Easy (Fixed)", "randomize": False, "difficulty": 0.0},
        {"name": "Easy (Random)", "randomize": True, "difficulty": 0.3},
        {"name": "Medium", "randomize": True, "difficulty": 0.5},
        {"name": "Hard", "randomize": True, "difficulty": 0.7},
        {"name": "Very Hard", "randomize": True, "difficulty": 1.0}
    ]
    
    all_results = defaultdict(list)
    
    for config in test_configs:
        if not test_active:
            break
            
        print(f"\n--- Testing {config['name']} ---")
        config_results = []
        
        for episode in range(num_episodes // len(test_configs)):
            if not test_active:
                break
                
            reset_simulation(config['randomize'], config['difficulty'])
            
            obs = get_observation()
            episode_reward = 0
            positions = []
            velocities = []
            actions = []
            thrust_values = []
            latent_states = []
            
            for step in range(max_steps):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    
                    # Encode observation to latent space
                    latent_dist = world_model.encode(obs_tensor)
                    latent = latent_dist.mode  # Use mode for deterministic testing
                    
                    # Get action from policy
                    action_dist = actor(latent)
                    action = action_dist.mode.cpu().numpy().squeeze()  # Deterministic action
                
                # Store data for analysis
                positions.append(data.qpos[0:3].copy())
                velocities.append(data.qvel[0:3].copy())
                actions.append(action.copy())
                thrust_values.append(data.ctrl[0])
                latent_states.append(latent.cpu().numpy().squeeze())
                
                apply_continuous_action(action)
                obs = get_observation()
                
                # Simple reward for testing
                pos = data.qpos[0:3]
                target = np.array([0.0, 0.0, 1.0])
                distance = np.linalg.norm(pos - target)
                reward = max(0, 100 - 100 * distance)  # Simple distance-based reward
                episode_reward += reward
                
                # Check termination
                if pos[2] < 0.3 or distance > 2.0:
                    break
                
                if not visualize:
                    time.sleep(0.01)
            
            # Calculate metrics
            positions = np.array(positions)
            velocities = np.array(velocities)
            actions = np.array(actions)
            thrust_values = np.array(thrust_values)
            latent_states = np.array(latent_states)
            
            if len(positions) > 0:
                target = np.array([0.0, 0.0, 1.0])
                distances = np.linalg.norm(positions - target, axis=1)
                
                # Key metrics
                metrics = {
                    'episode_reward': episode_reward,
                    'episode_length': len(positions),
                    'final_distance': distances[-1] if len(distances) > 0 else float('inf'),
                    'avg_distance': np.mean(distances),
                    'min_distance': np.min(distances),
                    'time_within_01': np.mean(distances < 0.1),
                    'time_within_02': np.mean(distances < 0.2),
                    'avg_velocity': np.mean(np.linalg.norm(velocities, axis=1)),
                    'action_magnitude': np.mean(np.linalg.norm(actions, axis=1)),
                    'thrust_std': np.std(thrust_values),
                    'latent_consistency': np.std(latent_states) if len(latent_states) > 1 else 0
                }
                
                config_results.append(metrics)
                
                print(f"Episode {episode+1}: Reward={episode_reward:.1f}, "
                      f"FinalDist={distances[-1]:.3f}, "
                      f"TimeIn0.1m={metrics['time_within_01']:.2f}, "
                      f"Steps={len(positions)}")
        
        if config_results:
            # Aggregate results for this configuration
            for metric in config_results[0].keys():
                values = [r[metric] for r in config_results]
                all_results[config['name']].append({
                    'metric': metric,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                })
    
    # Print comprehensive results
    print(f"\n{'='*70}")
    print("DETAILED RESULTS")
    print(f"{'='*70}")
    
    for config_name, results in all_results.items():
        print(f"\n{config_name}:")
        print("-" * 50)
        for result in results:
            metric = result['metric']
            mean = result['mean']
            std = result['std']
            print(f"{metric:20s}: {mean:8.3f} ± {std:6.3f}")
    
    # Key performance indicators
    print(f"\n{'='*70}")
    print("KEY PERFORMANCE INDICATORS")
    print(f"{'='*70}")
    
    # Extract key metrics
    easy_precision = all_results["Easy (Fixed)"][5]['mean'] if "Easy (Fixed)" in all_results else 0  # time_within_01
    hard_precision = all_results["Hard"][5]['mean'] if "Hard" in all_results else 0  # time_precise
    
    print(f"\n{'='*70}")
    print("PERFORMANCE ASSESSMENT")
    print(f"{'='*70}")
    
    if easy_precision > 0.8:
        print("EXCELLENT: Achieves precise hovering (>80% time within 0.1m)")
    elif easy_precision > 0.6:
        print("GOOD: Solid hovering performance (>60% time within 0.1m)")
    elif easy_precision > 0.4:
        print("MODERATE: Reasonable hovering (>40% time within 0.1m)")
    else:
        print("POOR: Struggles with hovering task")
    
    if hard_precision > 0.3:
        print("ROBUST: Maintains performance even in difficult conditions")
    elif hard_precision > 0.1:
        print("ADAPTIVE: Some degradation but still functional in hard mode")
    else:
        print("LIMITED: Performance degrades significantly with difficulty")
    
    # Model-based specific assessment
    if len(all_results["Easy (Fixed)"]) > 0:
        latent_consistency = [r for r in all_results["Easy (Fixed)"] if r['metric'] == 'latent_consistency'][0]
        if latent_consistency['mean'] < 0.5:
            print("CONSISTENT: Stable latent representations")
        elif latent_consistency['mean'] < 1.0:
            print("MODERATE: Some latent variability")
        else:
            print("VARIABLE: High latent space variability")
    
    # Sample efficiency assessment
    action_magnitude = [r for r in all_results["Easy (Fixed)"] if r['metric'] == 'action_magnitude'][0]
    if action_magnitude['mean'] < 0.3:
        print("EFFICIENT: Conservative, learned control")
    elif action_magnitude['mean'] < 0.6:
        print("MODERATE: Reasonable control effort")
    else:
        print("AGGRESSIVE: High action magnitudes")
    
    print("MODEL-BASED: Uses learned world model for planning and control")
    
    test_active = False
    return all_results

def main():
    """Main test function"""
    global test_active, viewer_handle
    
    parser = argparse.ArgumentParser(description='Test Dreamer Crazyflie Model')
    parser.add_argument('--episodes', type=int, default=20, help='Number of test episodes')
    parser.add_argument('--model', type=str, help='Specific model checkpoint to load')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    try:
        print("Initializing simulation...")
        initialize_simulation()
        
        # Check if models directory exists
        if not os.path.exists('models'):
            print("No models directory found!")
            return
        
        # Find model to load
        if args.model:
            checkpoint_path = f"models/{args.model}"
        else:
            # Look for best Dreamer model first
            if os.path.exists('models/crazyflie_dreamer_best.pth'):
                checkpoint_path = 'models/crazyflie_dreamer_best.pth'
            else:
                # Find latest Dreamer checkpoint
                dreamer_checkpoints = [f for f in os.listdir('models') 
                                     if f.startswith('crazyflie_dreamer_checkpoint_') and f.endswith('.pth')]
                if dreamer_checkpoints:
                    latest = max(dreamer_checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    checkpoint_path = f"models/{latest}"
                else:
                    print("No Dreamer models found!")
                    return
        
        print(f"Loading model: {checkpoint_path}")
        world_model, actor = load_model(checkpoint_path)
        if world_model is None or actor is None:
            return
        
        print(f"\nTesting Dreamer model with {args.episodes} episodes...")
        print("This will test across multiple difficulty levels")
        print("Dreamer uses a learned world model for imagination-based planning")
        
        # Start test in separate thread
        test_thread = threading.Thread(
            target=test_model, 
            args=(world_model, actor, args.episodes, 1000, not args.no_viz),
            daemon=True
        )
        test_thread.start()
        
        if not args.no_viz:
            # Launch viewer
            with mujoco.viewer.launch_passive(model, data) as viewer_handle:
                while viewer_handle.is_running() and test_active:
                    with sim_lock:
                        viewer_handle.sync()
                    time.sleep(0.01)
        else:
            # Wait for test to complete
            test_thread.join()
        
        # Cleanup
        test_active = False
        test_thread.join(timeout=5.0)
        
        print("\nTest completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()