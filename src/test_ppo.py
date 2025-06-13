#!/usr/bin/env python3
"""
Test script for PPO-trained Crazyflie model
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
from torch.distributions import Normal
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

# Same Actor-Critic architecture as training
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        features = self.features(state)
        return features
    
    def act(self, state, deterministic=False):
        features = self.forward(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)
        
        if deterministic:
            action = action_mean
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
        
        action = torch.tanh(action)
        return action

def initialize_simulation():
    """Initialize MuJoCo simulation"""
    global model, data
    
    with sim_lock:
        try:
            model = mujoco.MjModel.from_xml_path('scene_ppo.xml')
        except:
            model = mujoco.MjModel.from_xml_path('scene.xml')
        
        data = mujoco.MjData(model)
        model.opt.timestep = 0.01
        model.opt.iterations = 20
        model.opt.tolerance = 1e-3
        
        reset_simulation()

def reset_simulation(randomize=False, difficulty=0.5):
    """Reset simulation with optional randomization"""
    global model, data
    
    with sim_lock:
        try:
            mujoco.mj_resetData(model, data)
            
            if randomize:
                pos_range = 0.05 + 0.15 * difficulty
                vel_range = 0.05 + 0.15 * difficulty
                angle_range = 0.02 + 0.08 * difficulty
                
                # Position randomization
                data.qpos[0] = np.random.uniform(-pos_range, pos_range)
                data.qpos[1] = np.random.uniform(-pos_range, pos_range)
                data.qpos[2] = np.random.uniform(0.95, 1.05)
                
                # Orientation randomization
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
                # Fixed start position
                data.qpos[:3] = [0.0, 0.0, 1.0]
                data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                data.qvel[:] = 0.0
            
            # Start near hover thrust
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
    """Apply continuous action"""
    global model, data
    
    with sim_lock:
        try:
            data.ctrl[0] = 0.26 + action[0] * 0.04
            data.ctrl[1] = action[1] * 0.03
            data.ctrl[2] = action[2] * 0.03
            data.ctrl[3] = action[3] * 0.02
            
            data.ctrl[1:] *= 0.95
            
            for _ in range(4):
                mujoco.mj_step(model, data)
                
        except Exception as e:
            print(f"Error applying action: {e}")

def load_model(checkpoint_path):
    """Load trained PPO model"""
    policy = ActorCritic(state_dim=18, action_dim=4).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint['model_state_dict'])
        policy.eval()
        
        print(f"Loaded model from episode {checkpoint['episode']}")
        print(f"Training reward: {checkpoint['reward']:.2f}")
        if 'curriculum_stage' in checkpoint:
            print(f"Curriculum stage: {checkpoint['curriculum_stage']}")
        
        return policy
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Global test flag
test_active = True

def test_model(policy, num_episodes=20, max_steps=1000, visualize=True):
    """Comprehensive model testing"""
    global test_active
    
    print(f"\n{'='*70}")
    print("PPO MODEL EVALUATION")
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
            
            state = get_observation()
            episode_reward = 0
            positions = []
            velocities = []
            actions = []
            thrust_values = []
            
            for step in range(max_steps):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy.act(state_tensor, deterministic=True)
                    action_np = action.cpu().numpy()[0]
                
                actions.append(action_np)
                apply_continuous_action(action_np)
                
                # Record data
                pos = data.qpos[0:3].copy()
                vel = data.qvel[0:3].copy()
                positions.append(pos)
                velocities.append(np.linalg.norm(vel))
                thrust_values.append(data.ctrl[0])
                
                # Calculate reward (same as training)
                position_error = np.linalg.norm(pos - np.array([0.0, 0.0, 1.0]))
                reward = 5.0 - position_error * 10.0 - np.linalg.norm(vel) * 3.0
                
                if position_error < 0.1:
                    reward += 20.0
                elif position_error < 0.2:
                    reward += 10.0
                
                episode_reward += reward
                
                # Check termination
                if pos[2] < 0.2 or pos[2] > 2.5 or abs(pos[0]) > 1.5 or abs(pos[1]) > 1.5:
                    break
                
                state = get_observation()
                
                if visualize:
                    time.sleep(0.02)
            
            # Calculate metrics
            positions = np.array(positions)
            errors = np.linalg.norm(positions - np.array([0.0, 0.0, 1.0]), axis=1)
            
            metrics = {
                'reward': episode_reward,
                'episode_length': len(positions),
                'final_error': errors[-1] if len(errors) > 0 else float('inf'),
                'mean_error': np.mean(errors),
                'min_error': np.min(errors),
                'time_within_10cm': np.mean(errors < 0.1),
                'time_within_20cm': np.mean(errors < 0.2),
                'mean_velocity': np.mean(velocities),
                'action_magnitude': np.mean([np.linalg.norm(a) for a in actions]),
                'action_smoothness': np.std([np.linalg.norm(a) for a in actions]),
                'thrust_stability': np.std(thrust_values)
            }
            
            config_results.append(metrics)
            
            print(f"Episode {episode+1}: R={episode_reward:.1f}, "
                  f"FinalErr={errors[-1]:.3f}, "
                  f"Time<10cm={metrics['time_within_10cm']:.2f}, "
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
    easy_precision = all_results["Easy (Fixed)"][5]['mean'] if "Easy (Fixed)" in all_results else 0  # time_within_10cm
    hard_precision = all_results["Hard"][5]['mean'] if "Hard" in all_results else 0  # time_within_10cm
    
    print(f"\n{'='*70}")
    print("PERFORMANCE ASSESSMENT")
    print(f"{'='*70}")
    
    if easy_precision > 0.8:
        print("🏆 EXCELLENT: Achieves precise hovering (>80% time within 0.1m)")
    elif easy_precision > 0.6:
        print("✅ GOOD: Solid hovering performance (>60% time within 0.1m)")
    elif easy_precision > 0.4:
        print("⚠️  MODERATE: Reasonable hovering (>40% time within 0.1m)")
    else:
        print("❌ POOR: Struggles with hovering task")
    
    if hard_precision > 0.3:
        print("🎯 ROBUST: Maintains performance even in difficult conditions")
    elif hard_precision > 0.1:
        print("📊 ADAPTIVE: Some degradation but still functional in hard mode")
    else:
        print("⚠️  LIMITED: Performance degrades significantly with difficulty")
    
    # Action diversity check
    easy_results = [r for r in all_results["Easy (Fixed)"] if r['metric'] == 'action_smoothness'][0]
    if easy_results['std'] > 0.001:
        print("✅ DIVERSE: Shows varied control strategies")
    else:
        print("⚠️  RIGID: Limited behavioral diversity")
    
    test_active = False
    return all_results

def main():
    """Main test function"""
    global test_active, viewer_handle
    
    parser = argparse.ArgumentParser(description='Test PPO Crazyflie Model')
    parser.add_argument('--episodes', type=int, default=20, help='Number of test episodes')
    parser.add_argument('--model', type=str, help='Specific model checkpoint to load')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    try:
        print("Initializing simulation...")
        initialize_simulation()
        
        # Find model to load
        if args.model:
            checkpoint_path = f"models/{args.model}"
        else:
            # Look for best model first
            if os.path.exists('models/crazyflie_ppo_best.pth'):
                checkpoint_path = 'models/crazyflie_ppo_best.pth'
            else:
                # Find latest checkpoint
                ppo_checkpoints = [f for f in os.listdir('models') 
                                 if f.startswith('crazyflie_ppo_checkpoint_') and f.endswith('.pth')]
                if ppo_checkpoints:
                    latest = max(ppo_checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    checkpoint_path = f"models/{latest}"
                else:
                    print("No PPO models found!")
                    return
        
        print(f"Loading model: {checkpoint_path}")
        policy = load_model(checkpoint_path)
        if policy is None:
            return
        
        print(f"\nTesting PPO model with {args.episodes} episodes...")
        print("This will test across multiple difficulty levels")
        
        # Start test in separate thread
        test_thread = threading.Thread(
            target=test_model, 
            args=(policy, args.episodes, 1000, not args.no_viz),
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