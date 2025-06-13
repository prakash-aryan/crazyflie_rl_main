#!/usr/bin/env python3
"""
Test script for Dueling DQN-trained Crazyflie model
Comprehensive evaluation with value/advantage analysis
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
import threading
import argparse
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

# Dueling DQN architecture (same as training)
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
    
    def get_value_advantage(self, x):
        """Get separate value and advantage estimates for analysis"""
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        return value, advantage

def initialize_simulation():
    """Initialize simulation"""
    global model, data
    with sim_lock:
        try:
            model = mujoco.MjModel.from_xml_path('scene_dueling_dqn.xml')
        except:
            model = mujoco.MjModel.from_xml_path('scene.xml')
        
        data = mujoco.MjData(model)
        model.opt.timestep = 0.02
        model.opt.iterations = 10
        model.opt.tolerance = 1e-2
        
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
                
                data.qpos[0] = np.random.uniform(-pos_range, pos_range)
                data.qpos[1] = np.random.uniform(-pos_range, pos_range)
                data.qpos[2] = np.random.uniform(0.85, 1.15)
                
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

def get_state():
    """Get current state vector (same as training)"""
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

def apply_action(action):
    """Apply discrete action (same as training)"""
    global model, data
    
    with sim_lock:
        try:
            current_thrust = data.ctrl[0]
            
            # Action mapping
            thrust_step = 0.008
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
            elif action == 6:  # Stabilize
                data.ctrl[1] *= 0.6
                data.ctrl[2] *= 0.6
                data.ctrl[3] *= 0.6
            
            # Moment decay
            data.ctrl[1] = np.clip(data.ctrl[1] * 0.9, -0.05, 0.05)
            data.ctrl[2] = np.clip(data.ctrl[2] * 0.9, -0.05, 0.05)
            data.ctrl[3] = np.clip(data.ctrl[3] * 0.9, -0.05, 0.05)
            
            # Step simulation
            for _ in range(2):
                mujoco.mj_step(model, data)
                
        except Exception as e:
            print(f"Error in apply_action: {e}")

def load_trained_model(checkpoint_path):
    """Load trained Dueling DQN model"""
    policy_net = DuelingDQN(n_observations=13, n_actions=7).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        policy_net.eval()
        
        print(f"Loaded model from episode {checkpoint['episode']}")
        print(f"Training reward: {checkpoint['reward']:.2f}")
        if 'curriculum_stage' in checkpoint:
            print(f"Curriculum stage: {checkpoint['curriculum_stage']}")
        
        return policy_net
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Global test flag
test_active = True

def test_model(policy_net, num_episodes=20, max_steps=500, visualize=True):
    """Comprehensive model testing with value/advantage analysis"""
    global test_active
    
    print(f"\n{'='*70}")
    print("DUELING DQN MODEL EVALUATION")
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
                
            randomize = config['randomize']
            difficulty = config['difficulty']
            
            print(f"\n  Episode {episode + 1}: {config['name']}")
            reset_simulation(randomize=randomize, difficulty=difficulty)
            
            total_reward = 0
            steps = 0
            position_errors = []
            heights = []
            velocities = []
            actions_taken = []
            q_values = []
            state_values = []
            advantage_ranges = []
            
            # Let simulation settle
            time.sleep(0.3)
            
            for step in range(max_steps):
                if not test_active:
                    break
                    
                # Get current state
                state = get_state()
                
                # Get action and analyze Q-function
                with torch.no_grad():
                    state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
                    
                    # Get Q-values
                    q_vals = policy_net(state_tensor)
                    action = q_vals.max(1).indices.item()
                    
                    # Get value and advantage decomposition
                    value, advantage = policy_net.get_value_advantage(state_tensor)
                    state_values.append(value.item())
                    advantage_ranges.append((advantage.max() - advantage.min()).item())
                    q_values.append(q_vals.max().item())
                
                actions_taken.append(action)
                
                # Apply action
                apply_action(action)
                
                # Record metrics
                pos = data.qpos[0:3]
                vel = data.qvel[0:3]
                
                desired_position = np.array([0.0, 0.0, 1.0])
                position_error = np.linalg.norm(pos - desired_position)
                velocity_magnitude = np.linalg.norm(vel)
                
                position_errors.append(position_error)
                heights.append(pos[2])
                velocities.append(velocity_magnitude)
                
                # Calculate reward (same as training)
                height_error = abs(pos[2] - 1.0)
                xy_error = np.linalg.norm(pos[0:2])
                
                reward = 15.0
                if position_error < 0.05:
                    reward += 40.0
                elif position_error < 0.1:
                    reward += 25.0
                elif position_error < 0.2:
                    reward += 15.0
                else:
                    reward += max(0.0, 10.0 - position_error * 10.0)
                
                if height_error < 0.02:
                    reward += 25.0
                elif height_error < 0.05:
                    reward += 15.0
                elif height_error < 0.1:
                    reward += 8.0
                
                if velocity_magnitude < 0.1:
                    reward += 10.0
                elif velocity_magnitude < 0.2:
                    reward += 5.0
                
                if position_error < 0.1 and velocity_magnitude < 0.1:
                    reward += 30.0
                elif position_error < 0.2 and velocity_magnitude < 0.2:
                    reward += 15.0
                
                # Penalties
                vertical_velocity = abs(vel[2])
                quat = data.qpos[3:7]
                quat_norm = np.linalg.norm(quat)
                if quat_norm > 0:
                    quat = quat / quat_norm
                w, x, y, z = quat
                upright_penalty = np.sqrt(x*x + y*y)
                
                reward -= velocity_magnitude * 5.0
                reward -= vertical_velocity * 8.0
                reward -= upright_penalty * 10.0
                
                # Time bonus
                reward += min(step * 0.1, 10.0)
                
                # Control effort penalty
                thrust_deviation = abs(data.ctrl[0] - 0.26)
                moment_magnitude = np.linalg.norm(data.ctrl[1:])
                reward -= thrust_deviation * 8.0
                reward -= moment_magnitude * 15.0
                
                # Terminal penalties
                if pos[2] < 0.2 or pos[2] > 2.5:
                    reward = -200.0
                elif xy_error > 1.5:
                    reward = -100.0
                
                total_reward += reward
                steps += 1
                
                # Check termination
                if pos[2] < 0.2 or pos[2] > 2.5 or abs(pos[0]) > 1.5 or abs(pos[1]) > 1.5:
                    break
                
                if visualize:
                    time.sleep(0.02)
            
            # Episode analysis
            if position_errors:
                avg_error = np.mean(position_errors)
                time_precise = np.mean([e < 0.1 for e in position_errors])
                time_good = np.mean([e < 0.2 for e in position_errors])
                avg_velocity = np.mean(velocities)
                avg_height = np.mean(heights)
                
                # Value/Advantage analysis
                avg_state_value = np.mean(state_values) if state_values else 0
                avg_advantage_range = np.mean(advantage_ranges) if advantage_ranges else 0
                max_q_value = np.max(q_values) if q_values else 0
                
                config_results.append({
                    'reward': total_reward,
                    'steps': steps,
                    'avg_error': avg_error,
                    'time_precise': time_precise,
                    'time_good': time_good,
                    'avg_velocity': avg_velocity,
                    'avg_height': avg_height,
                    'avg_state_value': avg_state_value,
                    'avg_advantage_range': avg_advantage_range,
                    'max_q_value': max_q_value
                })
                
                print(f"    Reward: {total_reward:.1f}, Error: {avg_error:.3f}m, "
                      f"Precise: {time_precise:.1%}, V(s): {avg_state_value:.2f}")
        
        # Aggregate results for this configuration
        if config_results:
            metrics = ['reward', 'steps', 'avg_error', 'time_precise', 'time_good', 
                      'avg_velocity', 'avg_height', 'avg_state_value', 'avg_advantage_range', 'max_q_value']
            
            for metric in metrics:
                values = [r[metric] for r in config_results]
                all_results[config['name']].append({
                    'metric': metric,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                })
    
    # Print comprehensive summary
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    
    for config_name, results in all_results.items():
        print(f"\n{config_name}:")
        for r in results:
            if r['metric'] in ['reward', 'avg_error', 'time_precise', 'avg_state_value', 'avg_advantage_range']:
                print(f"  {r['metric']:20s}: {r['mean']:8.3f} ± {r['std']:6.3f}")
    
    # Value/Advantage Analysis
    print(f"\n{'='*70}")
    print("DUELING ARCHITECTURE ANALYSIS")
    print(f"{'='*70}")
    
    for config_name, results in all_results.items():
        state_value_result = [r for r in results if r['metric'] == 'avg_state_value'][0]
        advantage_result = [r for r in results if r['metric'] == 'avg_advantage_range'][0]
        
        print(f"\n{config_name}:")
        print(f"  State Value V(s):     {state_value_result['mean']:8.3f} ± {state_value_result['std']:6.3f}")
        print(f"  Advantage Range:      {advantage_result['mean']:8.3f} ± {advantage_result['std']:6.3f}")
        
        # Analysis
        if state_value_result['mean'] > 50:
            print(f"  → High state values indicate good baseline policy")
        if advantage_result['mean'] > 10:
            print(f"  → Large advantage differences show clear action preferences")
    
    # Overall assessment
    easy_precision = all_results["Easy (Fixed)"][3]['mean'] if "Easy (Fixed)" in all_results else 0
    hard_precision = all_results["Hard"][3]['mean'] if "Hard" in all_results else 0
    
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
    
    test_active = False
    return all_results

def main():
    """Main test function"""
    global test_active, viewer_handle
    
    parser = argparse.ArgumentParser(description='Test Dueling DQN Crazyflie Model')
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
            # Look for best average model first
            if os.path.exists('models/crazyflie_dueling_dqn_best_avg.pth'):
                checkpoint_path = 'models/crazyflie_dueling_dqn_best_avg.pth'
            elif os.path.exists('models/crazyflie_dueling_dqn_best.pth'):
                checkpoint_path = 'models/crazyflie_dueling_dqn_best.pth'
            else:
                # Find latest checkpoint
                checkpoints = [f for f in os.listdir('models') 
                             if f.startswith('crazyflie_dueling_dqn_checkpoint_') and f.endswith('.pth')]
                if not checkpoints:
                    print("No Dueling DQN models found!")
                    return
                
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                checkpoint_path = f"models/{latest_checkpoint}"
        
        print(f"Loading model: {checkpoint_path}")
        policy_net = load_trained_model(checkpoint_path)
        if policy_net is None:
            return
        
        print(f"\nTesting Dueling DQN model with {args.episodes} episodes...")
        print("This will analyze value/advantage decomposition across difficulty levels")
        
        # Start testing in a separate thread
        test_thread = threading.Thread(
            target=test_model, 
            args=(policy_net, args.episodes, 500, not args.no_viz), 
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
        
        # Clean shutdown
        test_active = False
        test_thread.join(timeout=5.0)
        
        print("\nTest completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()