#!/usr/bin/env python3
"""
Test script for REDQ-trained Crazyflie model
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

# REDQ Network Components (same as training)
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

def initialize_simulation():
    """Initialize MuJoCo simulation"""
    global model, data
    
    with sim_lock:
        try:
            model = mujoco.MjModel.from_xml_path('scene_redq.xml')
        except:
            model = mujoco.MjModel.from_xml_path('scene.xml')
        
        data = mujoco.MjData(model)
        model.opt.timestep = 0.005
        model.opt.iterations = 50
        model.opt.tolerance = 1e-4
        
        reset_simulation()

def reset_simulation(randomize=False, difficulty=0.5):
    """Reset simulation with optional randomization"""
    global model, data
    
    with sim_lock:
        try:
            mujoco.mj_resetData(model, data)
            
            if randomize:
                pos_range = 0.02 + 0.18 * difficulty
                vel_range = 0.02 + 0.08 * difficulty
                angle_range = 0.01 + 0.09 * difficulty
                
                data.qpos[0] = np.random.normal(0, pos_range/2)
                data.qpos[1] = np.random.normal(0, pos_range/2)
                data.qpos[2] = 1.0 + np.random.uniform(-0.05, 0.05)
                
                angle = np.random.uniform(-angle_range, angle_range)
                axis = np.random.randn(3)
                axis = axis / (np.linalg.norm(axis) + 1e-8)
                
                data.qpos[3] = np.cos(angle/2)
                data.qpos[4] = np.sin(angle/2) * axis[0]
                data.qpos[5] = np.sin(angle/2) * axis[1]
                data.qpos[6] = np.sin(angle/2) * axis[2]
                
                data.qvel[:3] = np.random.normal(0, vel_range/3, size=3)
                data.qvel[3:6] = np.random.normal(0, vel_range/5, size=3)
            else:
                data.qpos[:3] = [0.0, 0.0, 1.0]
                data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                data.qvel[:] = 0.0
            
            data.ctrl[0] = 0.26
            data.ctrl[1:] = 0.0
            
            mujoco.mj_forward(model, data)
            
        except Exception as e:
            print(f"Error in reset: {e}")

def get_observation():
    """Get observation from simulation (same as training)"""
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
    """Load trained REDQ model"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get ensemble size from checkpoint
        num_q = checkpoint.get('num_q', 10)
        utd_ratio = checkpoint.get('utd_ratio', 20)
        
        # Create networks
        actor = Actor(state_dim=18, action_dim=4, hidden_dim=256).to(device)
        q_ensemble = QEnsemble(state_dim=18, action_dim=4, num_q=num_q, hidden_dim=256).to(device)
        
        # Load state dicts
        actor.load_state_dict(checkpoint['actor_state_dict'])
        q_ensemble.load_state_dict(checkpoint['q_ensemble_state_dict'])
        
        actor.eval()
        q_ensemble.eval()
        
        print(f"Loaded REDQ model from episode {checkpoint['episode']}")
        print(f"Training reward: {checkpoint['reward']:.2f}")
        print(f"Ensemble size: {num_q} Q-networks")
        print(f"UTD ratio: {utd_ratio}")
        if 'difficulty' in checkpoint:
            print(f"Training difficulty: {checkpoint['difficulty']}")
        if 'total_steps' in checkpoint:
            print(f"Total training steps: {checkpoint['total_steps']}")
        
        return actor, q_ensemble, num_q
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

# Global test flag
test_active = True

def test_model(actor, q_ensemble, num_q, num_episodes=20, max_steps=1000, visualize=True):
    """Comprehensive model testing"""
    global test_active
    
    print(f"\n{'='*70}")
    print("REDQ MODEL EVALUATION")
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
            q_values_history = []
            
            for step in range(max_steps):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    
                    # Use deterministic action for evaluation
                    _, _, action = actor.sample(state_tensor)
                    action_np = action.cpu().numpy()[0]
                    
                    # Get Q-values for analysis
                    action_tensor = torch.FloatTensor(action_np).unsqueeze(0).to(device)
                    q_values = q_ensemble(state_tensor, action_tensor, return_all=True)
                    q_mean = q_values.mean().item()
                    q_std = q_values.std().item()
                    q_values_history.append((q_mean, q_std))
                
                actions.append(action_np)
                apply_continuous_action(action_np)
                
                # Record data
                pos = data.qpos[0:3].copy()
                vel = data.qvel[0:3].copy()
                positions.append(pos)
                velocities.append(np.linalg.norm(vel))
                thrust_values.append(data.ctrl[0])
                
                # Calculate reward (same as training)
                target = np.array([0.0, 0.0, 1.0])
                position_error = np.linalg.norm(pos - target)
                height_error = abs(pos[2] - 1.0)
                xy_error = np.linalg.norm(pos[0:2])
                linear_velocity = np.linalg.norm(vel)
                
                angvel = data.qvel[3:6]
                angular_velocity = np.linalg.norm(angvel)
                quat = data.qpos[3:7]
                w, x, y, z = quat
                orientation_error = np.sqrt(x**2 + y**2) * 2.0
                
                # Dense reward calculation (same as training)
                reward = 3.0  # Alive bonus
                reward += np.exp(-2.0 * position_error) * 20.0
                reward += np.exp(-5.0 * height_error) * 15.0
                reward -= linear_velocity * 2.0
                reward -= angular_velocity * 1.5
                reward -= orientation_error * 3.0
                
                if position_error < 0.1 and linear_velocity < 0.1:
                    reward += 25.0
                elif position_error < 0.2 and linear_velocity < 0.2:
                    reward += 10.0
                
                thrust_effort = abs(data.ctrl[0] - 0.26) * 2.0
                moment_effort = np.linalg.norm(data.ctrl[1:]) * 5.0
                reward -= thrust_effort + moment_effort
                
                if pos[2] < 0.3 or pos[2] > 2.0:
                    reward = -50.0
                elif xy_error > 1.0:
                    reward = -30.0
                
                episode_reward += reward
                state = get_observation()
                
                # Check termination
                if pos[2] < 0.3 or pos[2] > 2.0 or abs(pos[0]) > 1.0 or abs(pos[1]) > 1.0:
                    break
                if np.linalg.norm(vel) > 5.0:
                    break
                
                if visualize:
                    time.sleep(0.01)  # Real-time visualization
            
            # Episode analysis
            positions = np.array(positions)
            velocities = np.array(velocities)
            actions = np.array(actions)
            thrust_values = np.array(thrust_values)
            
            # Calculate metrics
            errors = [np.linalg.norm(pos - np.array([0, 0, 1])) for pos in positions]
            avg_error = np.mean(errors)
            time_precise = np.mean([e < 0.1 for e in errors])  # Time within 10cm
            time_good = np.mean([e < 0.2 for e in errors])     # Time within 20cm
            avg_velocity = np.mean(velocities)
            action_smoothness = np.mean(np.std(actions, axis=0))
            thrust_stability = np.std(thrust_values)
            
            # REDQ-specific metrics
            q_means = [q[0] for q in q_values_history]
            q_stds = [q[1] for q in q_values_history]
            ensemble_uncertainty = np.mean(q_stds)  # Average ensemble disagreement
            q_value_trend = np.mean(q_means)
            action_magnitude = np.mean(np.linalg.norm(actions, axis=1))
            
            config_results.append({
                'reward': episode_reward,
                'steps': len(positions),
                'avg_error': avg_error,
                'time_precise': time_precise,
                'time_good': time_good,
                'avg_velocity': avg_velocity,
                'action_smoothness': action_smoothness,
                'thrust_stability': thrust_stability,
                'ensemble_uncertainty': ensemble_uncertainty,
                'q_value_trend': q_value_trend,
                'action_magnitude': action_magnitude
            })
            
            print(f"Episode {episode+1}: Reward={episode_reward:.1f}, "
                  f"Error={avg_error:.3f}m, Precise={time_precise:.1%}, "
                  f"Q_std={ensemble_uncertainty:.2f}")
        
        # Aggregate results for this configuration
        if config_results:
            metrics = ['reward', 'steps', 'avg_error', 'time_precise', 'time_good', 
                      'avg_velocity', 'action_smoothness', 'thrust_stability',
                      'ensemble_uncertainty', 'q_value_trend', 'action_magnitude']
            
            for metric in metrics:
                values = [r[metric] for r in config_results]
                all_results[config['name']].append({
                    'metric': metric,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                })
    
    # Print summary
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    
    for config_name, results in all_results.items():
        print(f"\n{config_name}:")
        for r in results:
            if r['metric'] in ['reward', 'avg_error', 'time_precise', 'ensemble_uncertainty']:
                print(f"  {r['metric']:20s}: {r['mean']:8.3f} ± {r['std']:6.3f}")
    
    # REDQ-specific analysis
    print(f"\n{'='*70}")
    print("REDQ ENSEMBLE ANALYSIS")
    print(f"{'='*70}")
    
    for config_name, results in all_results.items():
        uncertainty = [r for r in results if r['metric'] == 'ensemble_uncertainty'][0]
        q_trend = [r for r in results if r['metric'] == 'q_value_trend'][0]
        
        print(f"\n{config_name}:")
        print(f"  Ensemble Uncertainty: {uncertainty['mean']:8.3f} ± {uncertainty['std']:6.3f}")
        print(f"  Q-Value Trend:        {q_trend['mean']:8.3f} ± {q_trend['std']:6.3f}")
        
        # Analysis
        if uncertainty['mean'] < 5.0:
            print(f"  → Low uncertainty: Ensemble agrees on Q-values")
        elif uncertainty['mean'] > 15.0:
            print(f"  → High uncertainty: Ensemble shows disagreement")
        
        if q_trend['mean'] > 100:
            print(f"  → High Q-values indicate confident policy")
        elif q_trend['mean'] < 50:
            print(f"  → Lower Q-values suggest conservative estimates")
    
    # Overall assessment
    easy_precision = all_results["Easy (Fixed)"][3]['mean'] if "Easy (Fixed)" in all_results else 0  # time_precise
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
    
    # REDQ-specific assessment
    if len(all_results["Easy (Fixed)"]) > 0:
        uncertainty = [r for r in all_results["Easy (Fixed)"] if r['metric'] == 'ensemble_uncertainty'][0]
        if uncertainty['mean'] < 5.0:
            print("CONFIDENT: Low ensemble uncertainty indicates reliable Q-estimates")
        elif uncertainty['mean'] < 10.0:
            print("MODERATE: Reasonable ensemble uncertainty")
        else:
            print("UNCERTAIN: High ensemble disagreement")
    
    # Sample efficiency assessment (REDQ's strength)
    action_magnitude = [r for r in all_results["Easy (Fixed)"] if r['metric'] == 'action_magnitude'][0]
    if action_magnitude['mean'] < 0.3:
        print("EFFICIENT: Conservative, sample-efficient control")
    elif action_magnitude['mean'] < 0.6:
        print("MODERATE: Reasonable control effort")
    else:
        print("AGGRESSIVE: High action magnitudes")
    
    print(f"ENSEMBLE: Uses {num_q} Q-networks for reduced overestimation bias")
    print("SAMPLE EFFICIENT: REDQ's high UTD ratio enables faster learning")
    
    test_active = False
    return all_results

def main():
    """Main test function"""
    global test_active, viewer_handle
    
    parser = argparse.ArgumentParser(description='Test REDQ Crazyflie Model')
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
            # Look for best REDQ model first
            if os.path.exists('models/crazyflie_redq_best.pth'):
                checkpoint_path = 'models/crazyflie_redq_best.pth'
            else:
                # Find latest REDQ checkpoint
                redq_checkpoints = [f for f in os.listdir('models') 
                                  if f.startswith('crazyflie_redq_checkpoint_') and f.endswith('.pth')]
                if redq_checkpoints:
                    latest = max(redq_checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    checkpoint_path = f"models/{latest}"
                else:
                    print("No REDQ models found!")
                    return
        
        print(f"Loading model: {checkpoint_path}")
        actor, q_ensemble, num_q = load_model(checkpoint_path)
        if actor is None or q_ensemble is None:
            return
        
        print(f"\nTesting REDQ model with {args.episodes} episodes...")
        print("This will test across multiple difficulty levels")
        print("REDQ uses ensemble Q-learning with reduced overestimation bias")
        
        # Start test in separate thread
        test_thread = threading.Thread(
            target=test_model, 
            args=(actor, q_ensemble, num_q, args.episodes, 1000, not args.no_viz),
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