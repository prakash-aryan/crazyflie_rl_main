#!/usr/bin/env python3
"""
Test the trained Crazyflie hovering model
"""

import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import mujoco
from mujoco import viewer
import numpy as np
import time
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# Suppress additional warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Global simulation lock
sim_lock = threading.RLock()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global simulation objects
model = None
data = None
viewer_handle = None

# DQN architecture (same as training)
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def initialize_simulation():
    """Initialize simulation with proper memory allocation"""
    global model, data
    with sim_lock:
        try:
            # Try to load scene with DQN-specific settings, fallback to default
            model = mujoco.MjModel.from_xml_path('scene_dueling_dqn.xml')
        except:
            model = mujoco.MjModel.from_xml_path('scene.xml')
        
        data = mujoco.MjData(model)
        
        # Same physics settings as training
        model.opt.timestep = 0.02
        model.opt.iterations = 10
        model.opt.tolerance = 1e-2
        
        reset_simulation()

def reset_simulation(randomize=False):
    """Reset to initial state"""
    global model, data
    with sim_lock:
        try:
            mujoco.mj_resetData(model, data)
            
            if randomize:
                # Test with slight randomization
                data.qpos[0] = np.random.uniform(-0.1, 0.1)    # x
                data.qpos[1] = np.random.uniform(-0.1, 0.1)    # y  
                data.qpos[2] = np.random.uniform(0.9, 1.1)     # z
                
                # Small random orientation
                angle = np.random.uniform(-0.05, 0.05)
                data.qpos[3] = np.cos(angle/2)  # qw
                data.qpos[4] = np.sin(angle/2) * np.random.uniform(-0.05, 0.05)  # qx
                data.qpos[5] = np.sin(angle/2) * np.random.uniform(-0.05, 0.05)  # qy
                data.qpos[6] = 0.0  # qz
            else:
                # Fixed position
                data.qpos[0] = 0.0    # x
                data.qpos[1] = 0.0    # y  
                data.qpos[2] = 1.0    # z - at target height
                data.qpos[3] = 1.0    # qw (quaternion)
                data.qpos[4] = 0.0    # qx
                data.qpos[5] = 0.0    # qy
                data.qpos[6] = 0.0    # qz
            
            # Zero velocities
            data.qvel[:] = 0.0
            
            # Start with hover thrust
            data.ctrl[0] = 0.26
            data.ctrl[1] = 0.0
            data.ctrl[2] = 0.0
            data.ctrl[3] = 0.0
            
            mujoco.mj_forward(model, data)
            
        except Exception as e:
            print(f"Error in reset: {e}")

def get_state():
    """Get current state (same as training)"""
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
                quat = np.array([1, 0, 0, 0])
            
            # Target relative position (3)
            target_pos = np.array([0.0, 0.0, 1.0])
            rel_pos = target_pos - pos
            
            # Build state vector (13 dimensions total)
            state = np.concatenate([
                pos,           # 3: current position
                vel,           # 3: velocity
                quat,          # 4: orientation quaternion
                rel_pos        # 3: relative position to target
            ])
            
            return state.astype(np.float32)
            
        except Exception as e:
            print(f"Error in get_state: {e}")
            return np.zeros(13, dtype=np.float32)

def apply_action(action):
    """Apply action (same as training)"""
    global model, data
    
    with sim_lock:
        try:
            current_thrust = data.ctrl[0]
            
            # Smaller action steps for finer control
            thrust_step = 0.005  # Reduced from 0.01
            moment_step = 0.01   # Reduced from 0.02
            
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
                data.ctrl[1] *= 0.7
                data.ctrl[2] *= 0.7
                data.ctrl[3] *= 0.7
            
            # Stronger moment decay for stability
            data.ctrl[1] = np.clip(data.ctrl[1] * 0.85, -0.05, 0.05)
            data.ctrl[2] = np.clip(data.ctrl[2] * 0.85, -0.05, 0.05)
            data.ctrl[3] = np.clip(data.ctrl[3] * 0.85, -0.05, 0.05)
            
            # Step simulation
            for _ in range(2):
                try:
                    mujoco.mj_step(model, data)
                except Exception as e:
                    print(f"Step error: {e}")
                    reset_simulation()
                    return
                
        except Exception as e:
            print(f"Error in apply_action: {e}")

def load_trained_model(checkpoint_path):
    """Load the trained DQN model"""
    policy_net = DQN(13, 7).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        policy_net.eval()
        
        print(f"Loaded DQN model from episode {checkpoint['episode']}")
        print(f"Model reward: {checkpoint['reward']:.2f}")
        
        return policy_net
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Global test flag
test_active = True

def test_model(policy_net, num_episodes=10):
    """Test the model performance"""
    global test_active
    
    all_episode_stats = []
    
    for episode in range(num_episodes):
        if not test_active:
            break
            
        print(f"\n--- Episode {episode + 1} ---")
        
        # Alternate between fixed and randomized starts
        randomize = episode % 2 == 1
        reset_simulation(randomize=randomize)
        
        state = get_state()
        total_reward = 0
        steps = 0
        max_steps = 500
        
        positions = []
        velocities = []
        actions_taken = []
        
        for step in range(max_steps):
            # Get action from policy (greedy)
            with torch.no_grad():
                state_tensor = torch.tensor([state], device=device, dtype=torch.float32)
                q_values = policy_net(state_tensor)
                action = q_values.max(1)[1].item()
            
            actions_taken.append(action)
            
            # Apply action
            apply_action(action)
            
            # Record data
            pos = data.qpos[0:3].copy()
            vel = data.qvel[0:3].copy()
            positions.append(pos)
            velocities.append(np.linalg.norm(vel))
            
            # Get reward (simple distance-based for testing)
            target = np.array([0.0, 0.0, 1.0])
            distance = np.linalg.norm(pos - target)
            reward = max(0, 10 - distance * 10)  # Simple reward for testing
            total_reward += reward
            
            # Check termination
            if (pos[2] < 0.3 or pos[2] > 3.0 or 
                abs(pos[0]) > 2.0 or abs(pos[1]) > 2.0):
                break
            
            # Get next state
            state = get_state()
            steps += 1
            
            time.sleep(0.02)  # Real-time visualization
        
        # Calculate metrics
        positions = np.array(positions)
        target = np.array([0.0, 0.0, 1.0])
        errors = np.linalg.norm(positions - target, axis=1)
        
        avg_error = np.mean(errors)
        min_error = np.min(errors)
        max_error = np.max(errors)
        time_near_target = np.mean(errors < 0.2)  # Within 20cm
        time_very_close = np.mean(errors < 0.1)   # Within 10cm
        
        # Height stability
        heights = positions[:, 2]
        height_stability = np.std(heights)
        
        # Average velocity
        avg_velocity = np.mean(velocities)
        action_diversity = len(set(actions_taken))
        
        episode_stats = {
            'total_reward': total_reward,
            'steps': steps,
            'avg_error': avg_error,
            'min_error': min_error,
            'time_near_target': time_near_target,
            'time_very_close': time_very_close,
            'height_stability': height_stability,
            'avg_velocity': avg_velocity,
            'action_diversity': action_diversity
        }
        all_episode_stats.append(episode_stats)
        
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Steps survived: {steps}")
        print(f"  Average error: {avg_error:.3f}m (min: {min_error:.3f}m, max: {max_error:.3f}m)")
        print(f"  Time near target (<0.2m): {time_near_target:.1%}")
        print(f"  Time very close (<0.1m): {time_very_close:.1%}")
        print(f"  Height stability (std): {height_stability:.3f}m")
        print(f"  Average velocity: {avg_velocity:.3f}m/s")
        print(f"  Actions used: {action_diversity}/7")
        print(f"  Final position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        
        time.sleep(1.0)  # Pause between episodes
    
    # Overall statistics
    print(f"\n{'='*60}")
    print("OVERALL TEST RESULTS")
    print(f"{'='*60}")
    
    avg_reward = np.mean([s['total_reward'] for s in all_episode_stats])
    avg_steps = np.mean([s['steps'] for s in all_episode_stats])
    avg_error_overall = np.mean([s['avg_error'] for s in all_episode_stats])
    avg_time_near = np.mean([s['time_near_target'] for s in all_episode_stats])
    avg_time_close = np.mean([s['time_very_close'] for s in all_episode_stats])
    
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average position error: {avg_error_overall:.3f}m")
    print(f"Average time near target: {avg_time_near:.1%}")
    print(f"Average time very close: {avg_time_close:.1%}")
    
    test_active = False

def main():
    """Main testing function"""
    global test_active, viewer_handle
    
    print("Crazyflie Trained Model Test")
    print("="*40)
    
    # Initialize simulation
    print("Initializing simulation...")
    initialize_simulation()
    
    # Check if models directory exists
    if not os.path.exists('models'):
        print("No models directory found!")
        return
    
    # Look for best average model first, then best single episode
    best_avg_path = 'models/crazyflie_hover_best_avg.pth'
    best_path = 'models/crazyflie_hover_best.pth'
    
    if os.path.exists(best_avg_path):
        checkpoint_path = best_avg_path
        print("Loading best average model...")
    elif os.path.exists(best_path):
        checkpoint_path = best_path
        print("Loading best single episode model...")
    else:
        # Find the latest checkpoint
        checkpoints = [f for f in os.listdir('models') if f.endswith('.pth') and 'checkpoint' in f]
        if not checkpoints:
            print("No trained models found in models/ directory!")
            return
        
        # Get the latest checkpoint (highest episode number)
        try:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        except:
            # If parsing fails, just take the last one alphabetically
            latest_checkpoint = sorted(checkpoints)[-1]
        
        checkpoint_path = f"models/{latest_checkpoint}"
        print(f"Loading latest checkpoint: {latest_checkpoint}")
    
    # Load trained model
    policy_net = load_trained_model(checkpoint_path)
    if policy_net is None:
        return
    
    print("\nStarting evaluation...")
    print("Testing with 10 episodes (alternating fixed/randomized conditions)")
    print("Close the viewer window to end the test.")
    
    # Start testing in a separate thread
    test_thread = threading.Thread(target=test_model, args=(policy_net, 10), daemon=True)
    test_thread.start()
    
    try:
        # Launch viewer in main thread
        with mujoco.viewer.launch_passive(model, data) as viewer_handle:
            while viewer_handle.is_running() and test_active:
                with sim_lock:
                    viewer_handle.sync()
                time.sleep(0.01)
        
        # Clean shutdown
        test_active = False
        test_thread.join(timeout=5.0)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()