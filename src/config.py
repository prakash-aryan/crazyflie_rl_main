#!/usr/bin/env python3
"""
Configuration file for Crazyflie RL training
"""

class Config:
    # Environment settings
    MODEL_PATH = 'scene.xml'
    DESIRED_HOVER_HEIGHT = 1.0
    MAX_EPISODE_STEPS = 1000
    
    # Target position for hovering (x, y, z)
    TARGET_POSITION = [0.0, 0.0, 1.0]
    
    # Physics bounds
    MAX_HEIGHT = 3.0
    MAX_HORIZONTAL_DISTANCE = 2.0
    MIN_HEIGHT = 0.05
    
    # Control parameters
    THRUST_RANGE = [0.0, 0.35]  # From cf2.xml actuator limits
    MOMENT_RANGE = [-1.0, 1.0]
    CONTROL_STEP_SIZE = 0.05
    
    # RL Hyperparameters
    BATCH_SIZE = 128
    GAMMA = 0.99  # Discount factor
    EPS_START = 0.9  # Starting epsilon for epsilon-greedy
    EPS_END = 0.05   # Ending epsilon
    EPS_DECAY = 1000 # Epsilon decay rate
    TAU = 0.005      # Soft update parameter for target network
    LEARNING_RATE = 1e-4
    
    # Network architecture
    HIDDEN_LAYERS = [256, 256, 128]
    DROPOUT_RATE = 0.1
    
    # Memory
    REPLAY_MEMORY_SIZE = 10000
    
    # Training
    NUM_EPISODES = 1000
    LOG_INTERVAL = 50  # Episodes between logging
    SAVE_INTERVAL = 100  # Episodes between model saves
    
    # Reward function weights
    REWARD_WEIGHTS = {
        'base': 10.0,              # Base reward for staying airborne
        'position_error': 5.0,     # Penalty for distance from target
        'velocity': 2.0,           # Penalty for excessive velocity
        'tilt': 3.0,              # Penalty for tilting
        'angular_velocity': 1.0,   # Penalty for spinning
        'target_bonus': 20.0,      # Bonus for being close to target
        'crash_penalty': 100.0,    # Penalty for crashing
        'target_threshold': 0.1    # Distance threshold for target bonus
    }
    
    # Logging
    LOG_DIR = 'logs'
    MODEL_DIR = 'models'
    
    # Simulation
    SIMULATION_TIMESTEP = 0.01
    RENDER_FPS = 60

# Different configurations for different tasks
class HoverConfig(Config):
    """Configuration optimized for hovering task"""
    TARGET_POSITION = [0.0, 0.0, 1.0]
    MAX_EPISODE_STEPS = 2000
    
    REWARD_WEIGHTS = {
        'base': 15.0,
        'position_error': 8.0,
        'velocity': 3.0,
        'tilt': 5.0,
        'angular_velocity': 2.0,
        'target_bonus': 30.0,
        'crash_penalty': 150.0,
        'target_threshold': 0.1
    }

class NavigationConfig(Config):
    """Configuration for point-to-point navigation (future use)"""
    TARGET_POSITION = [1.0, 1.0, 1.0]  # Example target
    MAX_EPISODE_STEPS = 3000
    
    REWARD_WEIGHTS = {
        'base': 5.0,
        'position_error': 10.0,
        'velocity': 1.0,
        'tilt': 2.0,
        'angular_velocity': 1.0,
        'target_bonus': 50.0,
        'crash_penalty': 100.0,
        'target_threshold': 0.2
    }

# Global config - change this to switch between configurations
CURRENT_CONFIG = HoverConfig()