# 🚁 Crazyflie Reinforcement Learning Project

[Screencast from 05-28-2025 03:28:21 PM.webm](https://github.com/user-attachments/assets/63ecfeb7-8ff0-4766-8ab6-6d1c3a7abc73)


## Comparative Analysis of Deep Reinforcement Learning Algorithms for Autonomous UAV Navigation in MuJoCo Simulation Environment

This repository contains the implementation code for my **Master's Dissertation** at BITS Pilani Dubai Campus, supervised by Dr. Sujala D. Shetty. The project evaluates and compares seven state-of-the-art deep reinforcement learning algorithms for autonomous quadcopter hovering control using the Bitcraze Crazyflie 2.0 model in MuJoCo physics simulation.

## 📋 Abstract

This research investigates the performance of seven distinct deep reinforcement learning algorithms applied to UAV navigation and hovering tasks within the MuJoCo physics simulation environment. The study implements and evaluates **DQN**, **Dueling DQN**, **PPO**, **SAC**, **TD3**, **REDQ**, and **Dreamer** algorithms for autonomous quadcopter control, focusing on achieving stable hovering while maintaining precise position control in three-dimensional space.

## 🎯 Key Findings

- **SAC** achieves exceptional sample efficiency with convergence in only 32 training episodes
- **PPO** delivers superior position accuracy with 53.3% of flight time within 0.1 meters of target
- **TD3** attains the highest cumulative rewards but requires extensive training investment
- **REDQ** provides balanced performance across multiple evaluation criteria
- Continuous action algorithms generally outperform discrete action methods for fine-grained positioning tasks

## 🛠️ System Requirements

### Hardware
- **GPU**: NVIDIA GeForce RTX 4070 or similar (CUDA-compatible)
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 2GB free space for models and logs

### Software
- **OS**: Ubuntu 22.04 LTS (tested only on Ubuntu)
- **Python**: 3.10 or higher
- **CUDA**: 12.8 (or compatible version with your GPU)
- **UV**: Package manager for Python dependencies

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/prakash-aryan/crazyflie_rl_main.git
cd crazyflie_rl_main
```

### 2. Install UV Package Manager

If you don't have UV installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart your shell or run:
source ~/.bashrc
```

### 3. Create Virtual Environment and Install Dependencies

```bash
# Create virtual environment with UV
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies using UV
uv pip install -r requirements.txt
```

### 4. Verify Installation

Check if CUDA is available and MuJoCo is properly installed:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import mujoco; print('MuJoCo installed successfully')"
```

## 📁 Project Structure

```
crazyflie_rl_main/
├── src/                          # Source code directory
│   ├── config.py                 # Configuration settings
│   ├── train_*.py               # Training scripts for each algorithm
│   │   ├── train_dqn.py
│   │   ├── train_dueling_dqn.py
│   │   ├── train_ppo.py
│   │   ├── train_sac.py
│   │   ├── train_td3.py
│   │   ├── train_redq.py
│   │   └── train_dreamer.py
│   └── test_*.py                # Testing scripts for each algorithm
│       ├── test_dqn.py
│       ├── test_dueling_dqn.py
│       ├── test_ppo.py
│       ├── test_sac.py
│       ├── test_td3.py
│       ├── test_rdq.py
│       └── test_dreamer.py
├── models/                      # Pre-trained model checkpoints
├── assets/                      # Additional assets
├── scene_*.xml                  # MuJoCo scene files for each algorithm
├── cf2.xml                      # Crazyflie 2.0 model definition
├── cf2.png                      # Crazyflie 2.0 texture
├── comparison_graphs.py         # Performance analysis and visualization
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🎮 Usage

### Training Models

Each algorithm has its dedicated training script. To train a specific algorithm:

```bash
# Activate virtual environment
source .venv/bin/activate

# Train Deep Q-Network
python src/train_dqn.py

# Train Dueling DQN
python src/train_dueling_dqn.py

# Train Proximal Policy Optimization
python src/train_ppo.py

# Train Soft Actor-Critic
python src/train_sac.py

# Train Twin Delayed DDPG
python src/train_td3.py

# Train Randomized Ensembled Double Q-learning
python src/train_redq.py

# Train Dreamer
python src/train_dreamer.py
```

### Testing Pre-trained Models

To evaluate trained models:

```bash
# Test any algorithm (replace with desired algorithm)
python src/test_sac.py     # Test SAC model
python src/test_ppo.py     # Test PPO model
python src/test_td3.py     # Test TD3 model
# ... and so on
```




### Model Naming Convention
```
crazyflie_{algorithm}_{type}_{episode}.pth

Examples:
- crazyflie_sac_best.pth           # Best SAC model
- crazyflie_ppo_checkpoint_1000.pth # PPO checkpoint at episode 1000
- crazyflie_td3_best_avg.pth       # TD3 model with best average performance
```

## 🔧 Configuration

Algorithm hyperparameters and training settings can be modified in `src/config.py`. Key parameters include:

- **Learning rates**: Different for each algorithm
- **Network architectures**: Hidden layer sizes and activation functions
- **Training episodes**: Total number of training episodes
- **Evaluation frequency**: How often to evaluate and save models
- **Curriculum learning**: Progressive difficulty settings

## 📊 Performance Metrics

The project evaluates algorithms based on:

1. **Training Efficiency**: Episodes required for convergence
2. **Hovering Precision**: Distance from target position
3. **Flight Stability**: Consistency of control
4. **Sample Efficiency**: Learning speed
5. **Computational Requirements**: Training time and resource usage

## 🎯 Scene Files

Each algorithm uses a specific MuJoCo scene file optimized for its characteristics:

- `scene_dqn.xml` - Deep Q-Network environment
- `scene_dueling_dqn.xml` - Dueling DQN environment  
- `scene_ppo.xml` - PPO environment
- `scene_sac.xml` - SAC environment
- `scene_td3.xml` - TD3 environment
- `scene_redq.xml` - REDQ environment
- `scene_dreamer.xml` - Dreamer environment

## 📈 Results Summary

Based on extensive experimentation:

| Algorithm    | Sample Efficiency | Position Accuracy | Training Time | Best Use Case |
|--------------|------------------|-------------------|---------------|---------------|
| **SAC**      | ⭐⭐⭐⭐⭐         | ⭐⭐⭐⭐           | ⭐⭐⭐⭐⭐      | Quick deployment |
| **PPO**      | ⭐⭐⭐            | ⭐⭐⭐⭐⭐         | ⭐⭐⭐⭐        | Precise control |
| **TD3**      | ⭐⭐⭐            | ⭐⭐⭐⭐           | ⭐⭐           | Maximum performance |
| **REDQ**     | ⭐⭐⭐⭐           | ⭐⭐⭐⭐           | ⭐⭐⭐         | Balanced approach |
| **DQN**      | ⭐⭐              | ⭐⭐⭐             | ⭐⭐⭐         | Simple discrete control |
| **Dueling DQN** | ⭐⭐⭐         | ⭐⭐⭐             | ⭐⭐⭐         | Improved DQN |
| **Dreamer**  | ⭐⭐              | ⭐⭐               | ⭐⭐           | Research/exploration |

## 🔍 Troubleshooting

### Common Issues

1. **CUDA not available**
   ```bash
   # Check NVIDIA driver
   nvidia-smi
   
   # Reinstall PyTorch with CUDA support
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **MuJoCo installation issues**
   ```bash
   # Install MuJoCo dependencies
   sudo apt-get update
   sudo apt-get install libglfw3-dev libgl1-mesa-dev libxinerama-dev libxcursor-dev libxi-dev
   
   # Reinstall MuJoCo
   uv pip install mujoco>=3.0.0
   ```

3. **Virtual environment activation**
   ```bash
   # Make sure you're in the project directory
   cd crazyflie_rl
   source .venv/bin/activate
   ```

## 📚 Dependencies

Core dependencies as specified in `requirements.txt`:

- `torch>=2.0.0` - Deep learning framework
- `numpy>=1.21.0` - Numerical computations
- `mujoco>=3.0.0` - Physics simulation
- `tensorboard>=2.10.0` - Training visualization
- `matplotlib>=3.5.0` - Plotting and visualization
- `scipy>=1.9.0` - Scientific computing
- `gymnasium>=0.26.0` - RL environment interface
- `opencv-python>=4.5.0` - Computer vision utilities

## 📖 Citation

If you use this code in your research, please cite:
(todo)

<!-- ```bibtex
@mastersthesis{aryan2025comparative,
  title={Comparative Analysis of Deep Reinforcement Learning Algorithms for Autonomous UAV Navigation in MuJoCo Simulation Environment},
  author={Aryan, Prakash},
  year={2025},
  school={BITS Pilani Dubai Campus},
  supervisor={Dr. Sujala D. Shetty},
  type={Master's Dissertation}
}
``` -->

## 👨‍💻 Author

**Prakash Aryan**  
Master's Student, BITS Pilani Dubai Campus  
ID: 2023H1120010U  
📧 [Contact](https://github.com/prakash-aryan)

**Supervisor**: Dr. Sujala D. Shetty, Associate Professor, BITS Pilani Dubai Campus

## 🙏 Acknowledgments

Special thanks to:
- **Dr. Sujala D. Shetty** for supervision and guidance
- **Dr. Sebastiano Panichella** and **Prof. Dr. Timo Kehrer** from University of Bern
- **BITS Pilani Dubai Campus** for providing research facilities
- The open-source community for the tools and libraries used in this project

## 📝 Research Context

This implementation serves as the practical foundation for the dissertation **"Comparative Analysis of Deep Reinforcement Learning Algorithms for Autonomous UAV Navigation in MuJoCo Simulation Environment"** submitted to BITS Pilani Dubai Campus in June 2025. The research contributes empirical insights into deep reinforcement learning for UAV navigation, providing practical guidance for algorithm selection in autonomous aerial vehicle applications.

---

*This project is part of academic research and is intended for educational and research purposes. The code and methodologies have been developed as part of a Master's dissertation course.*
