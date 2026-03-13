# Crazyflie Reinforcement Learning

[Screencast from 05-28-2025 03:28:21 PM.webm](https://github.com/user-attachments/assets/63ecfeb7-8ff0-4766-8ab6-6d1c3a7abc73)


## Comparative Analysis of Deep RL Algorithms for Autonomous UAV Navigation in MuJoCo

Implementation code for my Master's Dissertation. The project compares seven deep reinforcement learning algorithms for quadcopter hovering control using the Bitcraze Crazyflie 2.0 model in MuJoCo.

## Abstract

This work evaluates seven deep reinforcement learning algorithms — **DQN**, **Dueling DQN**, **PPO**, **SAC**, **TD3**, **REDQ**, and **Dreamer** — for autonomous quadcopter hovering and position control in the MuJoCo physics simulation environment.

## Key Findings

- **SAC** converges in only 32 training episodes (best sample efficiency)
- **PPO** keeps 53.3% of flight time within 0.1m of target (best position accuracy)
- **TD3** reaches the highest cumulative rewards but needs more training time
- **REDQ** performs well across all evaluation criteria
- Continuous action algorithms outperform discrete ones for fine-grained positioning

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX 4070 or similar (CUDA-compatible)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space for models and logs

### Software
- **OS**: Ubuntu 22.04 LTS (only tested on Ubuntu)
- **Python**: 3.10+
- **CUDA**: 12.8 (or compatible version)
- **UV**: Python package manager

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/prakash-aryan/crazyflie_rl.git
cd crazyflie_rl
```

### 2. Install UV Package Manager

If you don't have UV installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 3. Create Virtual Environment and Install Dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import mujoco; print('MuJoCo installed successfully')"
```

## Project Structure

```
crazyflie_rl/
├── src/                          # Source code
│   ├── config.py                 # Configuration settings
│   ├── train_*.py                # Training scripts for each algorithm
│   │   ├── train_dqn.py
│   │   ├── train_dueling_dqn.py
│   │   ├── train_ppo.py
│   │   ├── train_sac.py
│   │   ├── train_td3.py
│   │   ├── train_redq.py
│   │   └── train_dreamer.py
│   └── test_*.py                 # Testing scripts for each algorithm
│       ├── test_dqn.py
│       ├── test_dueling_dqn.py
│       ├── test_ppo.py
│       ├── test_sac.py
│       ├── test_td3.py
│       ├── test_rdq.py
│       └── test_dreamer.py
├── models/                       # Pre-trained model checkpoints
├── assets/                       # Additional assets
├── scene_*.xml                   # MuJoCo scene files for each algorithm
├── cf2.xml                       # Crazyflie 2.0 model definition
├── cf2.png                       # Crazyflie 2.0 texture
├── comparison_graphs.py          # Performance analysis and visualization
├── requirements.txt              # Python dependencies
└── README.md
```

## Usage

### Training

Each algorithm has a dedicated training script. By default, a MuJoCo viewer window opens to visualize training in real time. Use `--headless` to train without a display (useful for SSH, servers, or CI):

```bash
source .venv/bin/activate

# With GUI (default)
python src/train_sac.py

# Headless (no display required)
python src/train_sac.py --headless
```

Available training scripts:

```bash
python src/train_dqn.py           # DQN
python src/train_dueling_dqn.py   # Dueling DQN
python src/train_ppo.py           # PPO
python src/train_sac.py           # SAC
python src/train_td3.py           # TD3
python src/train_redq.py          # REDQ
python src/train_dreamer.py       # Dreamer
```

### Testing

Use `--no-viz` to run tests without the viewer:

```bash
# With GUI (default)
python src/test_sac.py

# Headless
python src/test_sac.py --no-viz

# Specify number of test episodes
python src/test_sac.py --no-viz --episodes 50
```

### Model Naming Convention
```
crazyflie_{algorithm}_{type}_{episode}.pth

Examples:
- crazyflie_sac_best.pth
- crazyflie_ppo_checkpoint_1000.pth
- crazyflie_td3_best_avg.pth
```

## Configuration

Hyperparameters and training settings are in `src/config.py`:

- Learning rates (per algorithm)
- Network architectures
- Training episodes
- Evaluation frequency
- Curriculum learning settings

## Performance Metrics

Algorithms are evaluated on:

1. **Training Efficiency** — episodes to convergence
2. **Hovering Precision** — distance from target position
3. **Flight Stability** — consistency of control
4. **Sample Efficiency** — learning speed
5. **Computational Cost** — training time and resources

## Scene Files

Each algorithm uses a MuJoCo scene file tuned for its characteristics:

- `scene_dqn.xml`, `scene_dueling_dqn.xml`, `scene_ppo.xml`, `scene_sac.xml`, `scene_td3.xml`, `scene_redq.xml`, `scene_dreamer.xml`

## Results

| Algorithm    | Sample Efficiency | Position Accuracy | Training Time | Best Use Case |
|--------------|:-:|:-:|:-:|---------------|
| **SAC**      | High | High | Low | Quick deployment |
| **PPO**      | Medium | Highest | Medium | Precise control |
| **TD3**      | Medium | High | High | Maximum performance |
| **REDQ**     | High | High | Medium | Balanced approach |
| **DQN**      | Low | Medium | Medium | Simple discrete control |
| **Dueling DQN** | Medium | Medium | Medium | Improved DQN |
| **Dreamer**  | Low | Low | High | Research / exploration |

## Troubleshooting

**CUDA not available**
```bash
nvidia-smi
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**MuJoCo installation issues**
```bash
sudo apt-get update
sudo apt-get install libglfw3-dev libgl1-mesa-dev libxinerama-dev libxcursor-dev libxi-dev
uv pip install mujoco>=3.0.0
```

**Virtual environment activation**
```bash
cd crazyflie_rl
source .venv/bin/activate
```

## Dependencies

- `torch>=2.0.0` — deep learning framework
- `numpy>=1.21.0` — numerical computing
- `mujoco>=3.0.0` — physics simulation
- `tensorboard>=2.10.0` — training visualization
- `matplotlib>=3.5.0` — plotting
- `scipy>=1.9.0` — scientific computing
- `gymnasium>=0.26.0` — RL environment interface
- `opencv-python>=4.5.0` — computer vision

## Publication

**Aryan, Prakash**, *Comparative Analysis of Deep Reinforcement Learning Algorithms for Autonomous UAV Navigation in MuJoCo Simulation Environment Dissertation* (June 03, 2025). Available at SSRN: [https://ssrn.com/abstract=5398703](https://ssrn.com/abstract=5398703) or [http://dx.doi.org/10.2139/ssrn.5398703](http://dx.doi.org/10.2139/ssrn.5398703)

## Citation

```bibtex
@mastersthesis{aryan2025comparative,
  title={Comparative Analysis of Deep Reinforcement Learning Algorithms for Autonomous UAV Navigation in MuJoCo Simulation Environment},
  author={Aryan, Prakash},
  year={2025},
  school={BITS Pilani Dubai Campus},
  supervisor={Dr. Sujala D. Shetty},
  type={Master's Dissertation},
  note={Available at SSRN: https://ssrn.com/abstract=5398703}
}
```

## Author

**Prakash Aryan**
Master's Student, BITS Pilani Dubai Campus
ID: 2023H1120010U
[GitHub](https://github.com/prakash-aryan)

**Supervisor**: Dr. Sujala D. Shetty, Associate Professor, BITS Pilani Dubai Campus

## Acknowledgments

- **Dr. Sujala D. Shetty** for supervision and guidance
- **Dr. Sebastiano Panichella** and **Prof. Dr. Timo Kehrer** from University of Bern
- **BITS Pilani Dubai Campus** for research facilities

## Research Context

This code is the practical foundation for the dissertation "Comparative Analysis of Deep Reinforcement Learning Algorithms for Autonomous UAV Navigation in MuJoCo Simulation Environment" submitted to BITS Pilani Dubai Campus in June 2025.

---

*Academic research project developed as part of a Master's dissertation.*
