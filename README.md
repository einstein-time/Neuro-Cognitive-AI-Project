# Cognitive Load Theory & Attention Mechanisms: Predicting Human Learning with Deep Learning, Validated Through Reinforcement Learning

## Overview

This project bridges cognitive science and artificial intelligence by modeling human cognitive processes during learning tasks. The system uses **Deep Learning** (Transformers with attention mechanisms) to predict human performance based on Cognitive Load Theory, and then uses **Reinforcement Learning** agents to demonstrate that predicted patterns emerge naturally when agents learn under similar constraints.

## Central Hypothesis

**If RL agents with human-like cognitive constraints (limited memory, attention bottlenecks, forgetting curves) naturally develop the same learning patterns as humans, this validates that cognitive constraints CAUSE these patterns, rather than merely correlating with them.**

## Theoretical Foundation

This project is grounded in established cognitive psychology research:

- **Cognitive Load Theory** (Sweller, 1988): Working memory limitations affect learning
- **Attention Theory** (Broadbent, 1958; Treisman, 1969): Selective attention as information bottleneck
- **Ebbinghaus Forgetting Curve** (1885): Memory decay follows exponential function R(t) = exp(-λt)
- **Power Law of Learning** (Newell & Rosenbloom, 1981): Performance improves as power function of practice
- **Cowan's Capacity Model** (2001): Working memory capacity approximately 4 items
- **Spacing Effect**: Distributed practice superior to massed practice

## Project Architecture

### Phase 1: Deep Learning - Predictive Modeling

Build a Transformer-based model that predicts human performance on cognitive tasks.

**Input**: Task parameters (sequence length, study time, retention interval, practice repetitions)

**Output**: Predicted recall accuracy, reaction time, confidence

**Novel Components**:
- Custom positional encoding with temporal decay (models forgetting)
- Multi-task prediction heads
- Interpretable attention weights (cognitive resource allocation)

### Phase 2: Reinforcement Learning - Mechanistic Validation

Build RL agents with cognitive constraints and observe emergent learning patterns.

**Environment**: Sequential memory task with constraints:
- Limited working memory capacity (5 items)
- Attention bottleneck (1 item per timestep)
- Exponential forgetting (memory decay over time)
- Interference (new memories displace old ones)

**Agent**: PPO-based policy learning optimal attention allocation and memory management strategies

**Hypothesis Test**: Do constrained RL agents naturally exhibit human-like patterns?

### Phase 3: Validation

Comprehensive statistical comparison:
1. Learning curves (power law fit)
2. Forgetting curves (exponential decay)
3. Spacing effects (distributed vs. massed practice)
4. Capacity limits (Miller's 7±2)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/cognitive-dl-rl.git
cd cognitive-dl-rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start: Complete Pipeline

Run all three phases sequentially:

```bash
# Phase 1: Train Deep Learning model
python cognitive_dl_rl/train_dl.py --epochs 50 --batch_size 32

# Phase 2: Train RL agent
python cognitive_dl_rl/train_rl.py --episodes 10000 --curriculum

# Phase 3: Validate hypothesis
python cognitive_dl_rl/validate_hypothesis.py
```

### Individual Phase Execution

#### Phase 1: Deep Learning Training

```bash
python cognitive_dl_rl/train_dl.py \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --seed 42 \
    --checkpoint_dir checkpoints \
    --results_dir results
```

**Expected Output**:
- Trained Cognitive Load Transformer
- Validation correlation > 0.8
- Visualizations of training progress
- Model checkpoint at `checkpoints/best_model.pth`

#### Phase 2: RL Agent Training

```bash
python cognitive_dl_rl/train_rl.py \
    --episodes 10000 \
    --update_freq 2048 \
    --seed 42 \
    --curriculum \
    --checkpoint_dir checkpoints_rl \
    --results_dir results
```

**Expected Output**:
- Trained RL agent with cognitive constraints
- Learning curves showing improvement over episodes
- Final accuracy > 0.7
- Agent checkpoint at `checkpoints_rl/agent_final.pth`

#### Phase 3: Hypothesis Validation

```bash
python cognitive_dl_rl/validate_hypothesis.py \
    --results_dir results \
    --seed 42
```

**Expected Output**:
- Comprehensive validation report
- Statistical comparisons (correlations, t-tests)
- Visualization of RL vs. Human patterns
- Overall validation score
- Hypothesis verdict (supported/not supported)

## Project Structure

```
cognitive_dl_rl/
├── config.py                           # Configuration management
├── data/
│   ├── cognitive_data_generator.py     # Generate psychologically realistic data
│   └── preprocessing.py                # Data preprocessing utilities
├── models/
│   ├── cognitive_transformer.py        # Main DL model
│   └── positional_encoding.py         # Forgetting-aware positional encoding
├── rl/
│   ├── cognitive_environment.py        # RL environment with constraints
│   ├── cognitive_agent.py             # PPO agent implementation
│   └── training.py                     # RL training loop
├── training/
│   └── dl_trainer.py                  # DL training procedures
├── validation/
│   └── comparative_analysis.py         # Hypothesis validation
├── visualization/
│   └── plots.py                       # Visualization functions
├── utils/
│   └── metrics.py                     # Performance metrics
├── train_dl.py                         # Main DL training script
├── train_rl.py                         # Main RL training script
└── validate_hypothesis.py              # Main validation script
```

## Key Results

### Expected Findings

If hypothesis is supported, RL agents should exhibit:

1. **Learning Curves**: Power law relationship (correlation > 0.7 with human data)
2. **Forgetting Curves**: Exponential decay matching human forgetting rates
3. **Spacing Effect**: 15-25% performance benefit for distributed practice
4. **Capacity Limits**: Effective capacity of 5-7 items (Miller's range)

### Validation Metrics

- **Overall Score**: Composite metric [0, 1] combining all validations
  - > 0.8: Strong support for hypothesis
  - 0.6-0.8: Moderate support
  - < 0.6: Weak support

- **Component Metrics**:
  - Learning curve correlation
  - Forgetting rate similarity
  - Spacing effect similarity
  - Capacity difference

## Theoretical Implications

### If Hypothesis is Supported

1. **Causality Confirmed**: Cognitive constraints CAUSE observed learning patterns
2. **Mechanism Validated**: Attention bottlenecks and memory limitations explain human learning
3. **Predictive Power**: DL models capture true cognitive mechanisms
4. **Practical Applications**:
   - Adaptive educational systems
   - Personalized learning optimization
   - Cognitive load assessment tools
   - Human-AI collaboration design

## Customization

### Adjusting Hyperparameters

Edit `cognitive_dl_rl/config.py` to modify:

- Model architecture (layers, heads, dimensions)
- Training parameters (learning rate, epochs, batch size)
- RL environment constraints (memory capacity, forgetting rate)
- Agent configuration (hidden dimensions, PPO parameters)

### Using Real Human Data

Replace synthetic data generation with real datasets:

```python
from cognitive_dl_rl.data.preprocessing import split_data, create_dataloaders

# Load your data
human_data = pd.read_csv('your_human_data.csv')

# Ensure required columns: participant_id, trial_number, performance, etc.

# Continue with standard pipeline
train_data, val_data, test_data = split_data(human_data)
```

## Computational Requirements

### Recommended Specifications

- **CPU**: 8+ cores
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Storage**: 5GB for checkpoints and results

### Training Time Estimates

- **Phase 1 (DL)**: 1-2 hours (GPU) / 4-6 hours (CPU)
- **Phase 2 (RL)**: 2-3 hours (GPU) / 6-8 hours (CPU)
- **Phase 3 (Validation)**: < 10 minutes

**Total**: 3-5 hours on GPU, 10-15 hours on CPU

## Citation

If you use this code in your research, please cite:

```bibtex
@software{cognitive_dl_rl_2024,
  title={Cognitive Load Theory and Attention Mechanisms: Predicting Human Learning with Deep Learning, Validated Through Reinforcement Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/cognitive-dl-rl}
}
```

## References

1. Sweller, J. (1988). Cognitive load during problem solving: Effects on learning. Cognitive Science, 12(2), 257-285.
2. Cowan, N. (2001). The magical number 4 in short-term memory: A reconsideration of mental storage capacity. Behavioral and Brain Sciences, 24(1), 87-114.
3. Ebbinghaus, H. (1885). Memory: A contribution to experimental psychology.
4. Newell, A., & Rosenbloom, P. S. (1981). Mechanisms of skill acquisition and the law of practice. Cognitive Skills and Their Acquisition, 1, 1-55.
5. Broadbent, D. E. (1958). Perception and communication. Elmsford, NY: Pergamon Press.

## License

MIT License - See LICENSE file for details

## Contact

For questions, issues, or collaborations:
- GitHub Issues: https://github.com/yourusername/cognitive-dl-rl/issues
- Email: your.email@example.com

## Acknowledgments

This project bridges cognitive science and machine learning, building on decades of psychological research and recent advances in deep learning and reinforcement learning. Special recognition to the researchers whose theoretical work forms the foundation of this computational validation.
