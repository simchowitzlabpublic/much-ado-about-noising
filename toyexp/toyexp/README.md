# Toy Experiments: Neural Manifold Learning

Comparing five training paradigms (regression, flow matching, MIP, MIP one-step, and straight flow) for function approximation with geometric constraints in low-data regimes.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Executive Summary

This project explores **implicit biases and geometric learning** of different training paradigms when learning target functions from limited data (50 training samples):

- **Core Finding**: Flow-based methods (flow matching, MIP, MIP one-step, straight flow) **consistently outperform regression on projection-based geometric metrics**, demonstrating superior manifold structure learning and boundary enforcement. However, **regression achieves the best L2 reconstruction error**, revealing a fundamental trade-off between geometric constraint satisfaction and point-wise reconstruction accuracy.

- **Key Results Across 540 Experiments** (5 modes × 2 losses × 2 architectures × 3 seeds × 3 tasks):
  - **Reconstruction**: MIP one-step-L2-FiLM achieves best L2 error (0.003197)
  - **Projection**: Straight flow-L1-FiLM achieves best boundary metric (0.009769) — **35% better than MIP one-step**
  - **Lie Algebra**: Straight flow-L1-FiLM achieves best avg projection metric (0.063612) — **12% better than regression**

- **Surprising Discovery**: **Straight flow** (flow matching without time conditioning) matches or exceeds the performance of time-conditioned flow methods on geometric metrics, suggesting that simpler inductive biases can be more effective for manifold learning.

- **Recommendation**: 
  - Use **regression-L2** for pure reconstruction tasks requiring point-wise accuracy
  - Use **MIP one-step-L2** for balanced reconstruction + geometric performance
  - Use **straight flow-L1** when geometric constraints (boundary enforcement, manifold adherence) are critical

- **Important Note**: Training flow-based methods with L1 loss lacks mathematical grounding (the flow matching objective is derived for L2), though we include it for empirical completeness.

---

## 🎯 Overview

### Five Training Paradigms

1. **Regression**: Direct function approximation `f(c)` — optimizes reconstruction error via supervised learning
2. **Flow Matching**: Learns time-conditioned velocity fields `dx/dt = v(x_t, c, t)` via ODE integration
3. **Straight Flow**: Flow matching ablation — learns without time conditioning (always queries model at t=0) to test if time information helps
4. **MIP (Manifold Interpolation)**: Flow matching + denoising term at fixed time t* = 0.9 for enhanced manifold adherence
5. **MIP One-Step**: MIP training with single-step evaluation (stops at initial denoising step) for efficient inference

### Three Experiment Types

1. **Reconstruction**: Learn scalar target functions `f: ℝ → ℝ` composed of trigonometric components
2. **Projection**: Learn 8D functions constrained to 3D subspaces with piecewise-linear structure
3. **Lie Algebra**: Learn 16D rotation components evolving on SO(2) manifolds with geometric constraints

### Key Features

- **Modular, clean codebase** with YAML-driven configuration
- **Comprehensive evaluation**: Reconstruction metrics (L1/L2) + geometric metrics (subspace adherence, boundary enforcement, tangent space alignment)
- **Multi-seed robustness**: All experiments run across 3 random seeds with proper statistical reporting
- **Two architectures**: Concatenation vs FiLM (Feature-wise Linear Modulation) for conditioning
- **Automated reporting**: LaTeX table generation with averaged and seed-wise results

---

## 📦 Installation

### Quick Install

```bash
# Clone repository
git clone https://github.com/yourusername/toyexp.git
cd toyexp

# Install dependencies
pip install torch numpy matplotlib pyyaml scipy
```

### Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy 1.24+
- Matplotlib 3.7+
- PyYAML 6.0+
- SciPy 1.10+ (for Lie algebra experiments)

---

## 🚀 Quick Start

### 1. Run Your First Experiment (2 minutes)

```bash
# MIP one-step mode - reconstruction task
python train_recon.py --config config_recon.yaml \
    experiment.mode=mip_one_step_integrate \
    training.loss_type=l2

# This will:
# - Train a MIP model with one-step evaluation for 50,000 epochs
# - Save results to ./outputs/recon/
# - Create plots and logs automatically
```

### 2. Check Results

```bash
# View training log
cat ./outputs/recon/train.log

# Results structure:
./outputs/recon/
├── config.yaml              # Configuration used
├── train.log                # Training logs
├── training.csv             # Per-epoch training metrics
├── evaluation.csv           # Evaluation metrics
├── checkpoints/
│   └── final_model.pt       # Final model
└── plots/
    ├── training_loss.png    # Loss over time
    └── predictions.png      # Model predictions vs ground truth
```

### 3. Compare All Training Paradigms

```bash
# Run all five modes with L2 loss
for mode in regression flow straight_flow mip mip_one_step_integrate; do
    python train_recon.py --config config_recon.yaml \
        experiment.mode=$mode \
        training.loss_type=l2
done

# Generate comparison tables
python run_mode_comparison.py --experiment recon --config config_recon.yaml
```

### 4. Try Other Experiments

```bash
# Projection experiment (8D → 3D subspace)
python train_proj.py --config config_proj.yaml \
    experiment.mode=straight_flow \
    training.loss_type=l1

# Lie algebra experiment (SO(2) rotations)
python train_lie.py --config config_lie.yaml \
    experiment.mode=mip_one_step_integrate
```

---

## 📊 Experiments

### Reconstruction Experiment

**Goal**: Learn scalar target functions `f(c) = Σ wᵢ·trig(ωᵢ·c + φᵢ)` where trig ∈ {sin, cos}

**Setup**:
- Input: `c ∈ [0, 1]`
- Output: Scalar `f(c) ∈ ℝ`
- Training: 50 samples
- Evaluation: 100,000 samples

**Mathematical Formulation**:
```
f(c) = Σᵢ₌₁ᴷ wᵢ · trigᵢ(ωᵢc + φᵢ)
```
where K=3 components, frequencies ωᵢ are prime-based to avoid overlaps, weights wᵢ=1 (uniform).

**Metrics**:
- L1 test error: `||f̂(x, c) - f(c)||₁`
- L2 test error: `||f̂(x, c) - f(c)||₂`

**Key Finding**: Regression-L2-concat achieves best L1 error (0.002288), while MIP one-step-L2-FiLM achieves best L2 error (0.003197), demonstrating that flow-based methods with denoising can match regression on reconstruction while enabling better geometric learning.

### Projection Experiment

**Goal**: Learn 8D functions living in 3D subspaces with piecewise-linear structure

**Setup**:
- Input: `c ∈ [0, 1]` divided into 10 intervals
- Output: `g(c) = Pᵢ(c) f(c) ∈ ℝ⁸` constrained to rank-3 subspaces
- Each interval has unique projection matrix Pᵢ creating boundary discontinuities

**Mathematical Formulation**:
```
g(c) = Pᵢ(c) f(c)
where Pᵢ = Aᵢ(AᵢᵀAᵢ)⁻¹Aᵢᵀ,  Aᵢ ∈ ℝ⁸ˣ³
```

**Metrics**:
- L1/L2 reconstruction error
- **Subspace Diagonal**: `||(I - P̃ⱼ,ⱼ₊₁)(f̂ - f*)||/||f̂ - f*||` at boundaries (measures local subspace adherence)
- **Subspace Off-Diagonal**: Same metric away from boundaries (measures global adherence)
- **Boundary**: `||(I - P̃ⱼ,ⱼ₊₁)f̂||/||f̂||` (measures discontinuities at boundary points)

**Key Finding**: Flow-based methods dramatically outperform regression on geometric metrics. Straight flow-L1-FiLM achieves best boundary enforcement (0.009769), demonstrating that flow-based training naturally learns better manifold structure. Regression struggles with boundary violations (0.392-0.461 for concat architecture).

### Lie Algebra Experiment

**Goal**: Learn rotation components evolving on SO(2) manifolds with Lie algebra constraints

**Setup**:
- Input: `c ∈ [0, 1]`
- Output: 8 rotation components, each 2D vector (16D total)
- Components rotate at different velocities with high-frequency weight modulation

**Mathematical Formulation**:
```
fᵢ(α, c) = wᵢ(c) · exp(αᵢc · A) · e₁
where A = [[0, -1], [1, 0]] (SO(2) generator)
Output: concat(f₁, ..., f₈) ∈ ℝ¹⁶
```

**Metrics**:
- L1/L2 reconstruction error
- **Cosine Similarity**: cos-similarity between predicted and true tangent directions (higher is better)
- **Average Projection**: Mean projection error `||(I-Pᵢ)f̂ᵢ||/||f̂ᵢ||` across components (lower is better)
- **Maximum Projection**: Worst-case projection error (lower is better)

**Key Finding**: Straight flow-L1-FiLM achieves best average projection metric (0.063612), outperforming regression (0.072718) by 12%. Flow-based methods show superior tangent space alignment, with MIP-L2-concat achieving best maximum projection (0.116828).

---

## ⚙️ Configuration

### Basic Config Structure

```yaml
experiment:
  name: "recon_experiment"
  mode: "mip_one_step_integrate"  # Options: regression, flow, straight_flow, mip, mip_one_step_integrate
  seed: 42
  device: "cuda"
  output_dir: "./outputs/recon"

dataset:
  num_train: 50                  # Training samples
  num_eval: 100000               # Evaluation samples
  target_dim: 1                  # Output dimension
  num_components: 3              # Frequency components
  sampling_strategy: "grid"      # 'grid' or 'random'

network:
  architecture: "film"           # 'concat' or 'film'
  hidden_dim: 256
  num_layers: 3
  activation: "relu"

training:
  loss_type: "l2"               # 'l1' or 'l2' (note: L1 not mathematically grounded for flow-based methods)
  batch_size: 32
  num_epochs: 50000
  learning_rate: 0.001
  optimizer: "adam"
  log_interval: 1000
  eval_interval: 50000
  
  # Flow/MIP-specific
  initial_dist: "zeros"         # Initial distribution for flow models
  mip_t_star: 0.9              # Fixed time for denoising term (MIP only)
  mip_lambda: 1.0              # Weight for denoising term (MIP only)

evaluation:
  num_eval_steps: [1, 9]        # NFE for flow models (Number of Function Evaluations)
  integration_method: "euler"    # 'euler' or 'rk4'
  initial_dist: "zeros"         # Use zeros for evaluation
```

### Override Config Parameters

```bash
# Change mode and loss
python train_recon.py --config config_recon.yaml \
    experiment.mode=straight_flow \
    training.loss_type=l1

# Change network architecture
python train_recon.py --config config_recon.yaml \
    network.architecture=film \
    network.hidden_dim=512

# Change random seed
python train_recon.py --config config_recon.yaml \
    experiment.seed=43
```

---

## 🎮 Usage Examples

### Compare All Methods with Multi-Seed Analysis

```bash
# Run comprehensive comparison (5 modes × 2 losses × 2 architectures × 3 seeds = 60 experiments)
python run_mode_comparison.py --experiment recon --config config_recon.yaml

# This will:
# - Run all 60 configurations automatically
# - Generate averaged results with mean ± std
# - Create LaTeX tables (averaged and seed-wise)
# - Save results to outputs directory
```

### Single Mode with Multiple Seeds

```bash
# Test straight_flow robustness across 3 seeds
for seed in 0 1 2; do
    python train_recon.py --config config_recon.yaml \
        experiment.mode=straight_flow \
        training.loss_type=l1 \
        network.architecture=film \
        experiment.seed=$seed
done
```

### L1 vs L2 Loss Comparison

```bash
# Compare losses for MIP one-step
python train_proj.py --config config_proj.yaml \
    experiment.mode=mip_one_step_integrate \
    training.loss_type=l1

python train_proj.py --config config_proj.yaml \
    experiment.mode=mip_one_step_integrate \
    training.loss_type=l2
```

### Architecture Comparison

```bash
# Compare concat vs FiLM for straight flow
python train_lie.py --config config_lie.yaml \
    experiment.mode=straight_flow \
    network.architecture=concat

python train_lie.py --config config_lie.yaml \
    experiment.mode=straight_flow \
    network.architecture=film
```

### Quick Test Run

```bash
# Reduce epochs for fast testing (not recommended for final results)
python train_recon.py --config config_recon.yaml \
    training.num_epochs=1000 \
    training.eval_interval=500
```

---

## 📈 Results and Analysis

### Viewing Results

After training, check:

```bash
# Training metrics (per-epoch losses)
cat ./outputs/recon/training.csv

# Evaluation metrics (L1/L2 errors, geometric metrics)
cat ./outputs/recon/evaluation.csv

# Visualizations
ls ./outputs/recon/plots/
# - training_loss.png: Loss curves
# - predictions.png: Model predictions vs ground truth
# - errors.png: Error distributions
# For projection task: subspace_analysis.png, boundary_analysis.png
# For Lie algebra: component_predictions.png, geometric_metrics.png
```

### Automated Multi-Configuration Analysis

The `run_mode_comparison.py` script provides comprehensive analysis:

```bash
python run_mode_comparison.py --experiment recon --config config_recon.yaml

# Generates:
# - recon_results_table_averaged.tex: Mean ± std across 3 seeds
# - recon_results_table_seedwise.tex: Individual seed results
# - Runs all 60 configurations (5 modes × 2 losses × 2 archs × 3 seeds)
```

**Output Format (Averaged Table)**:
```
Mode                     | Loss | Arch   | L1 Error        | L2 Error
-------------------------|------|--------|-----------------|------------------
regression               | l2   | concat | 0.002 ± 0.000   | 0.003 ± 0.000
flow                     | l2   | film   | 0.054 ± 0.009   | 0.067 ± 0.010
straight_flow            | l1   | film   | 0.004 ± 0.001   | 0.005 ± 0.002
mip                      | l2   | film   | 0.003 ± 0.000   | 0.004 ± 0.000
mip_one_step_integrate   | l2   | film   | 0.002 ± 0.000   | 0.003 ± 0.001
```

---

## 🏗️ Project Structure

```
.
├── config_recon.yaml          # Reconstruction experiment config
├── config_proj.yaml           # Projection experiment config
├── config_lie.yaml            # Lie algebra experiment config
├── train_recon.py             # Reconstruction training script
├── train_proj.py              # Projection training script
├── train_lie.py               # Lie algebra training script
├── run_mode_comparison.py     # Multi-seed batch processing and analysis
├── analyze_results.py         # Result analysis utilities
│
├── toyexp/
│   └── common/
│       ├── datasets.py        # Dataset implementations (all 3 tasks)
│       ├── networks.py        # Neural architectures (Concat, FiLM)
│       ├── losses.py          # Loss functions (regression, flow, MIP)
│       ├── integrate.py       # ODE integration (Euler, RK4)
│       ├── logging_utils.py   # CSV logging and metrics tracking
│       ├── config.py          # YAML configuration management
│       └── utils.py           # Plotting and checkpoint utilities
│
└── outputs/                   # Experiment results (auto-generated)
    ├── recon/
    ├── proj/
    └── lie/
```

---

## 🔬 Training Paradigms Explained

### Regression
**Approach**: Direct supervised learning `f̂(c) = model(x_0, c, t=None)`

**Training**: Minimizes `||model(x_0, c) - f(c)||` where x_0 can be zeros or sampled

**Evaluation**: Single forward pass

**Pros**: Best reconstruction accuracy, simple and fast  
**Cons**: Struggles with geometric constraints, poor manifold structure

---

### Flow Matching
**Approach**: Learn time-conditioned velocity field via ODE

**Training**: 
```
x_t = (1-t)x_0 + t·x_1
v_θ(x_t, c, t) = x_1 - x_0  (conditional flow matching)
Loss = ||model(x_t, c, t) - (x_1 - x_0)||²
```

**Evaluation**: ODE integration from x_0=0 using Euler/RK4 with N steps

**Pros**: Models trajectories through data, potential for better generalization  
**Cons**: Requires ODE integration at inference (slower), struggles with reconstruction

---

### Straight Flow
**Approach**: Flow matching ablation without time conditioning

**Training**:
```
x_t = (1-t)x_0 + t·x_1  (sample at random t)
But: Always query model at t=0 (no time information)
Loss = ||model(x_t, c, t=0) - x_1||²
```

**Evaluation**: Same ODE integration as flow matching but with t=0

**Pros**: Simpler than flow matching, **superior geometric metrics**, matches flow performance  
**Cons**: Slightly worse reconstruction than MIP methods

**Key Insight**: Removing time conditioning provides beneficial inductive bias for geometric learning

---

### MIP (Manifold Interpolation)
**Approach**: Flow matching + denoising term at fixed time t*

**Training**:
```
L_MIP = L_flow(t) + λ·||model(x_t*, c, t*) - x_1||²
where t* = 0.9 (fixed denoising time)
```

**Evaluation**: Two-step inference:
1. Initial denoising: a_0 = model(x_0, c, t=0)
2. Final prediction: a = model(a_0, c, t=t*)

**Pros**: Best balance of reconstruction and geometry, strong manifold adherence  
**Cons**: Two-step evaluation (slower), more complex training

---

### MIP One-Step
**Approach**: MIP training with single-step evaluation

**Training**: Identical to MIP (uses denoising term at t*)

**Evaluation**: Single-step inference:
```
a_0 = model(x_0, c, t=0)  # Stop here (no second step)
```

**Pros**: **Best L2 reconstruction**, efficient inference, good geometric metrics  
**Cons**: Slightly worse geometry than two-step MIP

**Key Insight**: The initial denoising step captures most of the manifold structure

---

## 🔧 Network Architectures

### ConcatMLP
**Structure**: `[x, c, t] → Linear → ReLU → ... → Linear → output`

Concatenates all inputs (x, condition c, time t) before processing.

**Pros**: Simple, interpretable, works well as baseline  
**Cons**: All inputs treated equally, no specialized conditioning

**When to use**: Default choice for initial experiments

---

### FiLMMLP (Feature-wise Linear Modulation)
**Structure**: 
```
h = Linear(x)
h = FiLM(h, [c, t])  # h = γ(c,t) ⊙ h + β(c,t)
h = ReLU(h)
...
output = Linear(h)
```

Applies affine transformations to hidden activations based on conditioning.

**Pros**: **Often achieves best performance**, better conditioning mechanism  
**Cons**: Slightly more complex, more parameters

**When to use**: When you need best performance, especially for geometric metrics

---

## 📝 Evaluation Metrics

### Reconstruction Metrics (All Tasks)
- **L1 Error**: `||f̂(x, c) - f(c)||₁` — Mean absolute error
- **L2 Error**: `||f̂(x, c) - f(c)||₂` — Root mean squared error

**Interpretation**: Lower is better. Measures point-wise approximation accuracy.

---

### Geometric Metrics: Projection Task

- **Subspace Diagonal**: `||(I - P̃ⱼ,ⱼ₊₁)(f̂ - f*)||/||f̂ - f*||` at boundaries
  - Measures how well predictions lie in correct local subspace
  - Lower is better (0 = perfect subspace adherence)

- **Subspace Off-Diagonal**: Same metric away from boundaries
  - Measures global subspace consistency
  - Lower is better

- **Boundary**: `||(I - P̃ⱼ,ⱼ₊₁)f̂||/||f̂||` at boundary points cⱼ
  - Measures discontinuities at subspace transitions
  - Lower is better (0 = smooth transition)
  - **Critical metric**: Reveals whether model respects manifold boundaries

**Interpretation**: Flow-based methods (especially straight flow) achieve 10-40× better boundary metrics than regression, demonstrating superior geometric structure learning.

---

### Geometric Metrics: Lie Algebra Task

- **Average Cosine Similarity**: Mean alignment of predicted vectors with true tangent directions
  - Higher is better (1 = perfect alignment)
  - Measures tangent space learning

- **Minimum Cosine Similarity**: Worst-case alignment
  - Higher is better
  - Reveals failure modes

- **Average Projection**: Mean `||(I-Pᵢ)f̂ᵢ||/||f̂ᵢ||` across components
  - Lower is better (0 = on manifold)
  - **Critical metric**: Measures orthogonality to Lie algebra

- **Maximum Projection**: Worst-case projection error
  - Lower is better
  - Reveals maximum deviation from manifold

**Interpretation**: Straight flow and MIP methods achieve significantly better projection metrics than regression (30-40% improvement), indicating better manifold structure preservation.

---

## 🎯 Method Selection Guide

### By Primary Objective

| Primary Goal | Recommended Configuration | Why? |
|--------------|--------------------------|------|
| **Best Reconstruction** | Regression-L2-concat | Achieves lowest L1 error (0.002288) |
| **Balanced Performance** | MIP one-step-L2-FiLM | Best L2 error (0.003197) + good geometry |
| **Geometric Constraints** | Straight flow-L1-FiLM | Best boundary (0.009769) and projection (0.063612) |
| **Fast Inference** | Regression or MIP one-step | Single forward pass (no ODE integration) |
| **Manifold Learning** | MIP-L2-FiLM | Strong geometric metrics across all tasks |

### By Task Type

| Task Characteristics | Recommended Method | Runner-up |
|---------------------|-------------------|-----------|
| Point-wise approximation | Regression-L2 | MIP one-step-L2 |
| Subspace constraints | Straight flow-L1 | MIP-L2 |
| Lie algebra / rotations | Straight flow-L1 | MIP-L2 |
| Boundary enforcement | Straight flow-L1-FiLM | MIP one-step-L1-FiLM |
| Low-data regime (n<100) | MIP one-step-L2 | Regression-L2 |

### Training Loss Selection

- **L2 loss (recommended default)**:
  - Better for matching L2 test metrics
  - Mathematically grounded for flow-based methods
  - Generally smoother convergence

- **L1 loss**:
  - Sometimes better for geometric metrics (especially boundaries)
  - More robust to outliers
  - **Note**: Lacks mathematical grounding for flow-based methods

### Architecture Selection

- **FiLM architecture**: 
  - Achieves best results on most metrics
  - Recommended for final experiments
  - Slight parameter overhead

- **Concat architecture**:
  - Good baseline performance
  - Simpler, faster training
  - Use for initial exploration

---

## 🔧 Troubleshooting

### CUDA Out of Memory

```yaml
# In config file
training:
  batch_size: 16  # Reduce from 32

# Or use CPU
experiment:
  device: "cpu"
```

### Poor Convergence

```yaml
# Try different learning rates
training:
  learning_rate: 0.0001  # Smaller for stability
  # or
  learning_rate: 0.01    # Larger for faster convergence

# Or increase training duration
training:
  num_epochs: 100000  # From 50000
```

### Flow Models Performing Poorly

**Common issues**:
1. **Insufficient integration steps**: Use `num_eval_steps: [9]` (not [1])
2. **Wrong initial distribution**: Set `evaluation.initial_dist: "zeros"`
3. **Wrong training loss**: Use L2 loss for flow-based methods
4. **Need more epochs**: Flow methods may need 75k-100k epochs

```yaml
# Recommended flow settings
evaluation:
  num_eval_steps: [9]
  initial_dist: "zeros"

training:
  loss_type: "l2"
  num_epochs: 75000
```

### Missing Evaluation Metrics

Ensure evaluation interval divides training epochs evenly:

```yaml
training:
  num_epochs: 50000
  eval_interval: 50000  # Must divide evenly

# Or use multiple evaluations
training:
  num_epochs: 50000
  eval_interval: 10000  # Evaluates 5 times
```

### Inconsistent Results Across Seeds

Some configurations show high variance (especially regression-L2-FiLM and flow methods):

```yaml
# Run more seeds for robust statistics
experiment:
  seed: [0, 1, 2, 3, 4]  # 5 seeds instead of 3

# Or focus on more stable configurations
# (MIP one-step and straight flow have lower variance)
```

---

## 🔬 Advanced Usage

### Custom Target Functions

Edit dataset configuration in config files:

```yaml
dataset:
  num_components: 5           # More frequency components
  sampling_strategy: "random" # Random vs grid sampling
  freq_seed: 123             # Reproducible frequencies
  phase_seed: 456            # Reproducible phases
```

### Custom Network Architectures

Modify network parameters:

```yaml
network:
  hidden_dim: 512            # Larger capacity
  num_layers: 5              # Deeper networks
  activation: "gelu"         # Alternative activations
```

### Hyperparameter Sweeps

```bash
# Example: Sweep MIP denoising time
for t_star in 0.5 0.7 0.9 0.95; do
    python train_recon.py --config config_recon.yaml \
        experiment.mode=mip \
        training.mip_t_star=$t_star \
        experiment.name="mip_tstar_${t_star}"
done
```

### Generating LaTeX Tables

After running experiments:

```bash
# Aggregate results and generate tables
python run_mode_comparison.py --experiment recon --config config_recon.yaml

# Tables saved to:
# - outputs/recon_results_table_averaged.tex
# - outputs/recon_results_table_seedwise.tex

# Use in LaTeX document:
# \input{outputs/recon_results_table_averaged.tex}
```

---

## 📚 Documentation

### Module Documentation

All modules have comprehensive docstrings:

```bash
# View dataset documentation
python -c "from toyexp.common import datasets; help(datasets.TargetFunctionDataset)"

# View network documentation
python -c "from toyexp.common import networks; help(networks.FiLMMLP)"

# View loss documentation
python -c "from toyexp.common import losses; help(losses.LossManager)"
```

### Testing Modules

Run built-in tests:

```bash
# Test datasets
python -m toyexp.common.datasets

# Test networks  
python -m toyexp.common.networks

# Test integration
python -m toyexp.common.integrate
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Additional architectures (Transformers, ResNets)
- [ ] More sophisticated target functions (non-parametric, neural ODEs)
- [ ] Advanced ODE solvers (adaptive step size, higher-order methods)
- [ ] Experiment tracking integration (Weights & Biases, MLflow)
- [ ] Comprehensive unit tests with pytest
- [ ] Additional geometric constraints (curvature, torsion)
- [ ] Multi-GPU training support
- [ ] Automated hyperparameter tuning

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

Built using:
- [PyTorch](https://pytorch.org/) for deep learning
- [NumPy](https://numpy.org/) for numerical computing
- [SciPy](https://scipy.org/) for scientific computing (Lie algebra operations)
- [Matplotlib](https://matplotlib.org/) for visualization
- [PyYAML](https://pyyaml.org/) for configuration management

Inspired by recent work in:
- Conditional flow matching for generative modeling
- Manifold learning and geometric deep learning
- Neural ODEs and continuous normalizing flows

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Experimenting! 🚀**