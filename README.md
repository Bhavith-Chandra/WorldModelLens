# World Model Lens

**The definitive interpretability library for ANY world model.**

[![PyPI Version](https://img.shields.io/pypi/v/world-model-lens.svg)](https://pypi.org/project/world-model-lens/)
[![Python Versions](https://img.shields.io/pypi/pyversions/world-model-lens.svg)](https://pypi.org/project/world-model-lens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/Bhavith-Chandra/WorldModelLens/workflows/CI/badge.svg)](https://github.com/Bhavith-Chandra/WorldModelLens/actions)
[![Coverage](https://codecov.io/gh/Bhavith-Chandra/WorldModelLens/branch/main/graph/badge.svg)](https://codecov.io/gh/Bhavith-Chandra/WorldModelLens)

World Model Lens provides a unified, backend-agnostic interface for analyzing, probing, and understanding world models. It works with any model that implements latent state + dynamics — no RL assumptions required.

---

## Table of Contents

- [Philosophy](#philosophy)
- [Why World Model Lens?](#why-world-model-lens)
- [Supported Models](#supported-models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Features](#features)
- [Architecture](#architecture)
- [CLI Reference](#cli-reference)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Philosophy

**World Model Lens is not just for RL agents.** It's designed to be:

- **Backend-Agnostic**: Works with ANY world model architecture — RSSM, JEPA, transformers, video prediction, etc.
- **Minimal Interface**: Only core methods are required; RL-specific features are optional
- **Extensible**: Add new models by implementing a thin adapter
- **Non-RL First-Class**: Video prediction, planning, and scientific simulation models are treated as first-class citizens

---

## Why World Model Lens?

Inspired by [TransformerLens](https://github.com/neelnanda-io/TransformerLens) for language models, World Model Lens brings the same level of interpretability tooling to world models — but **without assuming reinforcement learning**.

Whether you're working on:

| Domain | Examples |
|--------|----------|
| **Reinforcement Learning** | DreamerV3, TD-MPC2, IRIS agents |
| **Video Prediction** | WorldDreamer, video generation models |
| **Planning & Control** | MPC, trajectory optimization |
| **Robotics** | Manipulation, locomotion, navigation |
| **Autonomous Driving** | Camera/LiDAR fusion world models |
| **Scientific Simulation** | Physics, chemistry, climate modeling |
| **Your Custom Model** | Any latent dynamics model |

World Model Lens provides the interpretability tools you need.

---

## Supported Models

| Model Family | Implementations | RL-Specific |
|-------------|----------------|-------------|
| **RSSM-based** | DreamerV1, DreamerV2, DreamerV3 | Optional |
| **JEPA-style** | TD-MPC2, Contrastive Predictive | Optional |
| **Transformer** | IRIS, Decision Transformer | Optional |
| **Video Prediction** | VideoWorldModel, Video Adapter | No |
| **Planning** | Planning Adapter, MPC wrappers | No |
| **World Models** | Ha & Schmidhuber, PlaNet | Optional |
| **Robotics** | Robotics Adapter | Optional |
| **Autonomous Driving** | AD Adapter | No |
| **Custom** | GenericAdapter, Template | Optional |

**Non-RL world models are first-class citizens.** The only required methods are `encode()` and `dynamics()`.

---

## Installation

### From PyPI

```bash
pip install world-model-lens
```

### From Source

```bash
git clone https://github.com/Bhavith-Chandra/WorldModelLens
cd world_model_lens
pip install -e ".[dev]"
```

### Optional Dependencies

```bash
# Visualization support
pip install world-model-lens[viz]

# API server
pip install world-model-lens[api]

# Documentation
pip install world-model-lens[docs]

# All extras
pip install world-model-lens[dev,viz,api,docs]
```

---

## Quick Start

### 1. Create an Adapter (or use a built-in one)

```python
from world_model_lens import WorldModelConfig
from world_model_lens.backends.generic_adapter import WorldModelAdapter

class MyWorldModel(WorldModelAdapter):
    def encode(self, obs, context=None):
        # Required: convert observation to latent state
        state = self.encoder(obs)
        return state, state  # (state, encoding)
    
    def dynamics(self, state, action=None):
        # Required: predict next state (imagination)
        return self.rnn(state)
    
    def transition(self, state, action=None, input_=None):
        # Optional: full transition function
        return self.dynamics(state, action)

config = WorldModelConfig(d_obs=128, d_state=64)
model = MyWorldModel(config)
```

### 2. Wrap and Analyze

```python
from world_model_lens import HookedWorldModel
import torch

# Wrap with hooks and caching
wm = HookedWorldModel(adapter=model, config=config)

# Run forward pass with activation caching
observations = torch.randn(10, 128)  # 10 timesteps
traj, cache = wm.run_with_cache(observations=observations)

# Imagine future trajectories
imagined = wm.imagine(start_state=traj.states[0], horizon=20)
```

### 3. Use Interpretability Tools

```python
# Linear probing for concepts
from world_model_lens.probing import LatentProber
prober = LatentProber(wm)
probe_results = prober.train_probe(concept="velocity", trajectories=[traj])

# Activation patching
from world_model_lens.patching import TemporalPatcher
patcher = TemporalPatcher(wm)
results = patcher.full_sweep(traj)

# Safety analysis
from world_model_lens.safety import SafetyAnalyzer
analyzer = SafetyAnalyzer(wm)
safety_report = analyzer.run_safety_audit(traj)
```

---

## Core Concepts

### WorldState

The fundamental unit of analysis:

```python
from world_model_lens import WorldState

state = WorldState(
    state=torch.randn(64),      # Required: latent state
    timestep=0,                  # Optional: timestep
    action=torch.randn(4),       # Optional: RL action
    reward=torch.tensor(1.0),   # Optional: RL reward
    metadata={"frame": frame},  # Optional: any data
)
```

### WorldTrajectory

A sequence of states:

```python
from world_model_lens import WorldTrajectory

traj = WorldTrajectory(states=[state1, state2, state3])
print(f"Length: {len(traj)}")
print(f"Total reward: {traj.total_reward}")
```

### HookedWorldModel

Wrapper that provides:
- **Activation caching**: Store any intermediate activation
- **Hook system**: Intercept and modify forward passes
- **Analysis tools**: Built-in interpretability methods

```python
wm = HookedWorldModel(adapter=model, config=config)
traj, cache = wm.run_with_cache(observations)

# Add custom hooks
def hook_fn(activation, hook):
    print(f"Layer: {hook.name}, Shape: {activation.shape}")

wm.run_with_hooks(observations, fwd_hooks=[("block.hook_resid_post", hook_fn)])
```

---

## Features

### Probing

| Feature | Description |
|---------|-------------|
| **Linear Probing** | Train probes on latent representations |
| **Temporal Memory** | Analyze memory retention over time |
| **Geometry Analysis** | PCA, clustering, manifold analysis |
| **Concept Discovery** | Find semantic concepts in latent space |

### Patching

| Feature | Description |
|---------|-------------|
| **Activation Patching** | Replace activations during forward pass |
| **Causal Tracing** | Trace causal pathways through the model |
| **Dimensional Patching** | Test individual latent dimensions |

### Branching

| Feature | Description |
|---------|-------------|
| **Imagination Branching** | Fork trajectories for comparison |
| **Counterfactual Analysis** | "What if" scenario exploration |

### Safety & Analysis

| Feature | Description |
|---------|-------------|
| **Safety Audit** | Detect OOD states, instabilities |
| **Belief Analysis** | Surprise, entropy, hallucination detection |
| **Robustness Testing** | Adversarial perturbation analysis |

### Production Tools

| Feature | Description |
|---------|-------------|
| **CLI** | Command-line interface for common tasks |
| **FastAPI Server** | REST API for analysis endpoints |
| **Monitoring** | Prometheus metrics, OpenTelemetry tracing |
| **Benchmarking** | Latency, memory, throughput profiling |

---

## Architecture

```
world_model_lens/
├── core/                      # Core abstractions (backend-agnostic)
│   ├── world_state.py        # WorldState, WorldDynamics
│   ├── world_trajectory.py   # WorldTrajectory
│   ├── config.py             # WorldModelConfig
│   ├── hooks.py             # Hook system
│   └── activation_cache.py   # Activation caching
│
├── backends/                  # Model adapters (implement these)
│   ├── generic_adapter.py    # Abstract base class
│   ├── dreamerv3.py         # DreamerV3 implementation
│   ├── dreamerv2.py         # DreamerV2 implementation
│   ├── tdmpc2.py            # TD-MPC2 implementation
│   ├── iris.py              # IRIS transformer
│   ├── video_adapter.py     # Video prediction adapter
│   └── planning_adapter.py  # Planning model adapter
│
├── analysis/                  # Analysis tools
│   ├── belief_analyzer.py   # Surprise, concepts, saliency
│   └── multimodal.py         # Multi-modal support
│
├── probing/                  # Probing tools
│   ├── prober.py            # Linear probing
│   ├── geometry.py          # Geometric analysis
│   └── temporal_memory.py    # Memory analysis
│
├── patching/                 # Patching tools
│   ├── patcher.py           # Activation patching
│   ├── causal_tracer.py     # Causal tracing
│   └── dim_patcher.py       # Dimension patching
│
├── branching/                # Branching tools
│   ├── brancer.py           # Imagination branching
│   └── counterfactual.py    # Counterfactual analysis
│
├── safety/                   # Safety analysis
│   ├── analyzer.py          # Safety auditing
│   └── robustness.py        # Robustness testing
│
├── benchmarks/               # Benchmarking
│   ├── suite.py             # Evaluation suite
│   └── perf_runner.py       # Performance profiling
│
├── monitoring/               # Production monitoring
│   ├── logging.py           # Structured logging
│   ├── metrics.py           # Prometheus metrics
│   └── tracing.py           # OpenTelemetry tracing
│
├── api/                      # FastAPI server
│   └── server.py           # REST API endpoints
│
├── cli/                      # CLI tools
│   └── commands.py         # Typer CLI commands
│
└── viz/                     # Visualization
    └── ...                  # Plotting utilities
```

---

## CLI Reference

### Installation

```bash
# Install as a command
pip install -e .
wml --help
```

### Common Commands

```bash
# Analyze a model checkpoint
wml analyze --checkpoint model.pt --backend dreamerv3

# Run safety audit
wml safety --checkpoint model.pt --trajectory traj.npy

# Benchmark performance
wml benchmark --checkpoint model.pt --dataset data/

# Probe for concepts
wml probe --checkpoint model.pt --concept velocity

# Interactive analysis
wml explore --checkpoint model.pt
```

### API Server

```bash
# Start the API server
wml serve --host 0.0.0.0 --port 8000

# With Docker
docker-compose up api
```

---

## API Reference

### Core Classes

```python
# Main wrapper
from world_model_lens import HookedWorldModel

wm = HookedWorldModel(adapter, config)
traj, cache = wm.run_with_cache(observations)
imagined = wm.imagine(start_state, horizon=20)

# State and trajectory
from world_model_lens import WorldState, WorldTrajectory

state = WorldState(state=torch.randn(64))
traj = WorldTrajectory(states=[state] * 10)

# Configuration
from world_model_lens import WorldModelConfig

config = WorldModelConfig(
    d_obs=128,           # Observation dimension
    d_state=64,          # Latent state dimension
    d_action=4,          # Action dimension (optional)
    backend="dreamerv3", # Model type
)
```

### Analysis Classes

```python
from world_model_lens.analysis import BeliefAnalyzer
from world_model_lens.probing import LatentProber, GeometryAnalyzer
from world_model_lens.patching import TemporalPatcher
from world_model_lens.safety import SafetyAnalyzer

# Belief analysis
analyzer = BeliefAnalyzer(wm)
surprise = analyzer.surprise_timeline(cache)

# Linear probing
prober = LatentProber(wm)
probe_results = prober.train_probe(concept="velocity", trajectories=[traj])

# Geometry analysis
geo = GeometryAnalyzer(wm)
pca = geo.pca_projection(cache)

# Safety audit
safety = SafetyAnalyzer(wm)
report = safety.run_safety_audit(traj)
```

### Creating Custom Adapters

```python
from world_model_lens.backends.generic_adapter import WorldModelAdapter

class MyAdapter(WorldModelAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = nn.Linear(config.d_obs, config.d_state)
        self.dynamics = nn.GRU(config.d_state, config.d_state)
    
    def encode(self, obs, context=None):
        return self.encoder(obs), self.encoder(obs)
    
    def dynamics(self, state, action=None):
        return self.dynamics(state.unsqueeze(0)).squeeze(0)
    
    def transition(self, state, action=None, input_=None):
        return self.dynamics(state, action)
    
    def initial_state(self, batch_size=1, device=None):
        return torch.zeros(batch_size, self.config.d_state, device=device)
```

---

## Examples

### Video Prediction Analysis

```python
from world_model_lens import HookedWorldModel
from world_model_lens.backends import VideoWorldModelAdapter
import torch

# Load a video prediction model
adapter = VideoWorldModelAdapter.from_pretrained("your/video-model")
wm = HookedWorldModel(adapter, adapter.config)

# Analyze a video sequence
video_frames = torch.randn(16, 3, 64, 64)  # 16 frames
traj, cache = wm.run_with_cache(video_frames)

# Find surprising frames
from world_model_lens.analysis import BeliefAnalyzer
analyzer = BeliefAnalyzer(wm)
surprise = analyzer.surprise_timeline(cache)
```

### Imagination Branching

```python
from world_model_lens.branching import ImaginationBrancher

brancer = ImaginationBrancher(wm)

# Branch at a specific state
branch = brancer.create_branch(traj, branch_point=5)

# Try different actions
for action_sequence in candidate_actions:
    forked = branch.fork()
    imagined = wm.imagine(forked.initial_state, horizon=10, actions=action_sequence)
    scores.append(evaluate(imagined))
```

### Causal Tracing

```python
from world_model_lens.patching import CausalTracer

tracer = CausalTracer(wm)
results = tracer.trace(
    clean_cache=clean_cache,
    corrupted_cache=corrupted_cache,
    layers=["encoder", "dynamics", "decoder"],
)
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Set up development environment
git clone https://github.com/Bhavith-Chandra/WorldModelLens
cd world_model_lens
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check world_model_lens/
mypy world_model_lens/
```

---

## Citation

If you use World Model Lens in your research, please cite:

```bibtex
@software{worldmodellens,
  title = {World Model Lens: Interpretability for Any World Model},
  author = {World Model Lens Team},
  year = {2024},
  url = {https://github.com/Bhavith-Chandra/WorldModelLens}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

World Model Lens is inspired by:

- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) — Interpretability for language models
- [Dreamer](https://github.com/danijar/dreamer) — World models for RL
- [TD-MPC](https://github.com/nick不通/tdmpc) — Implicit world models
- [IRIS](https://github.com/eloialonso/iris) — Transformer world models
