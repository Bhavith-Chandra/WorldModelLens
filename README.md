# World Model Lens

**Observability & Replay Tooling for AI Safety & Interpretability Research**

[![PyPI Version](https://img.shields.io/pypi/v/world-model-lens.svg)](https://pypi.org/project/world-model-lens/)
[![Python Versions](https://img.shields.io/pypi/pyversions/world-model-lens.svg)](https://pypi.org/project/world-model-lens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://codecov.io/gh/Bhavith-Chandra/WorldModelLens/branch/main/graph/badge.svg)](https://codecov.io/gh/Bhavith-Chandra/WorldModelLens)

World Model Lens provides **observability** and **replay** tooling for analyzing, debugging, and understanding world models through the lens of AI safety & interpretability research.

### Key Capabilities

| Category | Capabilities |
|----------|-------------|
| **Observability** | Activation caching, saliency maps, surprise detection, belief tracking, uncertainty quantification |
| **Replay** | Trajectory replay, intervention replay, imagination branching, counterfactual analysis |
| **Causal Analysis** | Causal tracing, circuit discovery, path patching, bottleneck detection |
| **Safety** | OOD detection, hallucination analysis, safety audits, robustness testing |
| **Probing** | Linear probes, semantic probes (DINO/CLIP), concept discovery, disentanglement metrics |

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

World Model Lens is designed as **observability infrastructure** for world model safety research:

- **Observability First**: Every activation is observable, every decision is traceable
- **Replay Everything**: Replay trajectories with interventions, counterfactuals, and what-if analysis
- **Safety-Centric**: Built-in tools for OOD detection, hallucination analysis, and safety auditing
- **Backend-Agnostic**: Works with ANY world model — RSSM, JEPA, transformers, video prediction, etc.
- **Minimal Interface**: Only `encode()` and `dynamics()` are required; RL-specific features are optional

---

## Why World Model Lens?

World Model Lens provides **production-grade observability** for world model research:

| Use Case | What You Can Do |
|----------|-----------------|
| **Debugging** | Replay failed trajectories, trace error propagation, find bottlenecks |
| **Safety Auditing** | Detect OOD states, hallucinations, belief instability |
| **Mechanistic Understanding** | Causal tracing, circuit discovery, probe for concepts |
| **Benchmarking** | Standardized metrics for disentanglement, probing, causal effects |
| **Research** | Novel MI techniques: SAEs, cross-modal probing, uncertainty estimation |

Whether you're working on:

| Domain | Examples |
|--------|----------|
| **AI Safety** | Reward hacking detection, goal misgeneralization |
| **Reinforcement Learning** | DreamerV3, TD-MPC2, IRIS agents |
| **Video Prediction** | WorldDreamer, video generation models |
| **Planning & Control** | MPC, trajectory optimization |
| **Robotics** | Manipulation, locomotion, navigation |
| **Autonomous Driving** | Camera/LiDAR fusion world models |
| **Scientific Simulation** | Physics, chemistry, climate modeling |

World Model Lens provides the observability tools you need.

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

### Dev environment notes

If you plan to run tests or examples that import PyTorch, install a CPU build of
PyTorch in your virtual environment. Example:

```bash
python -m pip install --upgrade pip
# CPU build
pip install torch
# or pin a version:
pip install "torch==2.2.0"
```
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
from world_model_lens.backends.base_adapter import BaseModelAdapter

class MyWorldModel(BaseModelAdapter):
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

### 2. Observe (Activation Caching)

```python
from world_model_lens import HookedWorldModel
import torch

# Wrap with hooks and caching
wm = HookedWorldModel(adapter=model, config=config)

# Run forward pass - ALL activations are cached
observations = torch.randn(10, 128)  # 10 timesteps
traj, cache = wm.run_with_cache(observations=observations)

# Inspect any activation at any timestep
h_t = cache["h", 5]  # hidden state at timestep 5
z_t = cache["z_posterior", 5]  # latent at timestep 5

# New: Store distributions for uncertainty analysis
import torch.distributions as dist
cache["z_posterior", 5] = dist.Normal(mean, std)
params = cache.get_distribution_params("z_posterior", 5)  # {"mean": ..., "std": ..., "variance": ...}
```

### 3. Replay & Debug

```python
# Replay a specific trajectory
from world_model_lens.branching import ImaginationBrancher

brancher = ImaginationBrancher(wm)
branch = brancher.create_branch(traj, branch_point=5)

# What if we took a different action?
forked = branch.fork()
different_traj = wm.imagine(forked.initial_state, horizon=10, actions=new_actions)

# Compare original vs counterfactual
compare(traj, different_traj)
```

### 4. Safety Audit

```python
# Run safety checks on observed behavior
from world_model_lens.analysis import BeliefAnalyzer

analyzer = BeliefAnalyzer(wm)

# Detect surprise (belief updates)
surprise = analyzer.surprise_timeline(cache)

# Check for hallucinations
hallucination = analyzer.hallucination_detection(traj, imagined_traj)

# OOD detection
ood_scores = analyzer.ood_detection(cache)
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

### 🔍 Observability

| Feature | Description |
|---------|-------------|
| **Activation Caching** | Capture all intermediate activations during forward pass |
| **Hook System** | Intercept and modify forward passes at any layer |
| **Saliency Maps** | Gradient, occlusion, integrated gradients |
| **Surprise Detection** | KL divergence timeline for belief updates |
| **Uncertainty Quantification** | Epistemic & aleatoric uncertainty in latent space |

### 🎬 Replay

| Feature | Description |
|---------|-------------|
| **Trajectory Replay** | Replay stored trajectories with full state inspection |
| **Intervention Replay** | Replay with modified activations (patching) |
| **Imagination Branching** | Fork trajectories for "what-if" exploration |
| **Counterfactual Analysis** | "What if this state/action were different?" |
| **Temporal Debugging** | Step-by-step replay with intervention points |

### 🔬 Safety & Auditing

| Feature | Description |
|---------|-------------|
| **OOD Detection** | Identify out-of-distribution latent states |
| **Hallucination Analysis** | Detect when model predictions diverge from reality |
| **Belief Instability** | Track sudden belief shifts (surprise peaks) |
| **Safety Audit** | Comprehensive safety checks for deployed models |
| **Robustness Testing** | Adversarial perturbation analysis |

### 🧠 Interpretability

| Feature | Description |
|---------|-------------|
| **Linear Probing** | Train probes to decode concepts from latents |
| **Semantic Probes (DINO/CLIP)** | Vision-language concept alignment |
| **Circuit Discovery** | Find important computation subgraphs |
| **Causal Tracing** | Trace causal pathways through the model |
| **Sparse Autoencoders** | Train SAEs for disentangled feature discovery |
| **Disentanglement Metrics** | MIG, DCI, SAP scores |

### 📊 Benchmarks & Metrics

| Feature | Description |
|---------|-------------|
| **Probing Benchmarks** | Standard concept classification tasks |
| **Latent Metrics** | MIG, DCI, SAP, SAP-Score |
| **Causal Benchmarks** | Path patching evaluation |
| **CartPole/Continuous Control** | RL benchmark environments |

### 🛠 Production Tools

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
│   ├── base_adapter.py       # Abstract base class
│   ├── dreamerv3.py         # DreamerV3 implementation
│   ├── dreamerv2.py         # DreamerV2 implementation
│   ├── dreamerv1.py         # DreamerV1 implementation
│   ├── tdmpc2.py            # TD-MPC2 implementation
│   ├── iris.py              # IRIS transformer
│   ├── video_world_model.py # Video prediction adapter
│   ├── toy_video_model.py   # Toy video model for testing
│   └── toy_scientific_model.py # Scientific dynamics models
│
├── analysis/                  # Analysis tools
│   ├── belief_analyzer.py   # Surprise, concepts, saliency, hallucination
│   ├── metrics.py           # Latent space metrics (MIG, DCI, SAP)
│   ├── uncertainty.py       # Belief uncertainty quantification
│   ├── continual_learning.py # Catastrophic forgetting detection
│   └── multimodal.py         # Multi-modal support
│
├── probing/                  # Probing tools
│   ├── prober.py            # Linear/ridge/logistic probing with CV
│   ├── semantic_probes.py  # DINO/CLIP vision-language probes
│   ├── crossmodal.py        # Cross-modal probing
│   ├── geometry.py          # Geometric analysis (PCA, clustering)
│   └── temporal_memory.py  # Memory retention analysis
│
├── patching/                 # Patching tools
│   ├── patcher.py           # Activation patching
│   ├── causal_tracer.py     # Causal tracing
│   ├── circuit.py           # Circuit discovery & comparison
│   ├── dim_patcher.py       # Dimension patching
│   └── sweep_result.py      # Patching sweep results
│
├── sae/                      # Sparse Autoencoders
│   ├── trainer.py           # SAE training with TopK ReLU
│   ├── evaluator.py         # SAE evaluation metrics
│   └── sae.py               # SAE model definition
│
├── branching/                # Branching tools
│   ├── brancher.py          # Imagination branching
│   └── counterfactual.py    # Counterfactual analysis
│
├── safety/                   # Safety analysis
│   ├── analyzer.py          # Safety auditing
│   └── robustness.py        # Robustness testing
│
├── benchmarks/               # Benchmarking
│   ├── suite.py             # Evaluation suite
│   ├── probing.py           # Probing benchmarks
│   ├── cartpole.py          # CartPole benchmark
│   ├── gridworld.py         # Gridworld benchmark
│   ├── continuous_control.py # MuJoCo-style benchmarks
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
├── causal/                   # Causal analysis
│   ├── effect_estimator.py  # Causal effect estimation
│   └── trajectory_attribution.py # Trajectory-level attribution
│
└── visualization/            # Visualization
    ├── latent_plots.py      # Latent space visualizations
    ├── prediction_plots.py  # Prediction visualizations
    └── intervention_plots.py # Patching intervention plots
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
from world_model_lens.backends.base_adapter import BaseModelAdapter

class MyAdapter(BaseModelAdapter):
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

### Observability + Replay Workflow

This example demonstrates the full **observe → replay → debug** cycle:

```python
import torch
from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends.toy_video_model import create_toy_video_adapter

# 1. OBSERVE: Run model with full activation caching
adapter = create_toy_video_adapter(latent_dim=64)
config = WorldModelConfig(d_obs=32*32*3, d_action=4, d_h=64)
wm = HookedWorldModel(adapter=adapter, config=config)

obs = torch.randn(50, 3, 32, 32)  # 50 frame video
actions = torch.randint(0, 4, (50, 4))
traj, cache = wm.run_with_cache(obs, actions)

# 2. INSPECT: Look at any activation, any timestep
print(f"Hidden states shape: {cache['h'].shape}")  # [50, 64]
print(f"Latents shape: {cache['z_posterior'].shape}")  # [50, 64]

# 3. SAFETY CHECK: Detect anomalies
from world_model_lens.analysis import BeliefAnalyzer
analyzer = BeliefAnalyzer(wm)
surprise = analyzer.surprise_timeline(cache)
print(f"Max surprise at timestep: {surprise.max_surprise_timestep}")

# 4. REPLAY WITH INTERVENTION: Patch at specific point
from world_model_lens.patching import TemporalPatcher
patcher = TemporalPatcher(wm)

# What if we intervene at the surprising point?
def intervene(cache):
    patched = cache.clone()
    patched["h", surprise.max_surprise_timestep] = cache["h", 0]  # reset
    return patched

result = patcher.patch_path(cache, intervene)
print(f"Intervention effect: {result.effect}")

# 5. IMAGINATION: Replay forward from new state
new_state = traj.states[surprise.max_surprise_timestep]
imagined = wm.imagine(start_state=new_state, horizon=20)
print(f"Imagined {imagined.length} steps into future")
```

### KV Cache Hook (Transformer Memory Editing)

Transformer-style world models often maintain a growing key/value (KV) memory
that serves as the model's temporal context. World Model Lens exposes a
dedicated hook point named `"kv_cache"` which runs once per timestep and is
passed the full `ActivationCache`. This lets you inspect or mutate a model's
memory (for example, "forgetting" a specific key) without re-running the
entire forward pass.

Hook signature

```python
```python
# Hook signature: fn(cache: ActivationCache, ctx: HookContext) -> None
def my_kv_hook(cache, ctx):
    # prefer helpers rather than touching _store directly
    cache.delete_kv(0, "k", 10)

from world_model_lens import HookPoint
# register the hook for a single timestep or a time slice
hp = HookPoint(name="kv_cache", fn=my_kv_hook, timestep=10)
wm.add_hook(hp)
```
```

Examples

- Create KV entries at a specific timestep:

```python
def create_kv(cache, ctx):
    cache.set_kv(0, "k", ctx.timestep, torch.tensor([1.0, 1.0]))
    cache.set_kv(0, "v", ctx.timestep, torch.tensor([0.5, 0.5]))

wm.add_hook(HookPoint(name="kv_cache", fn=create_kv, timestep=1))
```

- Modify a KV entry created earlier:

```python
def bump_prev(cache, ctx):
    prev = ctx.timestep - 1
    val = cache.get_kv(0, "k", prev, None)
    if val is not None:
        cache.set_kv(0, "k", prev, val + 10.0)

wm.add_hook(HookPoint(name="kv_cache", fn=bump_prev, timestep=2))
```

- Use a time slice to apply a hook across a range of timesteps:

```python
wm.add_hook(HookPoint(name="kv_cache", fn=my_kv_hook, time_slice=[5, 10]))
```

Notes

- `HookPoint.fn` for `kv_cache` hooks receives `(ActivationCache, HookContext)`
  instead of the usual `(tensor, HookContext)`. The hook system accepts this
  pattern and will call your function with the cache when the `kv_cache`
  component fires.
- Currently examples mutate `ActivationCache._store` directly. You may add
  helper methods (e.g., `get_kv`, `set_kv`, `delete_kv`) if you prefer a
  public API that encapsulates naming conventions.

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

### Circuit Discovery

```python
from world_model_lens.patching import CircuitDiscovery

circuit_finder = CircuitDiscovery(wm)
circuit = circuit_finder.discover_circuit(
    cache=cache,
    metric_fn=lambda x: x['z_posterior'].mean(),
    components=['encoder', 'dynamics', 'decoder'],
)
print(f"Found {len(circuit.nodes)} nodes, {len(circuit.edges)} edges")
```

### Sparse Autoencoder Training

```python
from world_model_lens.sae import SAETrainer

trainer = SAETrainer(latent_dim=256, sae_dim=1024)
sae = trainer.train(
    wm=wm,
    cache=cache,
    component='h',
    num_steps=10000,
)

# Evaluate SAE
from world_model_lens.sae import SAEEvaluator
evaluator = SAEEvaluator(sae)
metrics = evaluator.compute_metrics(cache, component='h')
print(f"L0: {metrics['l0']:.2f}, Reconstruct: {metrics['reconstruction_error']:.4f}")
```

### Probing with Cross-Validation

```python
from world_model_lens.probing import LatentProber

prober = LatentProber(seed=42, n_folds=5)
activations = cache['h']  # [T, D]
labels = torch.randint(0, 3, (T,))  # 3 classes

result = prober.train_probe(
    activations=activations,
    labels=labels.numpy(),
    concept_name='speed',
    activation_name='h',
    probe_type='ridge'
)
print(f"Accuracy: {result.accuracy:.3f} ± {result.std:.3f}")
```

### Disentanglement Metrics

```python
from world_model_lens.analysis import BeliefAnalyzer

analyzer = BeliefAnalyzer(wm)
factors = {
    'velocity': torch.randn(100, 1),
    'position': torch.randn(100, 1),
}
result = analyzer.disentanglement_score(cache, factors=factors)
print(f"MIG: {result.scores['MIG']:.3f}, DCI: {result.scores['DCI']:.3f}")
```

### Unified Disentanglement Evaluation Suite

For multi-component evaluation across `context_encoder`, `predictor`, `target_encoder`:

```python
from world_model_lens.analysis.metrics import DisentanglementEvaluationSuite

suite = DisentanglementEvaluationSuite()
result = suite.evaluate_components(
    cache=cache,
    factors=factors,
    components=['context_encoder', 'predictor', 'target_encoder', 'z_posterior'],
    metrics=['MIG', 'DCI', 'SAP']
)

# Per-component scores
for comp, scores in result.component_results.items():
    print(f"{comp} MIG: {scores['MIG']:.3f}")

# Summary across all components
print(f"Overall disentanglement: {result.summary_scores['DCI_disentanglement']:.3f}")
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
  title = {World Model Lens: Interpretability for World Models},
  author = {Bhavith Chandra Challagundla},
  year = {2026},
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
