# Non-RL World Models

These examples show that the same tooling works even when the model is not organized around rewards, value heads, or control signals.

## Example 07: Video Prediction Model

**File:** `examples/07_video_model.py`

**What this example shows:** How to use the library with a video prediction model that only needs observations and latent dynamics.

### Prerequisites

- Comfort with Example 01
- Understanding that some models will not have actions, rewards, or value predictions

### Modules Used

- `world_model_lens.backends.video_adapter.VideoWorldModelAdapter`
- `world_model_lens.HookedWorldModel`

Related API pages:
- [backends.md](C:\Users\user\Desktop\WorldModelLens\docs\api\backends.md)
- [core.md](C:\Users\user\Desktop\WorldModelLens\docs\api\core.md)

### How To Run

```bash
python examples/07_video_model.py
```

### Expected Output

You should see a successful forward pass over a frame sequence, plus cache keys for encoder, dynamics, and decoder stages.

### What To Inspect

- which components are cached for a non-RL model
- whether the adapter exposes enough structure for standard analysis tools
- how frame prediction differs from control-oriented rollout outputs

### Common Failure Modes

- observation tensors are not shaped as the adapter expects
- code assumes action or reward fields that do not exist for this backend

## Example 08: Toy Video World Model Analysis

**File:** `examples/08_toy_video_world_model.py`

**What this example shows:** How to run geometry and temporal-memory analyses on a synthetic video model.

### Prerequisites

- Comfort with Examples 02 and 05
- Basic understanding of PCA, clustering, and temporal dependence

### Modules Used

- `world_model_lens.probing.geometry.GeometryAnalyzer`
- `world_model_lens.probing.temporal_memory.TemporalMemoryProber`
- `world_model_lens.analysis.BeliefAnalyzer`

Related API pages:
- [probing.md](C:\Users\user\Desktop\WorldModelLens\docs\api\probing.md)
- [analysis.md](C:\Users\user\Desktop\WorldModelLens\docs\api\analysis.md)

### How To Run

```bash
python examples/08_toy_video_world_model.py
```

### Expected Output

You should see PCA summary information, trajectory metrics, cluster information, surprise statistics, and temporal memory measurements.

### What To Inspect

- whether latent trajectories form coherent low-dimensional paths
- whether temporal memory decays smoothly or abruptly
- whether surprise spikes align with visual changes in the synthetic video

### Common Failure Modes

- synthetic video generation does not match the model input format
- clustering and manifold estimates are over-interpreted from very small samples

## Example 09: Toy Scientific Dynamics

**File:** `examples/09_toy_scientific_dynamics.py`

**What this example shows:** How to analyze latent dynamics for scientific systems such as Lorenz attractors and pendulums.

### Prerequisites

- Comfort with Example 08
- Some intuition for chaotic vs periodic dynamics

### Modules Used

- `world_model_lens.backends.toy_scientific_model.ToyScientificAdapter`
- `BeliefAnalyzer`
- geometry and temporal-memory tools

Related API pages:
- [backends.md](C:\Users\user\Desktop\WorldModelLens\docs\api\backends.md)
- [analysis.md](C:\Users\user\Desktop\WorldModelLens\docs\api\analysis.md)
- [probing.md](C:\Users\user\Desktop\WorldModelLens\docs\api\probing.md)

### How To Run

```bash
python examples/09_toy_scientific_dynamics.py
```

### Expected Output

You should see summary statistics for the generated trajectories, followed by surprise, geometry, and temporal-memory comparisons between at least two dynamical systems.

### What To Inspect

- whether chaotic systems produce higher surprise than periodic ones
- whether geometry metrics separate attractor-like vs cyclic structure
- whether temporal memory behaves differently across systems

### Common Failure Modes

- the generated trajectories are too short to expose meaningful structure
- numerical scale differences across systems dominate the comparison unfairly

## What Changes In Non-RL Settings

| RL-oriented assumption | Non-RL replacement |
|---|---|
| rewards and value heads exist | only latent or reconstruction metrics may exist |
| actions always drive dynamics | dynamics may be autonomous or partially controlled |
| control quality is the main outcome | prediction fidelity or structure is often the main outcome |

## Next Example

- [causal-analysis.md](C:\Users\user\Desktop\WorldModelLens\docs\examples\causal-analysis.md)
