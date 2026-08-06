# Introduction

## What are World Models?

A **world model** is a neural network that learns to predict how an environment evolves over time.
Rather than mapping observations directly to actions, a world model builds an internal representation
of the world and uses it to simulate future states — enabling planning, imagination, and reasoning
about consequences before acting.

At their core, world models learn two things:

- **Encoding**: compress a raw observation (image, sensor reading, frame) into a compact latent state `z`
- **Dynamics**: given the current latent state and an action, predict the next latent state

```
obs_t  ──encoder──▶  z_t  ──dynamics(a_t)──▶  z_t+1  ──decoder──▶  obs_t+1
```

This makes world models powerful: once trained, the model can "imagine" trajectories entirely in
latent space without interacting with the real environment. Prominent examples include
[DreamerV3](https://arxiv.org/abs/2301.04104), [TD-MPC2](https://arxiv.org/abs/2310.16828),
[IRIS](https://arxiv.org/abs/2209.00588), and video prediction architectures.

---

## The Problem: World Models are Black Boxes

Despite their capabilities, world models are largely opaque. Questions that are difficult to answer today:

- **What does the model "know" at each timestep?** Which concepts are encoded in the latent state?
- **Why did a rollout diverge?** At which point did imagined trajectories start disagreeing with reality?
- **Is the model safe?** Does it hallucinate rewards? Does it generalise to out-of-distribution states?
- **What circuits implement a behaviour?** Which components causally determine a prediction?

These are not just research curiosities — they matter for deploying world models in safety-critical
settings and for understanding whether a model has learned the right causal structure.

---

## What is World Model Lens?

World Model Lens is a **backend-agnostic interpretability and observability library** for world models,
inspired by [TransformerLens](https://github.com/neelnanda-io/TransformerLens) for language models.

It wraps any world model in a `HookedWorldModel` that exposes:

- **Activation caching** — capture every intermediate representation during a forward pass
- **Hook system** — intercept and modify activations at any point
- **Analysis tools** — probing, patching, causal tracing, safety checks, and more

The library is designed around a minimal interface: any model that implements `encode()` and
`dynamics()` works. RL-specific features (actions, rewards) are optional.

---

## How it Works

### 1. Wrap your model

```python
from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends.base_adapter import BaseModelAdapter

class MyModel(BaseModelAdapter):
    def encode(self, obs, context=None):
        z = self.encoder(obs)
        return z, z

    def dynamics(self, state, action=None):
        return self.transition(state)

config = WorldModelConfig(d_obs=128, d_h=64)
wm = HookedWorldModel(adapter=MyModel(config), config=config)
```

### 2. Run with caching

```python
observations = torch.randn(20, 128)  # 20 timesteps
traj, cache = wm.run_with_cache(observations)

# Inspect any activation at any timestep
z_5 = cache["z_posterior", 5]
h_5 = cache["h", 5]
```

### 3. Analyse

```python
from world_model_lens.probing import LatentProber
from world_model_lens.patching import TemporalPatcher
from world_model_lens.causal import CausalEffectEstimator

# What concepts are encoded in the latent?
prober = LatentProber()
result = prober.train_probe(cache["z_posterior"], labels, concept_name="position")

# What happens if we patch a specific activation?
patcher = TemporalPatcher(wm)
patch_result = patcher.patch_state(clean_cache, corrupt_cache, patch_component="h", patch_at_timestep=5)

# Formal causal effect of an intervention
estimator = CausalEffectEstimator(wm)
effect = estimator.estimate(intervention, trajectory)
```

---

## Key Concepts

**`WorldState`** — a single timestep: latent state, optional action, optional reward, optional metadata.

**`WorldTrajectory`** — a sequence of `WorldState` objects representing a full episode or rollout.

**`ActivationCache`** — a dict-like store keyed by `(component_name, timestep)` holding every
cached tensor from a forward pass. Supports storing full distributions for uncertainty analysis.

**`HookPoint`** — a named location in the forward pass where a function can intercept and modify
the activation passing through it.

**`HookedWorldModel`** — the central wrapper. Adds caching and hooks to any adapter without
changing its behaviour.

See the {doc}`glossary` for a full reference of terms used across the library.

---

## What's Included

| Module | What it does |
|---|---|
| `core` | `WorldState`, `WorldTrajectory`, `ActivationCache`, hook system |
| `backends` | Adapters for DreamerV2/V3, TD-MPC2, IRIS, video models, and a generic base |
| `analysis` | Belief analysis, surprise detection, disentanglement metrics |
| `probing` | Linear probes, geometry analysis, temporal memory probing |
| `patching` | Activation patching, causal tracing, circuit discovery |
| `causal` | Formal causal effect estimation and trajectory attribution |
| `sae` | Sparse autoencoder training and evaluation |
| `safety` | OOD detection, hallucination analysis, safety auditing |
| `branching` | Imagination branching and counterfactual rollouts |
| `visualization` | Latent space plots, prediction visualizations, intervention maps |
| `cli` | `wml` command-line interface |

---

## Next Steps

- {doc}`installation` — install the library and build docs locally
- {doc}`getting-started` — a minimal working example
- {doc}`api/index` — full API reference
