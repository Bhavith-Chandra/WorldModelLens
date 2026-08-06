# WorldModelLens Glossary

Terminology for mechanistic interpretability of world models.

## Core Concepts

### Activation Cache
A storage mechanism that captures intermediate activations during a forward pass through a world model. Each activation is keyed by `(component_name, timestep)`, enabling detailed inspection of what information flows through the model at each processing step.

**Example:**
```python
cache = ActivationCache()
cache["z_posterior", 0] = tensor  # Store latent at t=0
latent = cache["z_posterior", 0]  # Retrieve it

# New: Store distributions for uncertainty analysis
import torch.distributions as dist
cache["z_posterior", 0] = dist.Normal(mean, std)
params = cache.get_distribution_params("z_posterior", 0)  # {"mean": ..., "std": ..., "variance": ...}
```

### Hook Point
A location in the model's forward pass where a custom function can intercept and modify activations. Hooks enable causal interventions like activation patching.

**Example:**
```python
hook = HookPoint(name="h", fn=my_hook_fn, timestep=5)
wm.add_hook(hook)
```

### Latent State
The internal representation produced by a world model's encoder. For discrete latents, this is typically represented as `n_cat` categorical variables with `n_cls` classes each. For continuous latents, this is a vector in latent space.

### World Model
A model that learns a predictive representation of environment dynamics. Unlike policy/value networks, world models predict: `z_t+1 = f(z_t, a_t)` and optionally reconstruct observations.

### Trajectory
A sequence of latent states (and optionally actions/rewards) representing an episode or imagined rollout.

## Interpretability Techniques

### Probing / Linear Probe
A linear model trained to decode a target property (e.g., position, reward) from latent activations. High probe accuracy indicates the latent encodes that property.

**Example:**
```python
prober = LatentProber(seed=42)
result = prober.train_probe(
    latents,
    labels,
    concept_name="position",
    activation_name="z_posterior",
)
print(f"Accuracy: {result.accuracy}")
```

### Activation Patching
A causal intervention technique where activations at one location are replaced with activations from another run (e.g., clean vs corrupted). Recovery rate measures causal importance.

**Example:**
```python
result = patcher.patch_state(
    clean_cache,
    corrupt_cache,
    patch_component="h",
    patch_at_timestep=10,
    metric_fn=metric_fn,
    clean_obs_seq=clean_obs_seq,
    clean_action_seq=clean_action_seq,
)
print(f"Recovery: {result.recovery_rate}")
```

### Patch Recovery
The fraction of performance restored when patching a component. High recovery (>0.5) suggests the component is causally important for the metric.

### SAE (Sparse Autoencoder)
An autoencoder trained to decompose latent representations into interpretable features. SAE features can then be analyzed for causal roles.

### Logit Lens
A technique (adapted for world models) that projects latent states to prediction heads (reward/action) to understand what information is available at each layer/timestep.

### Path Patching
An extension of activation patching that traces causal paths through multiple timesteps, testing how information flows from source to target components.

## World Model Specific

### RSSM (Recurrent State Space Model)
The Dreamer family architecture with: `h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})` and `z_t ~ q(z_t|h_t, o_t)`.

### Posterior / Prior
The posterior `q(z_t|h_t, o_t)` is the encoder's belief given observations. The prior `p(z_t|h_t)` is the model's prediction before seeing observations. KL divergence measures surprise when they diverge.

### Imagination / Rollout
Using the dynamics model to generate future trajectories without environment interaction. Key for model-based RL.

### Belief State
The hidden state `h_t` that accumulates information over time, serving as the model's "belief" about the unobserved world state.

### Planning Horizon
The number of timesteps into the future where model predictions remain accurate. Beyond this horizon, imagined rollouts diverge from reality.

### Overcommitment
When a world model's value predictions don't update after surprising events, indicating the model is "locked into" a belief despite evidence against it.

## Benchmarking

### Toy World Model
A tiny synthetic model with known planted circuits (e.g., position encoder, reward-gating unit). Used to test interpretability tools on ground truth.

### Circuit Discovery
The process of identifying computational subgraphs that implement specific behaviors (e.g., "this GRU unit gates reward prediction").

### Universality
The degree to which discovered circuits/probes transfer across different models (adapters, checkpoints, architectures) trained on the same task.

### Deceptive World Model
A model that predicts rewards accurately but encodes wrong causal structure - a safety concern where the model appears correct but reasons incorrectly.

## Analysis Metrics

### Disentanglement Score
Measures how well latent factors correspond to independent ground-truth variables. High disentanglement means each latent dimension encodes one factor.

### Causal Strength
The recovery rate from patching, indicating how much a component/timestep contributes to the output.

### Belief Drift
The accumulation of KL divergence over imagined rollouts, measuring how quickly the model's beliefs diverge from reality.

## Additional Terms

- **Hook Registry**: Collection of registered hooks in a HookedWorldModel
- **TensorStore**: Shared storage for large tensors across trajectory datasets
- **LatentTrajectoryLite**: Memory-efficient trajectory that stores indices into TensorStore
- **ReproConfig**: Configuration capturing all random seeds for reproducibility
