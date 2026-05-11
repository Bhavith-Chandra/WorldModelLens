"""WorldModelLens Documentation.

Core Concepts
=============

1. LATENT SPACE
---------------
The latent space is the learned representation of the world model.
In WorldModelLens, we treat any latent dynamical model uniformly:

    z_t = encode(x_t)       # Encode observation to latent
    z_t+1 = transition(z_t, a_t)  # Evolve latent with action
    x_hat_t = decode(z_t)   # Reconstruct observation

Key terms:
- z_t: Latent state at timestep t
- x_t: Observation at timestep t
- a_t: Action at timestep t
- encode: Encoder network (observation -> latent)
- transition: Dynamics model (latent + action -> next latent)
- decode: Decoder network (latent -> observation)

2. TRANSITIONS
--------------
Transitions are the core of world models - they predict how the world
evolves. In WorldModelLens:

- transition.pre: Hook before transition computation
- transition.post: Hook after transition computation
- z_t+1 = f(z_t, a_t) where f is the transition function

3. CAUSALITY
------------
WorldModelLens provides causal interpretability tools:

- CausalEffectEstimator: Formal A/B testing for world models
- TrajectoryAttribution: Which latent at t affects outcome at T?
- CounterfactualEngine: What if we had done something different?

The causal question: "What is the effect of intervening on X on Y?"

    Effect = E[Y | do(X = x_1)] - E[Y | do(X = x_0)]

4. HOOK SYSTEM
--------------
Three levels of hooks match TransformerLens capabilities:

Spatial: z[0:10]        # Dimension-level (first 10 dims)
Temporal: t=5.z         # Timestep-level (only at t=5)
Transition: transition.pre/post  # Pre/post dynamics

5. INTERPRETABILITY HIERARCHY
-----------------------------
Level 1 → Dimension (neuron-like, finest)
Level 2 → State (latent vector z_t)
Level 3 → Trajectory (sequence segment)

Workflow: Find behavior (L3) -> Examine state (L2) -> Find mechanism (L1)
"""

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends import ToyVideoAdapter
from world_model_lens.causal import CausalEffectEstimator, TrajectoryAttribution
from world_model_lens.visualization import LatentTrajectoryPlotter
from world_model_lens.benchmarks import CartPoleBenchmark

__doc__ = __doc__


def quickstart():
    """Quick start example.

    >>> from world_model_lens import HookedWorldModel, WorldModelConfig
    >>> from world_model_lens.backends import ToyVideoAdapter
    >>>
    >>> # Create world model
    >>> adapter = ToyVideoAdapter(latent_dim=32, hidden_dim=64)
    >>> config = WorldModelConfig(d_h=64, d_obs=64*64*3)
    >>> wm = HookedWorldModel(adapter=adapter, config=config)
    >>>
    >>> # Run forward pass
    >>> import torch
    >>> obs = torch.randn(10, 3, 64, 64)
    >>> traj, cache = wm.run_with_cache(obs)
    >>>
    >>> # Analyze
    >>> analysis = wm.analyze(traj, cache)
    """
    pass


def comparison_with_transformer_lens():
    """Comparison with TransformerLens.

    TransformerLens:
    - Focus: Transformer language models
    - Hooks: Layer-by-layer, attention heads
    - Attribution: Direct logit attribution

    WorldModelLens:
    - Focus: Latent dynamical models (Dreamer, RSSM, etc.)
    - Hooks: Spatial (dim), Temporal (t), Transition (pre/post)
    - Attribution: Causal effects, trajectory-level
    """
    pass
