# Getting Started with Examples

## Example 01: Quickstart

**File:** `examples/01_quickstart.py`

**What this example shows:** The minimum end-to-end workflow for wrapping a model, collecting activations, inspecting cached tensors, and rolling the model forward in imagination mode.

## Prerequisites

- Base installation of `world_model_lens`
- No optional extras required
- Familiarity with `HookedWorldModel`, `WorldModelConfig`, and tensor shapes

## Modules Used

- `world_model_lens.HookedWorldModel`
- `world_model_lens.WorldModelConfig`
- `world_model_lens.backends.dreamerv3.DreamerV3Adapter`

Related API pages:
- [core.md](C:\Users\user\Desktop\WorldModelLens\docs\api\core.md)
- [backends.md](C:\Users\user\Desktop\WorldModelLens\docs\api\backends.md)

## What It Does

1. Builds a model config and backend adapter.
2. Wraps the backend in `HookedWorldModel`.
3. Runs a forward pass with caching enabled.
4. Reads a few cached components by name and timestep.
5. Starts an imagination rollout from a state in the recorded trajectory.

## How To Run

```bash
python examples/01_quickstart.py
```

## Expected Output

You should see:

- configuration summary
- model and wrapper initialization logs
- trajectory length
- cache component names
- sample tensor shapes for one or two cached activations
- an imagination rollout summary

Exact numeric values can vary if the example uses random inputs.

## What To Inspect

Focus on these objects after the run:

- `traj.length`: confirms the forward pass produced a full trajectory
- `cache.component_names`: confirms which activations were captured
- `cache["h", 0]`: a single hidden state slice
- `cache["z_posterior"]`: a full component across all timesteps
- `imagined`: confirms the model can continue from latent state alone

## Why It Matters

This is the base pattern used by almost every other example:

1. run a trajectory
2. collect cache
3. extract components
4. analyze or intervene

If this example is unclear, the rest of the examples will be harder to reason about.

## Common Failure Modes

- Missing backend dependencies: verify the package installed correctly from the repo root.
- Shape mismatches between observations and actions: check `d_obs` and `d_action` in the config.
- Empty or unexpected cache contents: inspect backend hook names and cached component registration.

## Next Examples

- [analysis-techniques.md](C:\Users\user\Desktop\WorldModelLens\docs\examples\analysis-techniques.md) for probing, patching, and branching
- [advanced-analysis.md](C:\Users\user\Desktop\WorldModelLens\docs\examples\advanced-analysis.md) once you are comfortable reading cache outputs
