# Causal Analysis with Counterfactual Engine

## Example 10: Counterfactual Engine

**File:** `examples/10_causal_engine.py`

**What this example shows:** How to define interventions, run counterfactual rollouts, compare them against a baseline trajectory, and rank interventions by downstream effect.

## Prerequisites

- Comfort with Examples 03 and 04
- Familiarity with intervention design and trajectory comparison metrics

## Modules Used

- `world_model_lens.causal.CounterfactualEngine`
- `world_model_lens.causal.Intervention`
- `world_model_lens.causal.rollout_comparison`

Related API pages:
- [causal.md](C:\Users\user\Desktop\WorldModelLens\docs\api\causal.md)
- [patching.md](C:\Users\user\Desktop\WorldModelLens\docs\api\patching.md)
- [branching.md](C:\Users\user\Desktop\WorldModelLens\docs\api\branching.md)

## Workflow

1. Record a baseline trajectory.
2. Define one or more interventions.
3. Generate counterfactual trajectories.
4. Measure outcome deltas and divergence over time.
5. Compare interventions side by side.

## How To Run

```bash
python examples/10_causal_engine.py
```

## Expected Output

You should see:

- baseline trajectory summary
- one or more intervention descriptions
- rollout comparison metrics
- divergence traces for selected timesteps
- a table-like comparison across interventions

## What To Inspect

- which timestep the intervention targets
- whether the effect is immediate or delayed
- whether divergence grows monotonically or stabilizes
- whether outcome delta and trajectory divergence tell the same story

## Intervention Types To Look For

- dimension-level interventions
- action interventions
- component-level replacements or patches

## When To Use This Pattern

Use the counterfactual engine when the question is: "What changes downstream if I intervene here, and how large is that effect relative to other interventions?"

## Common Failure Modes

- baseline and counterfactual rollouts are compared over mismatched horizons
- interventions are too large to isolate a useful mechanism
- readers focus on final divergence only and ignore when the effect first appears

## Relationship To Earlier Examples

| Earlier example | Contribution to Example 10 |
|---|---|
| Example 03: patching | local causal intervention mindset |
| Example 04: branching | comparison across multiple futures |
| Example 05: belief analysis | a way to choose informative fork points or target timesteps |

## Next Steps

After this example, the natural next move is to adapt the workflow to your own model and define interventions that match your domain semantics.
