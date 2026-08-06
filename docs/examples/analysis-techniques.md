# Core Analysis Techniques

These three examples introduce the main interpretability workflows: probing for encoded information, patching for causal influence, and branching for alternative futures.

## Example 02: Linear Probing

**File:** `examples/02_probing.py`

**What this example shows:** How to test whether latent activations encode specific concepts by training simple probes.

### Prerequisites

- Comfort with Example 01
- Basic understanding of supervised classification metrics

### Modules Used

- `world_model_lens.probing.LatentProber`
- cached activations from `HookedWorldModel`

Related API pages:
- [probing.md](C:\Users\user\Desktop\WorldModelLens\docs\api\probing.md)
- [core.md](C:\Users\user\Desktop\WorldModelLens\docs\api\core.md)

### How To Run

```bash
python examples/02_probing.py
```

### Expected Output

You should see per-concept accuracy values such as:

- `reward_region: accuracy=...`
- `novel_state: accuracy=...`
- `high_value: accuracy=...`

The exact numbers may vary with synthetic data, but the output should clearly show which concepts are easier or harder to decode from the chosen activation.

### What To Inspect

- which activation component is being probed
- how labels are constructed
- whether accuracies are meaningfully above chance
- whether flattening / preprocessing changes probe performance

### When To Use This Pattern

Use probing when the question is: "What information is present in this representation?"

### Common Failure Modes

- labels do not align with the number of activation rows
- highly imbalanced labels make accuracy misleading
- strong probe performance is mistaken for causality rather than correlation

## Example 03: Activation Patching

**File:** `examples/03_patching.py`

**What this example shows:** How to intervene on model internals by replacing corrupted activations with clean activations and measuring recovery.

### Prerequisites

- Comfort with cached runs from Example 01
- Understanding of clean vs corrupted comparisons

### Modules Used

- `world_model_lens.patching.TemporalPatcher`
- cached clean and corrupted runs

### Sparse Autoencoder Note

This repository includes several Sparse Autoencoder (SAE) implementations used
for disentangling latent features. See the API page for details and examples:

- `docs/api/analysis.md` -> Sparse Autoencoders (SAE)
  - Example usage and notes: `docs/api/analysis.md`


Related API pages:
- [patching.md](C:\Users\user\Desktop\WorldModelLens\docs\api\patching.md)
- [core.md](C:\Users\user\Desktop\WorldModelLens\docs\api\core.md)

### How To Run

```bash
python examples/03_patching.py
```

### Expected Output

You should see recovery scores for components and timesteps, with a short ranked list of the most important patches.

### What To Inspect

- how the corrupted input is constructed
- which components are patched
- the metric used to score recovery
- which timesteps dominate the top-k patch results

### When To Use This Pattern

Use patching when the question is: "Which component or timestep is causally responsible for the behavior difference?"

### Common Failure Modes

- clean and corrupted runs are not aligned in shape or length
- the metric function is too noisy or too weak to distinguish effects
- recovery rates are interpreted without comparing to the original corruption severity

## Example 04: Imagination Branching

**File:** `examples/04_branching.py`

**What this example shows:** How to fork a trajectory from a real state and compare multiple imagined futures.

### Prerequisites

- Comfort with `wm.imagine(...)`
- Understanding of latent divergence metrics

### Modules Used

- `world_model_lens.HookedWorldModel.imagine`
- trajectory states from a recorded run

Related API pages:
- [branching.md](C:\Users\user\Desktop\WorldModelLens\docs\api\branching.md)
- [core.md](C:\Users\user\Desktop\WorldModelLens\docs\api\core.md)

### How To Run

```bash
python examples/04_branching.py
```

### Expected Output

You should see divergence statistics across multiple branches, usually reported as mean or max distances over time.

### What To Inspect

- the choice of fork timestep
- how action sequences are varied across branches
- whether divergence grows, plateaus, or collapses
- whether some branches remain clustered while others drift sharply

### When To Use This Pattern

Use branching when the question is: "What futures does the model consider plausible from this state?"

### Common Failure Modes

- branch comparisons use mismatched horizons
- divergence is measured on incompatible state tensors
- branch spread is over-interpreted without considering random action noise

## Comparing The Three

| Example | Main question | Signal type | Typical output |
|---|---|---|---|
| 02 | What is encoded? | Correlational | Probe accuracy |
| 03 | What causes the effect? | Interventional | Recovery score |
| 04 | What futures are plausible? | Counterfactual / rollout | Divergence curves |

## Next Examples

- [advanced-analysis.md](C:\Users\user\Desktop\WorldModelLens\docs\examples\advanced-analysis.md)
- [causal-analysis.md](C:\Users\user\Desktop\WorldModelLens\docs\examples\causal-analysis.md)
