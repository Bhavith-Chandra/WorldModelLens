# Advanced Analysis

These examples build on the basic cache and intervention workflows to answer higher-level questions about uncertainty, hallucination, saliency, and representation structure.

## Example 05: Belief Analysis

**File:** `examples/05_belief_analysis.py`

**What this example shows:** How to inspect model confidence, detect hallucination-like divergence, and identify dimensions that matter for a target prediction.

### Prerequisites

- Comfort with Examples 01-04
- Basic familiarity with surprise, saliency, and divergence metrics

### Modules Used

- `world_model_lens.analysis.BeliefAnalyzer`
- cached trajectories and imagined trajectories

Related API pages:
- [analysis.md](C:\Users\user\Desktop\WorldModelLens\docs\api\analysis.md)
- [causal.md](C:\Users\user\Desktop\WorldModelLens\docs\api\causal.md)

### How To Run

```bash
python examples/05_belief_analysis.py
```

### Expected Output

You should see:

- mean and peak surprise statistics
- top dimensions from concept search
- saliency tensor shapes
- hallucination severity and flagged timesteps

### What To Inspect

- timesteps where surprise peaks
- whether concept search separates the chosen timestep groups meaningfully
- whether saliency mass is concentrated or diffuse
- whether hallucination detection finds early drift or late drift

### When To Use This Pattern

Use belief analysis when the question is: "Where is the model uncertain, brittle, or internally inconsistent?"

### Common Failure Modes

- surprise peaks are treated as errors rather than candidates for inspection
- concept labels are poorly defined and produce noisy dimension rankings
- hallucination thresholds are chosen without calibrating on baseline trajectories

## Example 06: Disentanglement Analysis

**File:** `examples/06_disentanglement.py`

**What this example shows:** How to measure whether latent dimensions separate different factors of variation.

### Prerequisites

- Comfort with Example 02 style label construction
- Familiarity with MIG, DCI, and SAP as factor-structure metrics

### Modules Used

- `world_model_lens.analysis.BeliefAnalyzer`
- disentanglement scoring utilities inside the analysis package

Related API pages:
- [analysis.md](C:\Users\user\Desktop\WorldModelLens\docs\api\analysis.md)
- [probing.md](C:\Users\user\Desktop\WorldModelLens\docs\api\probing.md)

### How To Run

```bash
python examples/06_disentanglement.py
```

### Expected Output

You should see MIG, DCI, and SAP scores plus a mapping from factors to the most associated dimensions.

### What To Inspect

- whether all factors have some assigned dimensions
- whether one factor dominates many dimensions
- whether scores agree with the qualitative behavior of the model
- whether changes in factor construction alter the ranking substantially

### When To Use This Pattern

Use disentanglement analysis when the question is: "Is the representation organized into stable, interpretable factors?"

### Common Failure Modes

- synthetic factors do not correspond to anything meaningful in the trajectory
- high scores are over-trusted without qualitative checks
- factor scaling or binning choices distort the metric values

## Interpretation Guide

| Signal | What a high value usually means | Follow-up |
|---|---|---|
| Surprise | The model encountered something unexpected | Inspect the underlying state transition |
| Hallucination severity | Imagination diverges sharply from reality | Compare rollout segments before and after divergence |
| Saliency magnitude | Small set of dimensions strongly affects the target | Patch or ablate those dimensions next |
| MIG / DCI / SAP | Factors are more cleanly separated | Validate with probing or interventions |

## Next Examples

- [non-rl-models.md](C:\Users\user\Desktop\WorldModelLens\docs\examples\non-rl-models.md)
- [causal-analysis.md](C:\Users\user\Desktop\WorldModelLens\docs\examples\causal-analysis.md)
