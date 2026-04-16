# Examples

World Model Lens includes 10 example scripts covering the main workflows: collecting activations, probing representations, patching components, branching imagined futures, and running causal comparisons.

## How To Use This Section

Use the examples in sequence if you are new to the library:

1. Start with Example 01 for the basic wrapper + cache workflow.
2. Move to Examples 02-04 for the core interpretability techniques.
3. Continue to Examples 05-06 for higher-level analyses.
4. Use Examples 07-09 if you care about non-RL architectures.
5. Finish with Example 10 for systematic counterfactual workflows.

## Running Examples

All examples live in the `examples/` directory.

```bash
python examples/01_quickstart.py
python examples/02_probing.py
python examples/10_causal_engine.py
```

From the `examples/` directory itself:

```bash
cd examples
python 01_quickstart.py
```

## Example Guides

```{toctree}
:maxdepth: 2

examples/getting-started
examples/analysis-techniques
examples/advanced-analysis
examples/non-rl-models
examples/causal-analysis
```

## Example Index

| Example | File | Topic | Difficulty | Extras | Key modules | Expected output |
|---|---|---|---|---|---|---|
| 01 | `examples/01_quickstart.py` | Basic cache workflow | Beginner | None | `HookedWorldModel`, `DreamerV3Adapter` | Trajectory length, cache keys, sample activation shapes |
| 02 | `examples/02_probing.py` | Linear probing | Beginner | None | `LatentProber` | Probe accuracies by concept |
| 03 | `examples/03_patching.py` | Activation patching | Intermediate | None | `TemporalPatcher` | Recovery scores by component and timestep |
| 04 | `examples/04_branching.py` | Imagination branching | Intermediate | None | `HookedWorldModel.imagine` | Divergence statistics across branches |
| 05 | `examples/05_belief_analysis.py` | Surprise, saliency, hallucination | Intermediate | None | `BeliefAnalyzer` | Surprise peaks, top dimensions, saliency shapes |
| 06 | `examples/06_disentanglement.py` | Factor structure | Intermediate | None | `BeliefAnalyzer`, `DisentanglementEvaluationSuite` | MIG, DCI, SAP, factor-to-dimension assignments across components |
| 07 | `examples/07_video_model.py` | Video prediction | Intermediate | None | `VideoWorldModelAdapter` | Cache contents for frame prediction |
| 08 | `examples/08_toy_video_world_model.py` | Geometry and memory | Intermediate | None | `GeometryAnalyzer`, `TemporalMemoryProber` | PCA, trajectory metrics, memory retention |
| 09 | `examples/09_toy_scientific_dynamics.py` | Scientific dynamics | Intermediate | None | `ToyScientificAdapter`, geometry + belief tools | Surprise and geometry comparisons across systems |
| 10 | `examples/10_causal_engine.py` | Counterfactual engine | Advanced | None | `CounterfactualEngine`, `Intervention` | Intervention tables and divergence metrics |
| 11 | `examples/11_unified_disentanglement.py` | Multi-component evaluation | Intermediate | None | `DisentanglementEvaluationSuite` | Unified MIG/DCI/SAP across context_encoder, predictor, target_encoder |

## Suggested Paths

### New to World Model Lens

- Example 01: basic wrapping and caching
- Example 02: first inspection task
- Example 03: first causal intervention

### Interested In Causal Analysis

- Example 03: patching
- Example 04: branching
- Example 10: counterfactual engine

### Interested In Non-RL Models

- Example 07: video prediction
- Example 08: toy video analysis
- Example 09: scientific dynamics

### Interested In Safety / Reliability Questions

- Example 05: surprise, saliency, hallucination
- Example 10: intervention comparison and divergence tracing

## Common Patterns Across Examples

### Forward pass with activation caching

```python
traj, cache = wm.run_with_cache(obs_seq, action_seq)
h_t = cache["h", 0]
z_t = cache["z_posterior", 0]
```

### Imagination from an existing state

```python
imagined = wm.imagine(start_state=traj.states[5], horizon=20)
```

### Analysis after collection

```python
analyzer = BeliefAnalyzer(wm)
surprise = analyzer.surprise_timeline(cache)
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'world_model_lens'`

Install the package from the repo root:

```bash
pip install -e .
```

### CUDA / memory issues

Reduce batch size, number of timesteps, latent dimension, or image resolution in the example inputs.

### Outputs differ from the docs

Most examples use synthetic/random data. Shapes and workflow should match; exact metric values may vary.

## Related Docs

- [Getting Started](C:\Users\user\Desktop\WorldModelLens\docs\getting-started.md)
- [API Index](C:\Users\user\Desktop\WorldModelLens\docs\api\index.md)
- [Glossary](C:\Users\user\Desktop\WorldModelLens\docs\glossary.md)
