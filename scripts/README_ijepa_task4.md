# I-JEPA Task 4

Edit the `CONFIG` dictionary at the top of `run_ijepa_task4.py`, then execute:

```bash
python scripts/run_ijepa_task4.py
```

The demo uses WorldModelLens-native components:

- `ModelHub.load()` or `ModelHub.load_checkpoint()` for all three official
  I-JEPA modules: context encoder, EMA target encoder, and predictor;
- the adapter's context encoder, target encoder, and predictor for batched
  forward passes;
- native forward hooks on the predictor's exposed `hook_resid_post` modules for
  live activation replacement;
- `world_model_lens.data` for balanced ImageNet sampling, loading, and preprocessing;
- `MahalanobisOODDetector` for activation and prediction OOD diagnostics;
- `LatentProber` for ImageNet-label linear probing.

`HookedWorldModel.run_with_cache()` treats its leading input axis as time. The
demo calls the library adapter directly so ImageNet examples can remain batched
and the ViT-H experiment is practical on a GPU.

## Interventions

Every image uses the same context/target mask so token positions align:

- `zero`: return `zeros_like(layer_activation)`;
- `mean`: return the token-wise activation mean over all sampled images;
- `resample`: return the aligned activation from the next image in the
  deterministic sample order.

The intervention point is the complete post-block predictor residual exposed by
`adapter.predictor.blocks[N].hook_resid_post`. Layer indices are configured
through `TARGET_LAYERS`.

## Run outputs

Each run creates a UTC-stamped directory below `OUTPUT_ROOT` containing:

- `config.yaml`: the exact top-of-file configuration used for the run;
- `dataset_manifest.json`: paths, remapped labels, and ImageNet class names;
- `results.json`: configuration, model metadata, probe results, and summaries;
- `per_sample_metrics.json` and `summary_metrics.json`;
- `prediction_mse.png`, `target_cosine.png`,
  `classification_accuracy.png`, `substitution_maha.png`, and
  `prediction_maha.png`.

Prediction MSE and target cosine are the primary I-JEPA metrics. Classification
is a diagnostic of linear decodability. The two Mahalanobis plots show whether
the replacement activation and resulting prediction lie outside their clean
empirical distributions. Plots are saved and, when `SHOW_PLOTS` is enabled,
displayed during the run.
