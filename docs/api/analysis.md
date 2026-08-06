# Analysis API

Analysis modules for belief tracking, uncertainty, hallucination detection, disentanglement, and representation quality.

## Belief Analyzer

```{eval-rst}
.. automodule:: world_model_lens.analysis.belief_analyzer
   :members:
   :undoc-members:
   :show-inheritance:
```

## Belief Drift

```{eval-rst}
.. automodule:: world_model_lens.analysis.belief_drift
   :members:
   :undoc-members:
   :show-inheritance:
```

## Uncertainty

```{eval-rst}
.. automodule:: world_model_lens.analysis.uncertainty
   :members:
   :undoc-members:
   :show-inheritance:
```

## Hallucination

```{eval-rst}
.. automodule:: world_model_lens.analysis.hallucination
   :members:
   :undoc-members:
   :show-inheritance:
```

## Out-of-Distribution Detection

```{eval-rst}
.. automodule:: world_model_lens.analysis.ood_detection
   :members:
   :undoc-members:
   :show-inheritance:
```

## Disentanglement

```{eval-rst}
.. automodule:: world_model_lens.analysis.disentanglement
   :members:
   :undoc-members:
   :show-inheritance:
```

## Representation

```{eval-rst}
.. automodule:: world_model_lens.analysis.representation
   :members:
   :undoc-members:
   :show-inheritance:
```

## Sparse Autoencoders (SAE)

```{eval-rst}
.. automodule:: world_model_lens.sae.sae
   :members:
   :undoc-members:
   :show-inheritance:
```

### Usage Example

.. code-block:: python

    from world_model_lens.sae.trainer import SAETrainer
    import torch

    # synthetic activations: 1000 samples, 64-d input
    activations = torch.randn(1000, 64)

    # construct a trainer using a named SAE implementation
    trainer = SAETrainer(d_input=64, n_boj=32, k=4, l1_coefficient=1e-3, sae_type="gated")

    # quick one-epoch train (use small epochs in docs/examples)
    result = trainer.train(activations, epochs=5, batch_size=128, progress=False)

    print("final L0:", result.final_l0, "final recon:", result.final_reconstruction_loss)

Note: the trainer normalizes the sparsity penalty by taking the mean absolute
activation (over batch and features). If you are migrating older experiments
that used a sum-based L1 penalty, you may need to retune `l1_coefficient`.

### Fair‑comparison note

Historically some SAE implementations used a summed L1 penalty (e.g. `h.abs().sum()`),
which scales with batch size and the number of features. The trainer now uses
an averaged penalty (`h.abs().mean()`) so the `l1_coefficient` has consistent
meaning across different batch sizes and model widths.

If you need a rough conversion from an old `l1_sum_coeff` to the new mean-based
coefficient, use:

.. code-block:: python

    # approximate conversion (depends on exact averaging axes)
    old_coeff = 1e-3
    batch_size = 128
    n_features = 32
    new_coeff = old_coeff * (batch_size * n_features)

This is only an approximation; we recommend re-tuning `l1_coefficient` on a
small validation set for best results.

### Passing a custom SAE class

If you have a compact SAE class you want to use directly, pass it via
`sae_class`. The trainer will try several common constructor patterns, and
`sae_kwargs` can be used for additional keyword arguments.

.. code-block:: python

    from world_model_lens.sae.sae import TopKSparseAutoencoder
    trainer = SAETrainer(
        d_input=64,
        n_boj=32,
        k=4,
        sae_class=TopKSparseAutoencoder,
        sae_kwargs={"tie_weights": True},
    )

The trainer attempts these constructor styles in order: keyword-style
(`input_dim=.., n_features=.., k=..`), positional (`d_input, n_boj, k`),
simple positional (`d_input, n_boj`), and finally `(config, device)`.

Developer guide:

.. toctree::
   :hidden:

   ../developer/sae_interface.md

## Temporal Attribution

```{eval-rst}
.. automodule:: world_model_lens.analysis.temporal_attribution
   :members:
   :undoc-members:
   :show-inheritance:
```

## Ablation

```{eval-rst}
.. automodule:: world_model_lens.analysis.ablation
   :members:
   :undoc-members:
   :show-inheritance:
```

## Continual Learning

```{eval-rst}
.. automodule:: world_model_lens.analysis.continual_learning
   :members:
   :undoc-members:
   :show-inheritance:
```

## Metrics

```{eval-rst}
.. automodule:: world_model_lens.analysis.metrics
   :members:
   :undoc-members:
   :show-inheritance:
```

## Multimodal

```{eval-rst}
.. automodule:: world_model_lens.analysis.multimodal
   :members:
   :undoc-members:
   :show-inheritance:
```

## Video Metrics

```{eval-rst}
.. automodule:: world_model_lens.analysis.video_metrics
   :members:
   :undoc-members:
   :show-inheritance:
```
