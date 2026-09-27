# V-JEPA adapter

`VJEPAAdapter()` uses the ViT-L/16 architecture sizes from Meta's V-JEPA v1
pretraining configuration. It exposes encoder and predictor activations through
World Model Lens hooks.

## Official checkpoint

The ModelHub entry uses Meta's [ViT-L/16 checkpoint](https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar):

```python
from world_model_lens.hub import ModelHub

adapter = ModelHub.load("vjepa-vit-l-224", device="cpu")
```

`pull()` obtains the file. `load()` then strips the official
`module.backbone.` prefixes, maps Meta's encoder and predictor module names,
and strictly loads the context encoder, separate EMA target encoder, and
predictor. Missing, unexpected, or incompatible tensors raise an error.
The predictor preserves both checkpoint mask tokens and the checkpoint's fixed
3D positional embeddings.

For a video batch `[B, 3, 16, 224, 224]` and explicit context and target
patch indices, use `adapter.predict_masked(video, context_ids, target_ids)`.
The index tensors may have shape `[N]` for a shared mask or `[B, N]` for a
different mask per video. `mask_index` selects one of the two checkpoint mask
tokens. The default `encode()` then `dynamics()` path still uses a simple half
split when no masks are supplied; pass masks for meaningful predictions.

The loader is tested with small checkpoints that use Meta's state-dict layout,
including distinct target weights and strict missing-key failures. A full
forward comparison with Meta's implementation and the 5.1 GB checkpoint has
not yet been run, so numerical parity on the published weights remains to be
verified.

Sources: [Meta V-JEPA repository](https://github.com/facebookresearch/jepa),
[ViT-L pretraining config](https://github.com/facebookresearch/jepa/blob/main/configs/pretrain/vitl16.yaml),
[encoder](https://github.com/facebookresearch/jepa/blob/main/src/models/vision_transformer.py),
[predictor](https://github.com/facebookresearch/jepa/blob/main/src/models/predictor.py).
