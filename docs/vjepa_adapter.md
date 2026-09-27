# V-JEPA adapter status

`VJEPAAdapter` is the hookable video transformer copied from
`feat/mae-comparison`. It can run its own small checkpoints and expose
activations, but it is **not yet a faithful port of Meta's pretrained V-JEPA
v1 model**.

## Official checkpoint

Meta publishes the [V-JEPA v1 model zoo](https://github.com/facebookresearch/jepa#model-zoo),
including the [ViT-L/16 224px checkpoint](https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar).
The `vjepa-vit-l-224` ModelHub entry points to that file:

```python
from world_model_lens.hub import ModelHub

path = ModelHub.pull("vjepa-vit-l-224")
```

`ModelHub.load("vjepa-vit-l-224")` intentionally raises before downloading
because the adapter cannot load the official weights faithfully. The local
`vjepa_mini.pth` used in the feature branch is a small project checkpoint,
not Meta's pretrained model. V-JEPA 2 is a separate model family and its
checkpoints do not apply here.

## Compatibility gaps

1. **Encoder size and operations.** The adapter defaults to 768 channels,
   12 blocks and 12 heads. Meta's ViT-L/16 uses 1024 channels, 24 blocks and
   16 heads, with biased QKV projections and LayerNorm epsilon `1e-6`.
   Passing only the current default config cannot load the official weights.
2. **Position embeddings.** Meta uses fixed 3D sine/cosine embeddings for the
   encoder and predictor. The adapter initializes trainable position
   embeddings at random. Even if checkpoint keys and tensor shapes were
   mapped, missing or mishandled embeddings would change predictions.
3. **Target encoder.** Meta checkpoints contain a separate `target_encoder`
   produced by EMA training. The feature-branch loader copied `encoder`
   into both encoders, discarding Meta's target weights. This branch now
   rejects such checkpoints explicitly until separate target loading works.
4. **Predictor and masks.** Meta uses `mask_tokens` and predictor modules
   named `predictor_blocks`, `predictor_norm`, and `predictor_proj`.
   The adapter uses one `mask_token`, different state-dict names, and a
   simplified list-of-indices mask interface. Its current key remapping is
   insufficient for Meta's checkpoint.
5. **Verification.** The existing adapter test covers tubelet dimensions and
   a hook on a randomly initialized small model. It does not compare outputs
   against Meta's implementation or load an official checkpoint.

A faithful loader needs to infer the model configuration from the official
checkpoint, map every tensor including the separate target encoder, preserve
Meta's positional embeddings and predictor masks, load strictly, and compare
outputs with the reference implementation on the same preprocessed video.

Sources: [Meta V-JEPA README](https://github.com/facebookresearch/jepa/blob/main/README.md),
[encoder](https://github.com/facebookresearch/jepa/blob/main/src/models/vision_transformer.py),
[predictor](https://github.com/facebookresearch/jepa/blob/main/src/models/predictor.py).
