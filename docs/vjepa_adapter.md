# V-JEPA adapter status

`VJEPAAdapter` is the hookable video transformer copied from
`feat/mae-comparison`. It can run small configurations and expose
activations, but it is **not yet a faithful port of Meta's pretrained V-JEPA
v1 model**.

For Meta's ViT-L architecture defaults, construct `VJEPAAdapter()` or pass
`WorldModelConfig.vjepa_vitl16()`. A plain `WorldModelConfig(backend="vjepa")`
retains the shared config's generic transformer values.

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

1. **Architecture defaults.** `WorldModelConfig.vjepa_vitl16()` and
   `VJEPAAdapter()` now use Meta's ViT-L/16 sizes: 1024 channels, 24 encoder
   blocks, 16 encoder heads, 384 predictor channels and 12 predictor blocks
   with 16 heads. Tubelet, crop and patch sizes also match. Encoder QKV bias
   and LayerNorm epsilon match Meta's ViT-L builder. These defaults do not
   establish checkpoint compatibility by themselves.
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
[ViT-L pretraining config](https://github.com/facebookresearch/jepa/blob/main/configs/pretrain/vitl16.yaml),
[model builder](https://github.com/facebookresearch/jepa/blob/main/app/vjepa/utils.py),
[encoder](https://github.com/facebookresearch/jepa/blob/main/src/models/vision_transformer.py),
[predictor](https://github.com/facebookresearch/jepa/blob/main/src/models/predictor.py).
