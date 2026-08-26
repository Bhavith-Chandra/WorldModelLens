# Attribution Patching on the pretrained I-JEPA ViT-H/14

**Task 5** — reviewer Questions 3 & 4: can the causal analysis in this repo be run on the
*official* Meta ViT-H/14 checkpoint, rather than on 18 MB mini-checkpoints?

Short answer: **yes, and it costs 0.5 s per image pair instead of 87 s** — but the
first-order approximation is only trustworthy for the site families it is actually
suited to, and this experiment measures where the line falls.

Run it with:

```bash
python scripts/download_ijepa_weights.py --dest checkpoints/
python experiments/attribution_patching_vith14.py \
    --n_pairs 12 --validate --validate_pairs 4 --validate_heads 64 --ig_steps 8
```

Raw numbers: [`results_attribution_patching_vith14.json`](results_attribution_patching_vith14.json).
Plots: [`plots/`](plots/).

---

## 1. What was run

| | |
|---|---|
| Checkpoint | `IN1K-vit.h.14-300e.pth.tar`, 10.36 GB, epoch 299, official Meta release |
| Encoder | ViT-H/14, d=1280, 32 layers, 16 heads, **630.8 M parameters** |
| Predictor | d=384, 12 layers, 16 heads, 22.4 M parameters |
| Key coverage | context_encoder 389/389, target_encoder 389/389, predictor 152/152 tensors |
| Data | 13 ImageNet images from `data/mini_train`, 224 px, 12 clean/corrupt pairs |
| Hardware | RTX 3050 Laptop (4 GB), fp32, torch 2.12 |
| Sites scored | **836** — 512 encoder heads, 96 encoder block sites, 192 predictor heads, 36 predictor block sites |

### Metric

I-JEPA's own objective, sign-flipped so that higher is better:

```
L(run) = - mean_j || predictor(context)_j - target_encoder(x_clean)_j ||^2 / d
```

over one target block `j` of a structured I-JEPA multi-block mask. The **clean** run
encodes the clean image's context patches; the **corrupted** run encodes a *different*
image's context patches under the identical mask, so tokens align 1:1. Averaged over
12 pairs, `L_clean = -0.551`, `L_corrupt = -1.234`, so the gap the sweep explains is
`0.684`. Every score below is reported as a **fraction of that gap**: 1.0 means
"restoring this one site recovers the clean run entirely".

### Estimators

* **`ap`** — plain attribution patching. One backward pass at the corrupted run;
  score = `(a_clean - a_corrupt) · dL/da`.
* **`ap_ig8`** — the same displacement, but the gradient is averaged over 8 points along
  the straight line from the corrupted input to the clean one (integrated-gradient
  attribution patching, Hanna et al. 2024).
* **ground truth** — real activation patching: substitute the cached clean activation
  into the corrupted forward pass and re-measure `L`. One forward pass per site.

---

## 2. Loading the official checkpoint is not a no-op

Two traps, both silent:

1. This repo's `Block` hardcodes `qkv_bias=False`; the Meta ViT ships
   `blocks.*.attn.qkv.bias` for all 32 blocks.
2. Key names differ — Meta's `mlp.fc1`/`mlp.fc2` vs this repo's `nn.Sequential`
   indices `mlp.0`/`mlp.2`, and `predictor_blocks` / `predictor_norm` /
   `predictor_proj` / `predictor_pos_embed` vs `blocks` / `norm` /
   `predictor_project_back` / `pos_embed`.

A `load_state_dict(..., strict=False)` — which is what `ModelHub._load_ijepa` and
`IJEPAAdapter.from_checkpoint` currently do — therefore leaves **every MLP and every
qkv bias at random initialisation** and reports no error. The loader here remaps the
keys, attaches the biases, and *refuses to run* unless key coverage is exactly 100% in
both directions. The coverage table is written into the results JSON.

**Fitting 631 M parameters on a 4 GB laptop GPU.** Two things mattered, and both are
about ordering rather than precision:

* Only `encoder`, `target_encoder` and `predictor` are needed; the other half of the
  file is optimiser state. They are extracted once into per-component files
  (2.52 / 2.52 / 0.09 GB) that also carry the remapped key names, so later runs load in
  **6.8 s** instead of 81 s.
* A live 10.36 GB private memory-mapping charges against the Windows commit limit that
  the WDDM driver draws on. With that mapping open, CUDA allocation failed at **1.7 GB**
  on a GPU reporting 3.46 GB free — the ViT-H would not fit on an otherwise idle card.
  Building the model *before* opening a mapping, and only holding the (smaller) mapping
  while copying weights in place, fixed it. No mapping is live during the sweep.

A single encoder instance serves both roles: EMA target-encoder weights are loaded to
compute the prediction targets, then context-encoder weights are loaded over them.
That keeps peak memory at one ViT-H rather than two.

---

## 3. Cost

Per image pair, for all 836 sites:

| Estimator | Backward passes | Wall clock | vs. exhaustive |
|---|---|---|---|
| `ap` | 1 | **0.49 s** | **179x faster** |
| `ap_ig8` | 8 | 3.33 s | 26x faster |
| exhaustive activation patching | — | 87.3 s (measured 104 ms/site) | 1x |

The speedup is structural, not incidental: attribution patching is `O(1)` in the number
of sites, exhaustive patching is `O(n)`. Scoring all 512 encoder heads costs the same as
scoring one.

---

## 4. Fidelity — the actual result

The estimate was checked against real activation patching on 196 sites per pair
(all 132 block-level sites, plus 64 heads = 32 highest-scoring + 32 random) across 4
pairs, 784 measurements in total.

**Harness sanity check:** patching `enc.resid_pre.0` with the clean activation recovers
`1.000` of the gap in all 4 pairs, exactly as it must.

| Estimator | Subset | n | Pearson r | Spearman ρ | Sign agr. |
|---|---|---|---|---|---|
| `ap` | all sites | 784 | 0.531 | 0.613 | 0.77 |
| `ap` | **attention heads** | 256 | **0.695** | **0.668** | 0.79 |
| `ap_ig8` | **all sites** | 784 | **0.933** | **0.770** | 0.75 |
| `ap_ig8` | attention heads | 256 | 0.490 | 0.469 | 0.64 |

By site family (`ap_ig8`):

| Family | n | r | ρ |
|---|---|---|---|
| `pred.attn_out` | 48 | +0.954 | +0.789 |
| `pred.mlp_out` | 48 | +0.921 | +0.768 |
| `enc.attn_out` | 128 | +0.782 | +0.525 |
| `enc.mlp_out` | 128 | +0.758 | +0.572 |
| `enc.head` | 256 | +0.490 | +0.469 |
| `enc.resid_pre` | 128 | +0.242 | +0.192 |

**The two estimators fail in opposite directions, and effect size is what separates
them.** Measured true effects: individual heads move the metric by 1.1% of the gap on
average (max 10.5%), while attn/mlp outputs and residual-stream sites move it by 4.9% on
average (max 67.7%).

* **Large perturbations** (residual stream, whole block outputs). A one-point Taylor
  expansion is simply wrong at this magnitude: plain `ap` scatters residual-stream sites
  from −0.7 to +1.0 when the true effect is a fixed +0.55 to +0.79, and even flips sign.
  Averaging the gradient along the path repairs this (r 0.53 → 0.93 overall).
* **Small perturbations** (individual heads). Here the first-order expansion *at the
  corrupted run* is the correct linearisation, because that is the point the patch
  actually starts from. Path-averaging drags the gradient toward the clean input, which
  is the wrong reference — so `ap_ig8` is *worse* than `ap` on heads (r 0.695 → 0.490).

The right-hand panel of `plots/ap_vith14_validation_scatter.png` shows this directly: the
`ap_ig8` cloud hugs y=x, but the orange head points sit in a tight blob near the origin
where the path average buys nothing.

**Practical recommendation for broad ViT-H sweeps:** use **plain `ap` for head-level
sweeps** (r=0.70, ρ=0.67, top-10 overlap 0.70, sign agreement 0.79 — good enough to rank
candidates for follow-up) and **`ap_ig8` when the sites are large** (block outputs,
residual stream). Neither is a substitute for verifying a *shortlist* with real
activation patching, which at 104 ms/site is affordable for the top few dozen sites.

---

## 5. What the ViT-H/14 encoder actually looks like

From the 12-pair sweep (`ap_ig8` for block sites, `ap` for heads — each estimator on the
family it is faithful for):

**Residual stream.** `enc.resid_pre` decays monotonically from 1.10 at layer 0 to 0.42 at
layer 29 — the information the predictor needs is progressively "used up" rather than
concentrated at one depth — then rises to 0.59 at layer 31, where the last blocks write
the representation the predictor consumes.

**Blocks.** Two hotspots and a long quiet middle:

* **Layer 0 MLP: 0.354** of the gap, by far the largest single non-residual site. The
  first MLP is doing something the rest of the network cannot recover from.
* **Layers 27–31**: `mlp_out` climbs 0.077 → 0.161 and `attn_out` climbs to 0.208 at
  layer 31.
* Layers 1–26 contribute |effect| < 0.06 each, mostly < 0.03.

**Heads.** Attribution is sparse and late. Of 512 encoder heads, the top 8 carry 17.7% of
total |attribution| and the top 32 carry 47.9%. Per-head attribution mass by layer peaks
at layers 30–31 (0.157, 0.246) with a secondary bump at 18–21. The strongest and most
reproducible heads (|mean|/std across pairs in brackets) are
`L31.2` [2.15], `L31.8` [2.80], `L31.5` [2.76], `L31.3` [2.33], `L31.0` [1.73],
`L30.10` [1.58], `L31.4` [1.56]. `L11.12` has the single largest mean but is driven by
one pair (|mean|/std = 0.45) and should not be treated as a stable circuit component.

---

## 6. Limitations

* 12 pairs, 4 of them validated. Enough to separate the estimators, not enough to make
  claims about individual heads beyond the stability figures quoted.
* The head validation subset is deliberately biased (32 top-scoring + 32 random per
  pair), so the head correlations describe the regime a sweep actually cares about
  rather than a uniform sample of all 512.
* One corruption type (cross-image context substitution). It is a *large* perturbation
  by construction, which is exactly why the estimator comparison is informative, but a
  milder corruption would flatter plain `ap`.
* One target block per image and a single 224 px resolution.
* `enc.resid_pre` estimates exceed 1.0 (layer 0: 1.10). Restoring the whole residual
  stream can recover at most the full gap, so this is a 10% overshoot of a saturating
  quantity — expected for any linear extrapolation, and the reason those sites are
  reported separately.
