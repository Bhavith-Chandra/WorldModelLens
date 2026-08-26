# Latent Lens: layer-wise identity emergence in the I-JEPA predictor

**Task 6** — reviewer Question 4: how do you tell an *architectural* property apart from a
*representational* one?

Positional swaps and attention blockades show the predictor depends on the structures it
is built from. They cannot say *when*, along the predictor's depth, a target token stops
being "a position" and becomes "the content at that position". The Latent Lens measures
exactly that, and the architecture supplies its own control.

```bash
python experiments/latent_lens_trajectory.py --model both \
    --mini_checkpoint examples/ijepa/ijepa_mini.pth experiments/ckpt/ijepa_multi_noaug.pth \
    --n_images 64 --n_masks 4
python experiments/latent_lens_trajectory.py --replot   # redraw without re-running
```

Raw numbers: [`results_latent_lens_trajectory.json`](results_latent_lens_trajectory.json).
Figure: [`plots/latent_lens_trajectory.png`](plots/latent_lens_trajectory.png).

---

## 1. Method

I-JEPA's predictor ends with `prediction = predictor_project_back(norm(x))`, mapping the
predictor's 384-d residual width back into the encoder's representation space. The lens
applies **that same head at every predictor layer**, not just the last. Using the model's
own head rather than a fitted probe matters: at the final layer it reproduces the model's
real output exactly, so every intermediate layer is read off the same ruler.

**The control is architectural, not assumed.** At predictor layer 0 a target token is
exactly `mask_token + pos_embed[p]` — identical for every image, carrying position and
nothing else. Layer 0 *is* the purely architectural prior, and everything above it is
representation dynamics. The script asserts this: with a mask shared across images, the
measured spread of layer-0 target tokens is `0.00e+00` for all three models. If that check
ever fails, the trace is reading the wrong tensor.

Two galleries separate the two things a single MSE conflates:

| | question | position | chance |
|---|---|---|---|
| **Gallery A** | which patch *of this image*? | varies | 1/n_patches |
| **Gallery B** | which *image*, at this same patch index? | **fixed** | 1/n_images |

Gallery B holds position constant, so no positional prior can score above chance on it —
it is pure content identity. A third baseline, the **positional prototype**
`t_bar[p] = mean_i t[i, p]`, is the best a position-only predictor could possibly do.

64 images, 4 shared masks (~42 target patches each), 224 px, fp32 on an RTX 3050.

---

## 2. The three models disagree, and that is the result

| model | predictor | output cos | vs prototype | Gallery B: layer 0 → out | verdict |
|---|---|---|---|---|---|
| `ijepa_mini.pth` | 4 layers | +0.043 | **loses** (+0.245) | 1.6% → 1.6% | **MODEL TOO WEAK** |
| `ijepa_multi_noaug.pth` | 4 layers | +0.594 | wins (+0.215) | 1.6% → 27.7% | **AMBIGUOUS** |
| **official ViT-H/14** | 12 layers | +0.695 | wins (+0.363) | 1.6% → **93.3%** | **INFORMATIVE** |

`examples/ijepa/ijepa_mini.pth` is *worse than predicting the average representation at
that position*. A lens over it measures an untrained model, not a representation — which
is a statement about the checkpoint, and worth knowing before any mini-checkpoint result
is quoted. It is greyed out in the figure and excluded below.

### Official ViT-H/14 — identity emerges gradually

| layer | cos_target | cos_centered | MSE | A@1 | A@5 | **B@1** |
|---|---|---|---|---|---|---|
| 0 (mask+position) | 0.091 | 0.004 | 1.490 | 2.0% | 8.9% | **1.6%** |
| 1 | 0.182 | 0.018 | 1.399 | 4.7% | 19.0% | 2.6% |
| 2 | 0.240 | 0.061 | 1.359 | 8.2% | 30.3% | 7.3% |
| 3 | 0.292 | 0.114 | 1.319 | 12.2% | 37.9% | 20.9% |
| 4 | 0.327 | 0.176 | 1.221 | 14.9% | 42.6% | 44.5% |
| 5 | 0.409 | 0.274 | 1.104 | 15.7% | 43.8% | 66.2% |
| 6–9 | 0.441→0.490 | 0.315→0.373 | 1.062→0.980 | 16.2→18.6% | ~46% | 71.8→76.1% |
| 10 | 0.575 | 0.478 | 0.803 | 19.1% | 46.6% | 88.2% |
| 11 | 0.604 | 0.517 | 0.758 | 19.3% | 45.8% | 90.7% |
| **out** | **0.695** | **0.633** | **0.533** | **20.2%** | 45.9% | **93.3%** |
| *prototype* | *0.363* | *0.000* | *0.855* | *8.1%* | *27.8%* | *1.6%* |

* Layer 0 sits at 1.6% against a 1.56% chance floor — the architectural prior, exactly
  where the architecture says it must be.
* Content identity is built in a **hot zone at layers 2–5** (7.3% → 66.2%, more than half
  the total rise), with a distinct second push at layer 9→10 (+12.1 pp). The largest
  single step is 26% of the total rise, so this is genuinely a trajectory rather than one
  jump at the output head.
* `cos_centered` — cosine after removing everything position alone predicts — climbs from
  0.004 to 0.633, reaching half its final value only at layer 7. The content-specific
  component is built later than the raw alignment suggests, because early layers gain
  mostly by reproducing the positional prototype.
* The lens only overtakes the position-only baseline around layer 5 (cos 0.409 vs 0.363).
  **The first third of the predictor is, in representational terms, an elaborate
  positional prior.**

### The 4-layer predictor has no trajectory to observe

`ijepa_multi_noaug.pth` is a real model — it beats the prototype nearly threefold — but its
Gallery B goes 1.6% → **25.5%** → 26.4% → 27.3% → 27.7%. **91% of the entire rise happens
in the first block.** Layers 1–3 add 2.2 pp between them. The pre-registered criterion C2
(largest single step < 80% of the rise) fails, and the verdict is AMBIGUOUS.

So the task's framing — "layer-wise identity emergence across Predictor layers 0 through
3" — was never going to show emergence on this architecture. There is one step, then a
plateau.

### Localisation and identity dissociate

The trained mini answers "which image" at 27.7% (18x chance) while sitting **at chance on
Gallery A** (0.4% vs 0.51%) at every depth. It predicts something image-specific but
patch-generic — the same content vector wherever you ask. The ViT-H does both (A@1 20.2%,
52x chance). A single prediction-MSE number cannot tell these two models apart in the way
that matters; the two galleries can.

---

## 3. Watch-item status

The task's watch item fires on ambiguity. Its status is split:

* **Official ViT-H/14 — not triggered.** All three pre-registered criteria pass. The
  observational trajectory is clean enough to stand on.
* **4-layer mini predictor — triggered.** Single-step emergence (91% of the rise in one
  block) is exactly the ambiguity the watch item anticipated. Before the novelty claim in
  Task 13 is finalised, this needs the causal follow-up the task names: **QK-routing vs
  OV-content decomposition and path patching across the 4 predictor layers**, to establish
  whether block 0 is doing routing or content transport. That is a causal experiment and
  is not attempted here.

---

## 4. Limitations

* **Observational, by construction.** The lens shows what is *readable* at each depth, not
  what is *used*. It cannot substitute for the per-head causal decomposition.
* **Depth is confounded with capacity and training.** The 12-layer predictor that shows a
  graded trajectory is also 631 M parameters trained on ImageNet-1K for 300 epochs; the
  4-layer ones are ~2 M parameters trained on 240 images. This experiment cannot separate
  "deeper predictors build identity gradually" from "better-trained predictors do". A
  depth sweep at fixed capacity and training budget would be needed to claim the former.
* **Logit-lens caveat.** Intermediate layers were never trained to be readable by the final
  head, so early-layer numbers are a lower bound on what those states contain.
* Single dataset (64 ImageNet images), 4 masks, 224 px, one target block per mask.
  Gallery B's chance floor moves with the image count, so B@1 values are only comparable
  within this run.
