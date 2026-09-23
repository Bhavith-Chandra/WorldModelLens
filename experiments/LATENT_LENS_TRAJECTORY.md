# Latent Lens: layer-wise identity emergence in the I-JEPA predictor

**Task 6** — reviewer Question 4: how do you tell an *architectural* property apart from a
*representational* one?

Positional swaps and attention blockades show the predictor depends on the structures it
is built from. They cannot say *when*, along the predictor's depth, a target token stops
being "a position" and becomes "the content at that position". The Latent Lens measures
exactly that, and the architecture supplies its own control.

The measurement has two halves, reported separately because they can disagree:

* **6a — absolute.** Does the projection actually *get near* the one true target
  embedding for that patch? Cosine against `t[i, p]`, no gallery, no alternatives.
* **6b — relative.** Is the true target ranked *above* a gallery of wrong candidates?
  A discrimination test.

A predictor can pass 6b while failing 6a — ranking the right answer first without landing
anywhere near it — so neither number substitutes for the other. On the ViT-H it does
exactly that for the first third of its depth (§2).

```bash
# the traces (GPU; ~50 min for the ViT-H, ~14 min per mini on an RTX 3050)
python experiments/latent_lens_trajectory.py --model vith --slim_dir checkpoints \
    --n_images 240 --n_masks 32 --bootstrap 10000
python experiments/latent_lens_trajectory.py --model mini \
    --mini_checkpoint examples/ijepa/ijepa_mini.pth experiments/ckpt/ijepa_multi_noaug.pth \
    --n_images 240 --n_masks 64 --bootstrap 10000

# every number and figure below, rebuilt from the cached traces (no GPU, seconds)
python experiments/latent_lens_trajectory.py --reanalyze
```

Raw numbers: [`results_latent_lens_trajectory.json`](results_latent_lens_trajectory.json).
Figures: [`plots/latent_lens_trajectory.png`](plots/latent_lens_trajectory.png) (all three
panels), [`plots/latent_lens_absolute_6a.png`](plots/latent_lens_absolute_6a.png) (6a on
its own axes), [`plots/latent_lens_margin_crossing.png`](plots/latent_lens_margin_crossing.png)
(the complete margin curve and where it crosses zero).

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

### 6a — absolute alignment, and why the raw number cannot be read alone

`cos_target` is cosine between the layer-*j* projection and the true `t[i, p]`. It needs
two companions to mean anything:

* **`cos_centered`** — cosine after subtracting the positional prototype
  `t_bar[p] = mean_i t[i, p]` from both sides, isolating the content-specific component.
* **the positional prototype itself** — a "predictor" that never sees the image and
  returns `t_bar[p]`. It is the ceiling of a purely positional account.

This matters because target space has a large shared mean component. At layer 0 the ViT-H
target token carries *zero* image information (spread `0.00e+00`), yet raw cosine already
reads **0.092**; the prototype, still seeing no image, reaches **0.336**. Any vector
pointing into the bulk of the space scores respectably. So the headline 6a claim is not
"cosine reaches 0.724" but **how far the lens gets past the prototype, and at which
depth** — reported as a paired margin, with the crossing localised in §3.

### 6b — the two galleries

| | question | position | chance |
|---|---|---|---|
| **Gallery A** | which patch *of this image*? | varies | 1/n_patches |
| **Gallery B** | which *image*, at this same patch index? | **fixed** | 1/n_images |

Gallery B holds position constant, so no positional prior can score above chance on it —
it is pure content identity.

### Sampling and uncertainty

240 ImageNet images, 224 px, fp32 on an RTX 3050. Masks are shared across images: 32 for
the ViT-H, 64 for the 4-layer models. That is 332k–502k scored tokens per lens point.

95% CIs come from **10000 bootstrap replicates resampling clusters, not tokens**: one
image contributes `n_masks × n_target_patches` rows from the same encoder pass, so
token-level resampling would treat hundreds of thousands of correlated rows as independent
samples and report intervals that are far too tight. The **same resampled indices are
reused at every layer**, which makes layer-to-layer comparisons paired — the interval on
"layer 4 → layer 5" is an interval on that increment, not the gap between two independent
intervals.

Two resampling units are reported, because they answer different questions:

* **images** — "would another 240 images say the same?" Masks held fixed.
* **images × masks** — a two-way cluster bootstrap over the crossed design; "would another
  draw of images *and* masks say the same?" This is the interval a claim needs if it is
  meant to generalise past the particular masks that were drawn, and it is wider by
  construction. It is only computed when there are enough mask clusters to estimate the
  component (`--min_masks_two_way`, default 8).

Bootstrap cost is ~2 s. The trace is the expensive part, so its sufficient statistics are
cached in a ~1 MB `.npz`; `--reanalyze` rebuilds every number and figure below without
touching a model.

### Multiple comparisons

The trajectory asks the same question of every layer, so the tests arrive as **families**:
12 consecutive-layer increments on the ViT-H, 13 per-layer margins over the prototype.
Twelve tests at α = 0.05 uncorrected carry a **46% chance of at least one false positive**,
so "every step is significant" is not a claim pointwise intervals can support. Each family
is therefore reported with:

* **Holm–Bonferroni** adjusted p-values — familywise control under arbitrary dependence,
  which paired increments certainly have. This is the column a family-level claim reads off.
* **Benjamini–Hochberg** adjusted p-values — false-discovery control, the more forgiving
  standard, shown so the gap between the two is visible.
* a **simultaneous sup-t band** — the smallest `c` for which `Δ ± c·se` covers the whole
  family at 95%. It exploits the correlation between layers instead of assuming Bonferroni's
  worst case, so it is the tightest interval that still supports a statement about the whole
  trajectory. (`c = 2.85` for the ViT-H's 12 increments, against 2.87 for Bonferroni.)

A percentile bootstrap p-value floors at `2/(B+1)`, so B is what a corrected analysis
spends: Bonferroni over 12 tests compares against 0.05/12 = 0.0042, which B = 1000
(floor 0.002) can only just distinguish. B = 10000 (floor 0.0002) clears it comfortably,
and is why the default was raised.

---

## 2. The three models disagree, and that is the result

| model | predictor | 6a: output cos | 6a: margin over prototype | 6b: Gallery B, layer 0 → out | verdict |
|---|---|---|---|---|---|
| `ijepa_mini.pth` | 4 layers | +0.005 | **−0.220** (loses) | 0.4% → 0.4% | **MODEL TOO WEAK** |
| `ijepa_multi_noaug.pth` | 4 layers | +0.610 | +0.411 | 0.4% → 17.7% | **INFORMATIVE**\* |
| **official ViT-H/14** | 12 layers | +0.724 | +0.388 | 0.4% → **92.7%** | **INFORMATIVE** |

\* passes, but §5 explains what its margin of passing actually is.

`examples/ijepa/ijepa_mini.pth` is *worse than predicting the average representation at
that position*, at every depth, with all five margin CIs entirely negative (output
`[−0.251, −0.189]`, Holm p ≤ 1e-3). Its Gallery B never leaves chance and **0 of 4**
increments are significant even uncorrected. A lens over it measures an untrained model,
not a representation. It is greyed out in the figures and excluded below.

### Official ViT-H/14 — the complete trajectory

240 images × 32 masks, 331,680 tokens per lens point, B = 10000.

| layer | cos_target | 95% CI | cos_centered | margin vs proto | 95% CI | B@1 | 95% CI |
|---|---|---|---|---|---|---|---|
| 0 (mask+position) | 0.092 | [0.090, 0.095] | 0.003 | **−0.244** | [−0.249, −0.240] | 0.4% | [0.3, 0.6] |
| 1 | 0.177 | [0.174, 0.180] | 0.020 | **−0.160** | [−0.164, −0.156] | 0.9% | [0.6, 1.2] |
| 2 | 0.241 | [0.238, 0.245] | 0.076 | **−0.095** | [−0.099, −0.091] | 4.2% | [3.7, 4.8] |
| 3 | 0.303 | [0.299, 0.307] | 0.140 | **−0.034** | [−0.038, −0.029] | 16.8% | [15.4, 18.2] |
| **4** | 0.343 | [0.338, 0.347] | 0.211 | **+0.006** | [+0.001, +0.012] | 45.5% | [43.8, 47.1] |
| 5 | 0.431 | [0.427, 0.436] | 0.319 | **+0.095** | [+0.089, +0.101] | 70.7% | [69.4, 71.9] |
| 6 | 0.466 | [0.461, 0.471] | 0.363 | **+0.130** | [+0.123, +0.136] | 75.4% | [74.2, 76.6] |
| 7 | 0.486 | [0.482, 0.491] | 0.392 | **+0.150** | [+0.144, +0.156] | 78.0% | [76.9, 79.1] |
| 8 | 0.508 | [0.504, 0.513] | 0.420 | **+0.172** | [+0.166, +0.178] | 78.8% | [77.8, 79.8] |
| 9 | 0.515 | [0.510, 0.519] | 0.423 | **+0.178** | [+0.172, +0.185] | 77.8% | [76.8, 78.8] |
| 10 | 0.603 | [0.598, 0.609] | 0.529 | **+0.267** | [+0.259, +0.275] | 88.4% | [87.6, 89.1] |
| 11 | 0.634 | [0.628, 0.641] | 0.567 | **+0.298** | [+0.289, +0.307] | 90.6% | [89.9, 91.2] |
| **out** | **0.724** | [0.718, 0.731] | **0.678** | **+0.388** | [+0.379, +0.397] | **92.7%** | [92.2, 93.3] |
| *prototype* | *0.336* | *[0.332, 0.341]* | *0.000* | — | — | *0.4%* | *[0.3, 0.5]* |

**The 6a headline: the predictor does not beat a position-only baseline until it is about
a third of the way up.** Layers 0–3 are not merely unimpressive on 6a, they are
*significantly worse* than the prototype — every margin CI in that range lies entirely
below zero at Holm-adjusted p ≤ 2.6e-3, under both resampling units. **The first third of
the predictor is, in absolute terms, a more elaborate positional prior than simply
averaging.**

Layer 4 is the boundary case and the two resampling units disagree about it, which is
worth stating plainly rather than picking the flattering one:

| | layer 4 margin | Holm p | clears? |
|---|---|---|---|
| resampling images (these 32 masks) | +0.006 [+0.001, +0.012] | 0.020 | yes |
| resampling images **and** masks | +0.006 [−0.003, +0.016] | 0.197 | **no** |

So "the first layer that clears the prototype" is layer 4 conditional on this mask set and
layer 5 if the claim is meant to generalise over mask draws. That flip is exactly why §3
reports a crossing depth instead: the discrete statistic sits on a knife edge, while the
interpolated crossing is 3.84 under both units and excludes layer 5 under both.

That is a stronger statement than 6b alone supports, and it is the reason 6a was worth
adding. On Gallery B, layer 3 already scores 16.8% against a 0.42% floor — 40× chance,
which reads like real content — while its absolute alignment is still *below* what
position alone achieves. **The predictor learns to discriminate before it learns to
arrive.** A relative test cannot see that; an absolute one can.

---

## 3. Where the margin crosses zero

"Negative at layer 3, positive at layer 4" localises the crossing only to a whole block,
and states it in terms of which depths happened to be sampled. The crossing is instead
interpolated *inside each bootstrap replicate* — well defined because the replicates are
paired across layers, so within one replicate the whole curve moves together — and the
percentiles of that distribution give an interval on the crossing depth itself.

| | crossing depth | relative depth |
|---|---|---|
| resampling images | **layer 3.84**, 95% CI [3.71, 3.98] | 0.320, [0.310, 0.331] |
| resampling images **and masks** | layer 3.84, 95% CI [3.62, 4.04] | 0.320, [0.302, 0.337] |

100% of replicates cross. Right-hand panel of
[`plots/latent_lens_margin_crossing.png`](plots/latent_lens_margin_crossing.png).

**The crossing is the better-behaved statistic**, and §2 shows why. "The first layer whose
CI clears zero" answers 4 or 5 depending on whether masks are resampled, because layer 4's
margin is +0.006 and the answer turns on a hair. The crossing does not move at all between
the two units — 3.84 either way, with an interval that excludes layer 5 in both — because
it uses the whole curve rather than one test at one sampled depth. A discrete "first
significant layer" is a coarse readout of a continuous quantity, and it inherits all the
instability of thresholding.

**This corrects the earlier reading of "layer 5".** At 4 masks the margin at layer 4 came
out at −0.022 and the first CI to clear zero was layer 5's. At 32 masks layer 4 is
+0.006 and the crossing sits at 3.84. The estimate was mask-sensitive precisely where it
mattered, because the crossing falls in the steep part of the curve — an eighth of a block
of movement relocates it. That is an argument for sampling the mask axis, not against the
measurement: the four-mask interval was conditional on four masks and never claimed
otherwise, but nothing in the earlier writeup made the sensitivity visible.

The 4-layer `ijepa_multi_noaug` also crosses, at layer 0.33 `[0.29, 0.37]` — but that
number should be read as **"before layer 1", not as a resolved depth**. Its margin goes
−0.190 → +0.382 in a single block, and with no lens point inside block 0 the interpolation
is a straight line drawn across one enormous step. The ViT-H crossing is meaningful because
there are measured lens points on both sides of it.

---

## 4. Every layer-to-layer increment, corrected

All 12 ViT-H increments survive **Holm** correction, on both metrics, under both
resampling units. Largest Holm-adjusted p in the family is 2.4e-3 — which is the *floor*,
12 × 2/(B+1), not a measured value, so every step sits at the bootstrap's resolution limit.

| family (12 tests) | uncorrected | Holm | BH | sup-t simultaneous |
|---|---|---|---|---|
| Gallery B@1, images | 12/12 | **12/12** | 12/12 | 12/12 |
| Gallery B@1, images × masks | 12/12 | **12/12** | 12/12 | 12/12 |
| cos_centered, images | 12/12 | **12/12** | 12/12 | 12/12 |

| step | Δ B@1 | 95% CI | sup-t simultaneous | p Holm |
|---|---|---|---|---|
| 0→1 | +0.5 pp | [+0.2, +0.7] | [+0.1, +0.9] | 2.4e-3 |
| 1→2 | +3.3 pp | [+3.0, +3.8] | [+2.8, +3.9] | 2.4e-3 |
| 2→3 | +12.6 pp | [+11.6, +13.5] | [+11.1, +14.0] | 2.4e-3 |
| **3→4** | **+28.7 pp** | [+27.7, +29.7] | [+27.2, +30.2] | 2.4e-3 |
| **4→5** | **+25.2 pp** | [+24.3, +26.1] | [+23.9, +26.5] | 2.4e-3 |
| 5→6 | +4.7 pp | [+4.5, +5.0] | [+4.3, +5.2] | 2.4e-3 |
| 6→7 | +2.6 pp | [+2.4, +2.9] | [+2.3, +3.0] | 2.4e-3 |
| 7→8 | +0.8 pp | [+0.5, +1.0] | [+0.4, +1.1] | 2.4e-3 |
| **8→9** | **−1.0 pp** | [−1.2, −0.8] | [−1.3, −0.7] | 2.4e-3 |
| 9→10 | +10.5 pp | [+10.1, +11.0] | [+9.9, +11.2] | 2.4e-3 |
| 10→11 | +2.2 pp | [+2.1, +2.3] | [+2.0, +2.4] | 2.4e-3 |
| 11→out | +2.2 pp | [+2.0, +2.4] | [+1.9, +2.4] | 2.4e-3 |

The rise concentrates in a hot zone at layers 3–5 (+28.7, +25.2 pp) with a second push at
9→10 (+10.5 pp). The small **regression at 8→9** (−1.0 pp) survives correction too, so it
is a real dip rather than the noise a point estimate would leave ambiguous. Largest single
step = 31% of the total rise, CI [30, 32]% resampling images and [29, 33]% resampling
masks as well — so this is genuinely a trajectory, not one jump at the head.

The one place correction bites is the 13-test **margin** family, where `cos_centered` at
layer 0 (+0.0026) is significant uncorrected and fails Holm at p = 0.089 — leaving 12/13.
That is the correct outcome: layer 0 carries no image information by construction, so a
margin there should not be readable, and the uncorrected interval was claiming one.
(Gallery B's margin at layer 0 also fails, at p = 0.99, but that test was never near
significance — the prototype and layer 0 are both exactly at chance there.)

---

## 5. The 4-layer predictor: resolved, not ambiguous

`ijepa_multi_noaug.pth` is a real model — margin over prototype +0.411 `[+0.375, +0.446]` —
but its Gallery B goes 0.4% → **13.7%** → 15.6% → 16.8% → 17.7%, and on 6a the picture is
starker still: `cos_centered` goes −0.004 → **0.546** → 0.565 → 0.572 → 0.575.

The pre-registered C2 criterion asks whether the largest single-layer increment is under
**80%** of the total rise. Previous runs left this unresolved: at 4 masks the fraction read
0.786 with a CI reaching 0.852, so the registered PASS did not survive its own uncertainty
and the verdict was recorded as AMBIGUOUS. Sixteen times the mask sampling gives:

| | fraction | 95% CI | P(below 0.80) |
|---|---|---|---|
| 4 masks (previous) | 0.786 | [0.722, 0.852] | — |
| **64 masks, images** | **0.767** | [0.717, 0.813] | **92.0%** |
| **64 masks, images × masks** | 0.767 | [0.717, 0.815] | **90.8%** |

**The criterion passes, at roughly 9:1 odds.** The strict CI-bound test still fails, and
will keep failing: the quantity genuinely sits about one standard error from the threshold,
so no feasible amount of extra data on a 240-image pool settles a *binary* at that
distance. What the bootstrap can state exactly is how its mass falls, and reporting that is
a better answer than a shrug — it says which way the evidence points and how strongly.
The JSON records it as `c2_probability_below_threshold`, alongside the unchanged registered
fields.

The substantive finding does not depend on 77-vs-80 at all, and is now unambiguous in both
directions:

* **Block 0 does the overwhelming majority of the work** — 77% of the rise, 13.3 pp of a
  17.3 pp total.
* **It is not the only thing happening.** All four increments survive Holm correction under
  both resampling units: 1→2 (+1.9 pp), 2→3 (+1.2 pp), 3→out (+0.9 pp) are each small but
  real. The earlier run's "the 3→out step is the only increment whose CI includes zero" was
  a four-mask artefact; with 64 masks that step is solidly non-zero.

So the task's framing — "layer-wise identity emergence across Predictor layers 0 through 3"
— still finds one step and then a shallow but genuine slope, rather than the graded
trajectory the ViT-H shows. The causal follow-up the task names (**QK-routing vs OV-content
decomposition and path patching**) remains the right way to ask *what* block 0 is doing.
It is no longer needed to establish *that* the observational verdict is a pass.

### Localisation and identity dissociate

The trained mini answers "which image" at 17.7% (42× chance) while sitting **at chance on
Gallery A** (0.48% vs 0.51%) at every depth. It predicts something image-specific but
patch-generic — the same content vector wherever you ask. The ViT-H does both (A@1 25.4%
against a 0.39% floor, 65× chance). A single prediction-MSE number cannot tell these two
models apart in the way that matters; the two galleries can.

---

## 6. Watch-item status

The pre-registered criteria are evaluated on **point estimates and left exactly as
registered** — rewriting a threshold after seeing the interval is what pre-registration
exists to prevent. The bootstrap is reported alongside as a robustness check.

* **Official ViT-H/14 — not triggered.** All three criteria pass, and all three still pass
  at their CI bounds under both resampling units (C1 lower bound 92.2% against an 0.83%
  requirement; C2 upper bound 32% resampling images, 33% resampling masks too, against an
  80% threshold; 100% of replicates below threshold). Clean, with room to spare.
* **4-layer mini — passes, with the margin of passing quantified.** C1 and C3 pass at their
  CI bounds. C2 passes on the point estimate (0.767) and holds in 92.0% of replicates
  (90.8% with masks resampled), but its interval still touches the threshold, so
  `verdict_robust_to_ci` remains `false` and `verdict_if_ci_bounds_used` remains
  `AMBIGUOUS`. Those fields are unchanged and deliberately conservative; the probability is
  the number to read.
* **`ijepa_mini.pth` — MODEL TOO WEAK**, more clearly than before: with 64 masks its
  Gallery B does not leave chance at any depth and 0 of 4 increments are significant even
  without correction.

---

## 7. Limitations

* **Observational, by construction.** The lens shows what is *readable* at each depth, not
  what is *used*. It cannot substitute for the per-head causal decomposition.
* **Depth is confounded with capacity and training.** The 12-layer predictor that shows a
  graded trajectory is also 631 M parameters trained on ImageNet-1K for 300 epochs; the
  4-layer ones are ~2 M parameters trained on 240 images. This experiment cannot separate
  "deeper predictors build identity gradually" from "better-trained predictors do". A
  depth sweep at fixed capacity and training budget would be needed to claim the former.
* **Logit-lens caveat.** Intermediate layers were never trained to be readable by the final
  head, so early-layer numbers are a lower bound on what those states contain. This cuts
  specifically against the 6a claim in §2: "layers 0–3 are worse than the prototype" is a
  statement about readability through the final head, not proof those states lack content.
* **The crossing is an interpolation between lens points.** It is only as meaningful as the
  spacing around it — trustworthy on the 12-layer predictor, not on a 4-layer one where the
  sign change happens inside the first block (§3).
* **The bootstrap covers images and masks, not dataset or seed.** Both sampling axes are now
  resampled, but intervals remain conditional on this 240-image ImageNet subset, one target
  block per mask, 224 px, and seed 0.
* **B@1 is not comparable across runs with different `n_images`.** Gallery B's chance floor
  is `1/n_images`, so a 64-image run and a 240-image run are not measuring the same task.
  `cos_target` has no such dependence and is the safer cross-run check.
