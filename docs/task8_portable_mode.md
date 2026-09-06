# Task 8 runbook: I-JEPA EMA structure and controlled ambiguity

This is the complete runbook for Task 8. The experiment answers two questions:

1. At which depths do I-JEPA's frozen context encoder and EMA target encoder differ when given the same image?
2. Given fixed visible context and three plausible completions of a masked region, does I-JEPA's predictor align with the context-consistent completion, and are I-JEPA target representations less dispersed than a matched supervised ViT-H/14 baseline?

The evaluator measures linear CKA at I-JEPA layers 16, 24, and 32; predictor-to-candidate MSE/cosine alignment; candidate Gaussian differential entropy; and `KL(I-JEPA || ViT-H/14)` after validation-set representation alignment.

## What is and is not a completed result

The portable CIFAR-10/synthetic-bank run is an end-to-end implementation check only. It establishes that checkpoint loading, CKA, masking, predictor alignment, Gaussian metrics, bootstrap code, and PDF export execute correctly. It is **not** an ImageNet-1K or Stable-Diffusion result and must not appear as a paper result.

A reportable Task 8 result requires all of the following:

- Full ImageNet-1K validation data for CKA and representation alignment.
- A real, versioned controlled-ambiguity bank made from ImageNet source images and Stable Diffusion inpaintings.
- Many independent source/mask cases; at least 30 is a pilot floor and 100+ is preferable for submission-facing uncertainty estimates.
- A standard, non-portable cluster run and recorded software/data/checkpoint provenance.

## Relevant files

| File | Purpose | Needed for cluster result? |
| --- | --- | --- |
| `experiments/pvl/task8_ijepa_ema_ambiguity.py` | Main Task 8 evaluator and only experiment entry point. | Yes |
| `world_model_lens/hub/model_hub.py` | Strictly loads the official I-JEPA checkpoint and maps its predictor keys. | Yes |
| `world_model_lens/backends/ijepa_adapter.py` | Adapter architecture matching official I-JEPA ViT-H/14 parameter layout. | Yes |
| `world_model_lens/analysis/ema_structure.py` | Reusable encoder-divergence and candidate-scoring utilities. | Recommended |
| `tests/test_ema_structure.py` | Tests for the reusable EMA analysis utility. | Recommended |
| `scripts/download_ijepa_weights.py` | Downloads and verifies the official I-JEPA checkpoint. | Yes, unless checkpoint is already mounted |
| `scripts/download_task8_smoke_images.py` | Downloads CIFAR-10 images for a local smoke test. | No |
| `scripts/build_task8_smoke_bank.py` | Builds a synthetic bank for a local smoke test. | No |
| `docs/task8_portable_mode.md` | This runbook. | Yes |

The evaluator uses the project model library, specifically `ModelHub` and the I-JEPA adapter. It is independent of the other experimental tasks; it does not require their outputs.

## Prerequisites

Run all commands from the repository root. Use a CUDA-enabled PyTorch environment with `torchvision` installed. A 32 GB GPU can run the standard evaluator; start with batch size 2 or 4. A 24 GB+ GPU is preferred for the full run.

Download the official I-JEPA ImageNet-1K ViT-H/14 checkpoint once if it is not already mounted:

```powershell
venv\Scripts\python.exe scripts\download_ijepa_weights.py --dest weights --verify
```

Expected checkpoint:

```text
weights/vith14_in1k_ep300.pth.tar
```

The evaluator uses Torchvision's `IMAGENET1K_SWAG_LINEAR_V1` ViT-H/14 baseline by default. The first run downloads/caches it. To use an already available local baseline instead, pass:

```text
--baseline-checkpoint /path/to/supervised_vit_h_14_state_dict.pth
```

## Prepare the real data

### ImageNet validation images

Obtain ImageNet-1K validation data through authorized ImageNet access. Extract the validation archive to a mounted directory, for example:

```powershell
mkdir D:\datasets\imagenet\val
tar -xf D:\downloads\ILSVRC2012_img_val.tar -C D:\datasets\imagenet\val
```

The evaluator recursively discovers `.jpg`, `.jpeg`, `.png`, and `.webp` files. Class directories are not required by this script, but the folder must contain the actual ImageNet validation images.

### Controlled-ambiguity inpainting bank

Task 8 does not generate images. Build the bank with the team's approved Stable Diffusion inpainting process before running the evaluator. For every independent source/mask case:

- Select an ImageNet source image and one documented mask.
- Keep visible context and the exact mask fixed.
- Generate exactly three plausible structural inpaintings, varying only the hidden region.
- Save the prompt, negative prompt, model/version, scheduler, guidance scale, steps, seed, resolution, and mask-generation policy in the case metadata or a versioned manifest.
- Do not include the unmodified source as one of the three scientific candidates.

Required layout:

```text
/datasets/task8_inpainting_bank/
  case_0001/
    source.png
    candidate_0.png
    candidate_1.png
    candidate_2.png
    metadata.json
  case_0002/
    ...
```

`metadata.json` must include row-major patch IDs from the I-JEPA 16 x 16 ViT-H/14 patch grid:

```json
{
  "target_patch_ids": [85, 86, 87, 101, 102, 103],
  "context_patch_ids": [0, 1, 2],
  "diffusion_model": "team-approved-model-and-version",
  "seed": 1234
}
```

If `context_patch_ids` is omitted, the evaluator uses the complement of `target_patch_ids`. Context and target IDs must be non-empty, in `[0, 255]`, and non-overlapping.

## Cluster preflight: 32 GB GPU

Do this before committing the full allocation. It verifies real paths, baseline retrieval, non-portable memory use, and bank structure. Use the real mounted ImageNet and real inpainting bank even for this preflight.

```bash
python experiments/pvl/task8_ijepa_ema_ambiguity.py \
  --ijepa-checkpoint /checkpoints/vith14_in1k_ep300.pth.tar \
  --imagenet-val /datasets/imagenet/val \
  --inpainting-bank /datasets/task8_inpainting_bank \
  --n-validation 32 \
  --batch-size 2 \
  --device cuda \
  --log-every 1 \
  --output-dir experiments/pvl/reports/task8_cluster_smoke
```

The preflight succeeds only if it completes without out-of-memory errors and writes `cka_depth.pdf`, `kl_ambiguity_variance.pdf`, and `task8_metrics.json`. If it runs out of memory, reduce `--batch-size` to 1. Do not use `--portable-4gb` on the 32 GB GPU.

## Full cluster run

After the preflight succeeds, omit `--n-validation` so every available ImageNet validation image is used. Start at batch size 2 or 4; retain the largest batch size that is stable.

```bash
python experiments/pvl/task8_ijepa_ema_ambiguity.py \
  --ijepa-checkpoint /checkpoints/vith14_in1k_ep300.pth.tar \
  --imagenet-val /datasets/imagenet/val \
  --inpainting-bank /datasets/task8_inpainting_bank \
  --device cuda \
  --batch-size 4 \
  --log-every 50 \
  --output-dir experiments/pvl/reports/task8_full
```

Do not overwrite smoke output directories. Preserve the full-run logs and output directory unchanged after completion.

## Expected outputs and checks

The output directory contains:

- `cka_depth.pdf`: CKA of frozen context versus EMA target patch representations at layers 16, 24, and 32.
- `kl_ambiguity_variance.pdf`: per-case I-JEPA/baseline entropy, KL, and predictor cosine-distance panels.
- `task8_metrics.json`: raw CKA values, per-case results, bootstrap summary, and execution protocol.

Before interpreting results, verify:

- The JSON has `portable_4gb: false` and the expected full validation count.
- All CKA values are finite and lie in `[0, 1]`.
- Every ambiguity case has exactly three candidates and no candidate/image load failures occurred.
- Entropy/KL values are finite; investigate unusually large KL values by checking covariance regularization and candidate quality.
- The bootstrap count equals the number of independent source/mask cases, not three times that count.

Interpretation must be pre-specified:

- A negative I-JEPA-minus-baseline entropy difference with a 95% bootstrap CI wholly below zero supports a tighter I-JEPA candidate distribution.
- `KL(I-JEPA || baseline)` measures distribution mismatch. It is not, by itself, evidence that I-JEPA is tighter.
- Report the predictor-selected candidate and all per-candidate alignment distances. If a particular candidate is asserted to be context-consistent, establish that label independently of the model score.
- Report both positive and negative results; CKA need not change monotonically across depth.

## Portable 4 GB smoke protocol

Use this only to validate installation and the evaluator pipeline on low-memory hardware. It offloads inactive ViT-H components to CPU, forces batch size 1, defaults to 32 validation images, and allows at most 64.

```powershell
venv\Scripts\python.exe scripts\download_task8_smoke_images.py --n-images 32
venv\Scripts\python.exe scripts\build_task8_smoke_bank.py `
  --images data\task8_cifar10_smoke `
  --dest data\task8_synthetic_smoke_bank `
  --n-cases 3

venv\Scripts\python.exe experiments\pvl\task8_ijepa_ema_ambiguity.py `
  --ijepa-checkpoint weights\vith14_in1k_ep300.pth.tar `
  --imagenet-val data\task8_cifar10_smoke `
  --inpainting-bank data\task8_synthetic_smoke_bank `
  --portable-4gb `
  --device cuda `
  --log-every 1 `
  --output-dir experiments\pvl\reports\task8_portable
```

Portable outputs are implementation artifacts only. Do not commit downloaded data, checkpoints, generated reports, or smoke images to the pull request.

## Reproducibility record

Attach the following to the shared experiment tracker and the final result directory:

- Repository commit SHA and branch name.
- SHA-256/checkpoint verification result for I-JEPA and the exact baseline weight source.
- ImageNet split location/version and number of discovered images.
- Inpainting-bank version, per-case manifest, prompts, masks, seeds, diffusion configuration, and generation code commit.
- Command line, random seed, GPU model/count, CUDA version, PyTorch version, Torchvision version, and wall-clock runtime.
- Number of independent cases, entropy effect estimate, bootstrap CI, predictor-ranking summary, and any excluded/failed cases with reasons.
