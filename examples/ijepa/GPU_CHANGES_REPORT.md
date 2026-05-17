# GPU Support Changes — ijepa examples

**Branch:** `ijepa_probing`  
**Commit:** `694239b` — "corrected all GPU bugs"  
**GPU tested on:** NVIDIA GeForce RTX 5090, CUDA 13.0, PyTorch 2.10.0+cu130  
**Patch file:** `gpu_support.patch` (apply with `git apply gpu_support.patch`)

---

## Summary

15 Python files changed: 113 insertions, 74 deletions.  
1 new file added: `dog.jpg` (canonical sample image, 1546×1213 px).

All scripts now auto-detect CUDA at startup and move the model and input
tensors to GPU. If CUDA is unavailable they fall back to CPU transparently.

---

## Key patterns applied

### 1. Auto-device detection (every standalone script)
```python
# Added near top of file, after imports
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 2. Moving the adapter to GPU
The `BaseModelAdapter.to()` override requires `device` as a keyword argument —
passing it positionally raises `TypeError`. Every adapter move uses:
```python
adapter = adapter.to(device=DEVICE)   # keyword-only — required by BaseModelAdapter
```
Plain tensors use the normal positional form:
```python
img_tensor = preprocess_image(raw_img).to(DEVICE)
```

### 3. Utility loaders (utils.py / experiment_utils.py)
Both loader functions already accepted an optional `device` argument but
defaulted to `None` (CPU). Changed to auto-detect when caller passes nothing:
```python
# Before
if device is not None:
    adapter = adapter.to(torch.device(device))

# After
if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
adapter = adapter.to(device=torch.device(device))
```

### 4. Canonical dog image (image_utils.py / utils.py)
`image_utils.get_sample_image()` now checks for a local `dog.jpg` first
before hitting the network, and saves it on first download. `utils.py`'s
synthetic-gradient fallback was replaced with the same local file check.
`attribution_evaluation.py`'s five hard-coded Wikipedia URLs (all returning
403) were replaced with a single call to `get_sample_image()`.

---

## File-by-file changes

### `image_utils.py`
- Added `from pathlib import Path`
- Added module-level `_LOCAL_DOG = Path(__file__).with_name("dog.jpg")`
- `get_sample_image()`: checks `_LOCAL_DOG` first; downloads and caches to
  `dog.jpg` on first run; falls back to synthetic only if download also fails

### `utils.py`
- `load_ijepa_world_model()`: auto-detects CUDA when `device=None`; uses
  `adapter.to(device=...)` keyword form
- `get_sample_image()`: inserted `dog.jpg` check before the synthetic-gradient
  fallback so all scripts share the same image by default

### `experiment_utils.py`
- `load_condition_world_model()`: same auto-detect + keyword-form fix as
  `utils.py`

### `14_ijepa_experiment_suite.py`
- `--device` argparse default changed from `"cpu"` → `None` so auto-detection
  kicks in

### `train_ijepa.py`
- Added `DEVICE`
- `IJEPAAdapter(config).to(device=DEVICE)`
- `img_tensor = preprocess_image(raw_img).to(DEVICE)`

### `attribution_evaluation.py`
- Added `DEVICE`
- `adapter.to(device=DEVICE)` after weight load
- Replaced the 5-URL dataset loop with a single `get_sample_image()` call
  (50 samples, different target patch IDs, same dog image tensor)
- `img_tensor = preprocess_image(raw_img).to(DEVICE)`

### `attribution_graph.py`
- Added `DEVICE`
- Both `visualize_ig_vs_attention()` and `visualize_research_ijepa()`:
  `adapter.to(device=DEVICE)` and `img_tensor.to(DEVICE)`

### `circuit_view.py`
- Added `DEVICE`
- `ThresholdCircuitVisualizer.__init__`: `img_tensor.to(DEVICE)`,
  `adapter.to(device=DEVICE)`

### `counterfactual_analysis.py`
- Added `DEVICE`
- `IJEPACounterfactualAnalyzer.__init__`: `adapter.to(device=DEVICE)`,
  `img_a.to(DEVICE)`, `img_b.to(DEVICE)`

### `formal_circuits.py`
- Added `DEVICE`
- `FormalCircuitDiscoverer.__init__`: `adapter.to(device=DEVICE)`,
  `img_tensor.to(DEVICE)`

### `structural_circuits.py`
- Added `DEVICE`
- `IJEPAStructuralTracer.__init__`: `img_tensor.to(DEVICE)`,
  `adapter.to(device=DEVICE)`

### `causal_evaluator.py`
- Added `DEVICE` inside `if __name__ == "__main__"` block
- `model.to(device=DEVICE)`, `img_tensor.to(DEVICE)`

### `interactive_explorer.py`
- Added `DEVICE`
- `AttributionExplorer.__init__`: `img_tensor.to(DEVICE)`,
  `adapter.to(device=DEVICE)`

### `progressive_build.py`
- Added `DEVICE`
- `animate_ijepa_progressive()`: `img_tensor.to(DEVICE)`,
  `IJEPAAdapter(config).to(device=DEVICE)`

### `surgical_counterfactual.py`
- Added `DEVICE`
- `SurgicalCounterfactual.__init__`: `adapter.to(device=DEVICE)`,
  `img_cat.to(DEVICE)`, `img_dog.to(DEVICE)`

---

## Benchmark results (RTX 5090, dog.jpg, ijepa_mini.pth)

| Script | Time | Peak GPU | Status |
|---|---|---|---|
| `circuit_view.py` | 4.2s | 65.5 MB | OK |
| `formal_circuits.py` | 5.3s | 66.0 MB | OK |
| `causal_evaluator.py` | 14.3s | 68.0 MB | OK |
| `attribution_evaluation.py` | 129.0s | 126.2 MB | OK |
| `12_ijepa_probing.py` | 46.5s | 1408.6 MB | OK |
| `14_ijepa_experiment_suite.py` | 55.3s | 476.5 MB | OK |
| `counterfactual_analysis.py` | 3.5s | 66.4 MB | pre-existing hook bug |
| `structural_circuits.py` | 3.6s | 65.5 MB | pre-existing hook bug |

`12_ijepa_probing.py` peaks at 1.4 GB because it loads DINO ViT-B/16 on top
of the I-JEPA model for semantic alignment probes.  
`attribution_evaluation.py` is slow (2m 9s) due to 50-sample × 50-step
Integrated Gradients — all GPU compute, no CPU bottleneck.

## Known issues not introduced by these changes

- **`counterfactual_analysis.py`** and **`structural_circuits.py`** both
  assert that hooks fire during the forward pass, but the `HookRegistry` does
  not reach the named points (`context_encoder.norm`, `predictor.block_*`).
  This is a pre-existing hook-wiring bug unrelated to GPU migration.
- **`surgical_counterfactual.py`** has hardcoded Windows paths for cat/dog
  images in its `__main__` block — not runnable as-is, but GPU wiring is in
  place.
- **CLIP probes** in `12_ijepa_probing.py` require `transformers` which is
  not installed in this environment — they skip gracefully.

## How to apply locally

```bash
# From the repo root
git apply examples/ijepa/gpu_support.patch

# You also need dog.jpg in the same directory.
# Either copy it from the repo or let image_utils.py download it on first run:
python -c "from examples.ijepa.image_utils import get_sample_image; get_sample_image()"
```
