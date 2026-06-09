# Pull Request Summary: DreamerV3 Hub Dispatch Preparation

This log reflects the current state of the branch honestly: it prepares
`ModelHub` to load DreamerV3 adapters from compatible checkpoints, but it does
not yet provide a known-good public DreamerV3 Hugging Face checkpoint and does
not fully close issue #12 by itself.

## New Behavior

* **DreamerV3 adapter dispatch in `ModelHub.load()` (`world_model_lens/hub/model_hub.py`)**
  * Added a DreamerV3 branch in `ModelHub.load()` that calls `DreamerV3Adapter.from_checkpoint(...)`, moves the adapter to the requested device, sets eval mode, and returns the adapter.
  * This aligns DreamerV3 with the hub loading path once a compatible checkpoint path is available.

## Test Coverage

* **DreamerV3 hub dispatch test (`tests/test_hub.py`)**
  * Added a fixture that saves a real `DreamerV3Adapter` state dict to a temporary checkpoint file.
  * Added a test that mocks `ModelHub.info()` and `ModelHub.pull()` so `ModelHub.load()` exercises the new DreamerV3 branch and returns a `DreamerV3Adapter` in eval mode.

* **DreamerV3 registry guard test (`tests/test_hub.py`)**
  * Added a test confirming that registered `dreamerv3-*` entries still raise `NotImplementedError` while marked `coming_soon=True`.

## Docs / Notes

* **Changelog note (`CHANGELOG.md`)**
  * Documented that DreamerV3 adapter dispatch is now wired into `ModelHub.load()`.
  * Documented that public `dreamerv3-*` hub entries remain blocked until a known-good public PyTorch checkpoint source is validated.

## Important Scope Note

* **What this branch does not do**
  * Does not add a validated public DreamerV3 checkpoint source.
  * Does not make `dreamerv3-atari-*` downloadable through `ModelHub.pull()`.
  * Does not yet satisfy the original "clone-and-run pretrained DreamerV3 from hub" goal of issue #12.
