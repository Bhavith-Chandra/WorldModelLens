# Changelog

All notable changes to World Model Lens will be documented here.

## [Unreleased]

### Changed
- Wired `ModelHub.load()` to dispatch DreamerV3 checkpoints through `DreamerV3Adapter.from_checkpoint()` when a compatible local or downloaded checkpoint path is available.

### Notes
- Registered `dreamerv3-*` hub entries remain `coming_soon` until a known-good public PyTorch checkpoint source is validated for this adapter.

## [0.1.0] - 2024

### Added
- Initial release
- Core abstractions: LatentState, LatentTrajectory, ActivationCache
- Hook system for intercepting activations
- BaseModelAdapter abstract base class
- DreamerV3 backend adapter
- DreamerV2 backend adapter
- IRIS backend adapter
- TD-MPC2 backend adapter
- HookedWorldModel central wrapper class
- Linear probing tools
- Activation patching tools
- Imagination branching
- Belief analyzer (surprise, concepts, saliency)
- Model hub for pretrained models
- Trajectory hub for dataset management
- Episode collector for Gymnasium environments
- CLI with Typer
- Example scripts

### Coming Soon
- Sparse autoencoder training
- Circuit discovery
- Visualization suite
- Web UI
- Pretrained model downloads
