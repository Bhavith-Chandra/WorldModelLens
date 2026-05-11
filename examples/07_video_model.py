"""Example: Video Prediction World Model.

This demonstrates that World Model Lens works with ANY world model,
not just RL agents. Video prediction models are first-class citizens.

A video prediction model:
- Encodes video frames into latent states
- Predicts future frames using dynamics
- Has no reward, value, or action predictions
- Is fully supported by World Model Lens interpretability tools
"""

import torch
import numpy as np

from world_model_lens import HookedWorldModel
#from world_model_lens.core.config import WorldModelConfig
from world_model_lens.backends.generic_adapter import WorldModelConfig
from world_model_lens.backends.video_adapter import VideoWorldModelAdapter
from world_model_lens.visualization import plot_video_model_dashboard
import matplotlib.pyplot as plt
import pathlib

OUTPUT_DIR = pathlib.Path("assets/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 60)
    print("World Model Lens - Video Prediction Example")
    print("=" * 60)
    print("\nThis demonstrates non-RL world models with World Model Lens!")

    frame_shape = (3, 64, 64)
    d_obs = 3 * 64 * 64
    d_state = 256

    config = WorldModelConfig(
        d_state=d_state,
        d_obs=d_obs,
        is_discrete=False,
        model_type="video_prediction",
        name="video_predictor",
    )
    print(f"\n[1] Config: d_state={d_state}, frame_shape={frame_shape}")

    video_model = VideoWorldModelAdapter(
        config=config,
        d_obs=d_obs,
        frame_shape=frame_shape,
    )
    print("\n[2] Video world model created")

    wm = HookedWorldModel(adapter=video_model, config=config, name="video_model")
    print("\n[3] Wrapped in HookedWorldModel")

    T = 10
    frames = torch.randn(T, *frame_shape)
    print(f"\n[4] Created video sequence: {frames.shape}")

    traj, cache = wm.run_with_cache(frames)
    print(f"\n[5] Forward pass complete!")
    print(f"    Trajectory length: {traj.length}")
    print(f"    Cache keys: {cache.component_names}")

    print("\n[6] Predicting future frames...")
    current_frame = frames[0]
    n_pred = 5
    preds, states = video_model.predict_next_frame(current_frame, n_frames=n_pred)
    print(f"    Predicted frames shape: {preds.shape}")
    print(f"    Latent states: {len(states)} states, first state shape: {states[0].shape if states else 'N/A'}")

    print("\n[7] Running imagination (latent rollout)...")
    start_state = traj.states[0].state
    imagined_states, rewards = wm.adapter.imagine(
        start_state=start_state.unsqueeze(0),
        horizon=10,
    )
    print(f"    Imagined {len(imagined_states)} future states")

    print("\n[8] Plotting video model dashboard...")
    fig = plot_video_model_dashboard(traj, frames, preds, output_path=OUTPUT_DIR / "video_model_dashboard.png")
    plt.show()

    print("\n" + "=" * 60)
    print("Video prediction world model works with World Model Lens!")
    print("No RL components (reward, value, action) needed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
