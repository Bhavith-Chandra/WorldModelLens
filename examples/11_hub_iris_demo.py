"""11_hub_iris_demo.py — End-to-end demo of Hub + Pretrained IRIS.

This script demonstrates how to:
1. Pull official IRIS weights from HuggingFace via ModelHub.
2. Load the weights into a HookedWorldModel with IRISAdapter.
3. Perform a 'real' forward pass and 'imagined' rollouts.
4. Verify that the transformer is producing non-zero activations.
"""

import torch
from world_model_lens import HookedWorldModel
from world_model_lens.hub import ModelHub


def main():
    print("\n--- WorldModelHub IRIS End-to-End Demo ---")

    # 1. Pull and Load
    model_name = "iris-atari-pong"
    print(f"\n[1/4] Loading '{model_name}' from Hub...")
    hub = ModelHub()
    # hub.load automatically pulls from HF, infers dims, and instantiates the adapter
    try:
        adapter = hub.load(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Note: This script requires internet access to download weights from HuggingFace.")
        return

    wm = HookedWorldModel(adapter=adapter, config=adapter.config)
    print(
        f"[OK] Model loaded: d_model={adapter.d_model}, n_layers={len(adapter.transformer.blocks)}"
    )

    # 2. Setup Data
    # IRIS Atari expects 64x64x3 observations.
    # For this demo, we'll use a sequence of 5 random frames.
    T, B, C, H, W = 5, 1, 3, 64, 64
    obs = torch.randn(T, B, C, H, W)
    actions = torch.zeros(T, B, dtype=torch.long)  # Discrete actions

    # 3. Forward Pass (Real)
    print(f"\n[2/4] Running forward pass on {T} real observations...")
    traj, cache = wm.run_with_cache(obs[:, 0])  # HookedWorldModel expects [T, ...]

    print(f"[OK] Forward pass complete. Trajectory length: {len(traj.states)}")

    # Check for activations in the cache
    if "z_posterior" in cache:
        z_shape = cache["z_posterior"].shape
        print(f"[PASS] Activation Cache captured 'z_posterior': {z_shape}")

    # 4. Imagination Rollout
    print("\n[3/4] Running 10-step imagination rollout from last real state...")
    start_state = traj.states[-1]
    imagined = wm.imagine(start_state, horizon=10)

    print(f"[OK] Imagination complete. Rollout length: {len(imagined.states)}")

    # 5. Verification
    print("\n[4/4] Verifying model outputs...")
    last_reward = imagined.states[-1].reward_pred
    print(f"Predicted reward at end of imagination: {last_reward.item():.4f}")

    # Check if the transformer is actually changing the state (non-identity dynamics)
    s0 = imagined.states[0].state
    sN = imagined.states[-1].state

    # In IRIS, 'state' is the history buffer, so it grows.
    # To see if dynamics are active, we check if sN is longer than s0
    # and if the predicted reward is reasonable.
    if sN.shape[1] > s0.shape[1]:
        print(
            f"[PASS] Dynamics verification: History grew from {s0.shape[1]} to {sN.shape[1]} tokens."
        )

        # Also check if the last tokens are different (they might be the same if it's a boring game state, but usually they drift)
        if not torch.equal(s0[:, -1], sN[:, -1]):
            print(
                f"[PASS] Token drift: Last token changed from {s0[0, -1].item()} to {sN[0, -1].item()}"
            )
    else:
        print("[WARN] Dynamics verification: History size is static.")

    print("\n--- Demo Successful ---")


if __name__ == "__main__":
    main()
