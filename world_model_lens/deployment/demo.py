"""Gradio demo for WorldModelLens."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.core.activation_cache import ActivationCache


def create_demo(
    model_path: Optional[str] = None,
    config: Optional[WorldModelConfig] = None,
    title: str = "WorldModelLens Demo",
) -> gr.Blocks:
    """Create Gradio demo interface.

    Args:
        model_path: Optional path to model checkpoint.
        config: WorldModelConfig for the model.
        title: Title for the demo.

    Returns:
        Gradio Blocks interface.
    """
    if config is None:
        config = WorldModelConfig()

    wm = None
    if model_path:
        try:
            wm = HookedWorldModel.from_checkpoint(model_path, "dreamerv3", config)
        except Exception:
            pass

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown("""
        Upload a world model checkpoint or use a demo model to explore:
        - Activation caching and inspection
        - Probing for feature decoding
        - Patching for causal analysis
        """)

        with gr.Tab("Activation Cache"):
            with gr.Row():
                obs_input = gr.Textbox(
                    label="Observations (comma-separated, space-separated timesteps)",
                    placeholder="0.1, 0.2, 0.3 | 0.4, 0.5, 0.6 | ...",
                    lines=5,
                )
                component_dropdown = gr.Dropdown(
                    label="Component",
                    choices=["h", "z_posterior", "z_prior", "reward"],
                    value="h",
                )

            analyze_btn = gr.Button("Run Analysis")
            output_cache = gr.JSON(label="Cache Summary")

            analyze_btn.click(
                fn=lambda obs, comp: {"component": comp, "status": "ready"},
                inputs=[obs_input, component_dropdown],
                outputs=output_cache,
            )

        with gr.Tab("Probing"):
            with gr.Row():
                concept_input = gr.Textbox(
                    label="Concept Name", placeholder="position, velocity, reward..."
                )
                probe_type = gr.Dropdown(
                    label="Probe Type", choices=["linear", "ridge", "logistic"], value="ridge"
                )

            train_probe_btn = gr.Button("Train Probe")
            probe_output = gr.JSON(label="Probe Results")

            train_probe_btn.click(
                fn=lambda concept, ptype: {
                    "concept": concept,
                    "probe_type": ptype,
                    "status": "ready",
                },
                inputs=[concept_input, probe_type],
                outputs=probe_output,
            )

        with gr.Tab("Patching"):
            with gr.Row():
                source_comp = gr.Textbox(label="Source Component", placeholder="h")
                target_comp = gr.Textbox(label="Target Component", placeholder="reward")

            patch_btn = gr.Button("Run Patching")
            patch_output = gr.JSON(label="Patching Results")

            patch_btn.click(
                fn=lambda src, tgt: {"source": src, "target": tgt, "status": "ready"},
                inputs=[source_comp, target_comp],
                outputs=patch_output,
            )

        with gr.Tab("Visualization"):
            gr.Markdown("Visualization tools coming soon!")
            gr.Markdown("""
            - Activation heatmaps
            - Trajectory t-SNE/UMAP
            - Circuit diagrams
            - Probe weight visualizations
            """)

    return demo


def launch_demo(
    model_path: Optional[str] = None,
    config: Optional[WorldModelConfig] = None,
    share: bool = False,
    port: int = 7860,
) -> None:
    """Launch the Gradio demo.

    Args:
        model_path: Optional model checkpoint path.
        config: WorldModelConfig.
        share: Whether to create a public share link.
        port: Port to run on.
    """
    demo = create_demo(model_path, config)
    demo.launch(server_port=port, share=share)


class ToyModelDemo:
    """Demo with built-in toy models for quick testing."""

    def __init__(self):
        self.config = WorldModelConfig(d_h=64, n_cat=4, n_cls=8, d_obs=32)

    def create_interface(self) -> gr.Blocks:
        return create_demo(config=self.config, title="WorldModelLens Toy Demo")

    def launch(self, share: bool = False):
        demo = self.create_interface()
        demo.launch(share=share)


if __name__ == "__main__":
    demo = ToyModelDemo()
    demo.launch(share=True)
