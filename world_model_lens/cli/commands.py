"""CLI commands for World Model Lens using Typer.

This CLI supports both RL and non-RL world models:
- RL models: Full analysis including rewards, values, imagination faithfulness
- Non-RL models: Latent geometry, surprise, disentanglement, SAE analysis
"""

from typing import Any, List, Literal, Optional, cast
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from world_model_lens import __version__
from world_model_lens.hub.model_hub import ModelHub
from world_model_lens.hub.weights_downloader import WeightsDownloader
import torch
from world_model_lens.utils.device import get_device
from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.envs import EpisodeCollector

app = typer.Typer(
    name="wml",
    help="World Model Lens: Interpretability toolkit for world models.",
    add_completion=False,
)
console = Console()


@app.command()
def download(
    model: Optional[str] = typer.Argument(
        None,
        help="Model key to download (e.g. iris-atari-breakout). "
        "Omit to download ALL currently available models.",
    ),
    list_ready: bool = typer.Option(
        False, "--list", help="List all models ready for download and exit."
    ),
    list_all: bool = typer.Option(
        False, "--list-all", help="List all registered models (including coming-soon)."
    ),
    cache_dir: Optional[str] = typer.Option(
        None, "--cache-dir", help="Override the default cache directory."
    ),
    force: bool = typer.Option(False, "--force", help="Re-download even if already cached."),
    cache_info: bool = typer.Option(
        False, "--cache-info", help="Show local cache status for all models and exit."
    ),
):
    """Download known-good pretrained world model checkpoints from HuggingFace.

    Examples:

    \b
      wml download --list                  # show available models
      wml download iris-atari-breakout     # download one model
      wml download                         # download all ready models
      wml download --cache-info            # inspect local cache
      wml download --list-all              # include coming-soon entries
    """

    dl = WeightsDownloader(cache_dir=cache_dir)

    if list_ready or list_all:
        models = dl.list_all() if list_all else dl.list_ready()
        table = Table(
            title="WorldModelLens — Pretrained Checkpoints",
            show_lines=True,
        )
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Backend", style="magenta")
        table.add_column("Environment", style="green")
        table.add_column("Status")
        table.add_column("Description")
        for m in models:
            status = (
                "[dim]coming soon[/dim]" if m.coming_soon else "[bold green]✓ ready[/bold green]"
            )
            table.add_row(m.name, m.backend, m.environment, status, m.description[:55])
        console.print(table)
        return

    if cache_info:
        info = dl.cache_info()
        table = Table(title="WorldModelLens — Local Cache", show_lines=True)
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Cached", style="green")
        table.add_column("Size (MB)")
        table.add_column("Path", overflow="fold")
        for name, entry in info.items():
            cached_str = "[green]yes[/green]" if entry["cached"] else "[dim]no[/dim]"
            size_str = str(entry["size_mb"]) if entry["size_mb"] else "—"
            path_str = entry["path"] or "—"
            table.add_row(name, cached_str, size_str, path_str)
        console.print(table)
        return

    if model:
        console.print(f"[bold]Downloading:[/bold] {model}")
        try:
            path = dl.download(model, force=force)
            console.print(f"\n[green]✓ Checkpoint ready at:[/green] {path}")
        except NotImplementedError as exc:
            console.print(f"[yellow]NOT AVAILABLE:[/yellow] {exc}")
            raise typer.Exit(1)
        except KeyError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            console.print("Run [bold]wml download --list-all[/bold] to see all model keys.")
            raise typer.Exit(1)
        except Exception as exc:
            console.print(f"[red]Download failed:[/red] {exc}")
            raise typer.Exit(1)
    else:
        # Download all ready models
        ready = dl.list_ready()
        if not ready:
            console.print("[yellow]No models are currently ready for download.[/yellow]")
            console.print("Run [bold]wml download --list-all[/bold] to see the full registry.")
            return
        console.print(f"[bold]Downloading {len(ready)} model(s)...[/bold]\n")
        results = dl.download_all(force=force)
        console.print(f"\n[green]Done.[/green] {len(results)}/{len(ready)} model(s) downloaded.")


@app.command()
def version():
    """Print version information."""
    console.print(f"[bold blue]World Model Lens[/bold blue] v{__version__}")


@app.command()
def info():
    """Show device and version information."""

    device = get_device()
    console.print(f"[bold]World Model Lens[/bold] v{__version__}")
    console.print(f"PyTorch: {torch.__version__}")
    console.print(f"Device: [green]{device}[/green]")
    console.print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        console.print(f"GPU: {torch.cuda.get_device_name(0)}")


@app.command()
def analyze(
    checkpoint_path: str = typer.Argument(..., help="Path to model checkpoint"),
    env: str = typer.Option("Pendulum-v1", help="Gymnasium environment ID"),
    backend: str = typer.Option("dreamerv3", help="Backend architecture"),
    n_episodes: int = typer.Option(10, help="Number of episodes to collect"),
    output_dir: str = typer.Option("./results", help="Output directory"),
    mode: str = typer.Option("auto", help="Analysis mode: auto, rl, non-rl"),
    probes: bool = typer.Option(False, help="Run probe sweep"),
    patching_sweep: bool = typer.Option(False, help="Run patching sweep"),
    report: str = typer.Option("html", help="Report format (html/text)"),
):
    """Analyze a world model checkpoint.

    Supports both RL and non-RL world models. Use --mode auto (default) to
    automatically detect the model type based on its capabilities, or specify
    --mode rl or --mode non-rl to override.

    RL mode runs:
    - Imagination faithfulness
    - Reward attribution
    - Value analysis
    - Surprise calibration

    Non-RL mode runs:
    - Latent geometry analysis
    - Temporal memory analysis
    - Surprise timeline
    - Disentanglement metrics
    """

    console.print(f"[bold]Analyzing:[/bold] {checkpoint_path}")
    console.print(f"[bold]Backend:[/bold] {backend}")
    console.print(f"[bold]Episodes:[/bold] {n_episodes}")
    console.print(f"[bold]Mode:[/bold] {mode}")

    try:
        cfg = WorldModelConfig(d_action=4, d_obs=12288, backend=cast(Any, backend))
        wm = HookedWorldModel.from_checkpoint(checkpoint_path, backend=backend, config=cfg)

        caps = wm.capabilities if hasattr(wm, "capabilities") else None

        if mode == "auto" and caps is not None:
            if caps.is_rl_model():
                mode = "rl"
            else:
                mode = "non-rl"

        _print_analysis_profile(console, mode, caps)

        if report == "text":
            console.print("[yellow]Analysis report generation coming soon...[/yellow]")
        else:
            console.print("[yellow]HTML report generation coming soon...[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def _print_analysis_profile(console: Console, mode: str, caps) -> None:
    """Print a table showing which analyses will run based on mode and capabilities."""
    table = Table(title=f"Analysis Profile: {mode.upper()}")
    table.add_column("Analysis", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Requires", style="yellow")

    if mode == "rl" or (caps and caps.is_rl_model()):
        analyses = [
            ("Surprise Timeline", "Yes", "latents"),
            ("Latent Geometry", "Yes", "latents"),
            ("Temporal Memory", "Yes", "latents"),
            ("Disentanglement", "Yes", "latents + factors"),
            (
                "Reward Attribution",
                "Yes" if (caps and caps.has_reward_head) else "No",
                "reward head",
            ),
            ("Value Analysis", "Yes" if (caps and caps.has_critic) else "No", "critic head"),
            (
                "Imagination Faithfulness",
                "Yes" if (caps and caps.uses_actions) else "No",
                "actions",
            ),
            ("Safety Audit", "Yes" if (caps and caps.has_reward_head) else "No", "reward head"),
        ]
    else:
        analyses = [
            ("Surprise Timeline", "Yes", "latents"),
            ("Latent Geometry", "Yes", "latents"),
            ("Temporal Memory", "Yes", "latents"),
            ("Disentanglement", "Yes", "latents + factors"),
            ("SAE Analysis", "Yes", "latents"),
            ("Reward Attribution", "Skipped", "no reward head"),
            ("Value Analysis", "Skipped", "no critic head"),
            ("Imagination Faithfulness", "Skipped", "no actions"),
        ]

    for name, status, req in analyses:
        if status == "Yes":
            style = "green"
        elif status == "Skipped":
            style = "dim"
        else:
            style = "yellow"
        table.add_row(name, f"[{style}]{status}[/{style}]", req)

    console.print(table)


@app.command()
def compare(
    checkpoint_a: str,
    checkpoint_b: str,
    env: str = typer.Option("Pendulum-v1", help="Environment ID"),
    n_episodes: int = typer.Option(30, help="Number of episodes per model"),
    mode: str = typer.Option("auto", help="Comparison mode: auto, rl, non-rl"),
):
    """Compare two model checkpoints.

    Compares latent geometry, surprise, and (for RL models) reward prediction.
    """

    console.print("[bold]Model Comparison[/bold]")
    console.print(f"Model A: {checkpoint_a}")
    console.print(f"Model B: {checkpoint_b}")
    console.print(f"Episodes: {n_episodes}")
    console.print(f"Mode: {mode}")
    console.print("[yellow]Comparison coming soon...[/yellow]")


@app.command()
def imagine(
    checkpoint_path: str,
    env: str = typer.Option("Pendulum-v1", help="Environment ID"),
    fork_at: int = typer.Option(10, help="Fork point"),
    horizon: int = typer.Option(30, help="Imagination horizon"),
    n_branches: int = typer.Option(5, help="Number of branches"),
    output_dir: str = typer.Option("./imagination", help="Output directory"),
):
    """Run imagination branching experiments."""
    console.print("[bold]Imagination Branching[/bold]")
    console.print(f"Fork at: t={fork_at}")
    console.print(f"Horizon: {horizon}")
    console.print(f"Branches: {n_branches}")
    console.print("[yellow]Imagination coming soon...[/yellow]")


@app.command()
def probe(
    checkpoint_path: str,
    env: str = typer.Option("Pendulum-v1", help="Environment ID"),
    concepts: str = typer.Option("all", help="Concepts to probe"),
    activations: str = typer.Option("z_posterior,h", help="Activations to probe"),
):
    """Train linear probes on activations."""
    console.print("[bold]Probe Training[/bold]")
    console.print(f"Checkpoint: {checkpoint_path}")
    console.print(f"Activations: {activations}")
    console.print("[yellow]Probe training coming soon...[/yellow]")


@app.command()
def circuits(
    checkpoint_path: str = typer.Argument(..., help="Path to model checkpoint"),
    layers: str = typer.Option(
        "encoder,dynamics,decoder", help="Comma-separated layer names to analyze"
    ),
    sae_checkpoints: Optional[str] = typer.Option(
        None,
        help="Comma-separated SAE checkpoint files matching layers order (optional). If omitted, wml will look for files named <model>.{layer}.sae.pt next to the model checkpoint.",
    ),
    observations: str = typer.Option(
        ..., help="Path to a torch-saved tensor (.pt) containing observations to run with the model"
    ),
    output: str = typer.Option(
        "./circuits_graph.gml", help="Output path to save the graph (networkx gml)"
    ),
    threshold: float = typer.Option(1e-3, help="Edge threshold for including edges"),
    topk: int = typer.Option(50, help="Top-K outgoing edges to keep per source feature"),
):
    """Build cross-layer SAE feature circuits and save graph to disk.

    Example:
      wml circuits model.pt --layers encoder,dynamics --observations obs.pt --output circuits.gml
    """
    console.print(f"[bold]Building SAE feature circuits for:[/bold] {checkpoint_path}")

    try:
        cfg = WorldModelConfig(d_action=4, d_obs=12288, backend=cast(Any, "dreamerv3"))
        wm = HookedWorldModel.from_checkpoint(checkpoint_path, backend="dreamerv3", config=cfg)

        # load observations
        obs = torch.load(observations)

        layer_list = [s.strip() for s in layers.split(",") if s.strip()]

        sae_files = None
        if sae_checkpoints:
            sae_files = [s.strip() for s in sae_checkpoints.split(",")]
            if len(sae_files) != len(layer_list):
                console.print(
                    "[red]Error: number of SAE checkpoints must match number of layers[/red]"
                )
                raise typer.Exit(1)
        else:
            # try convention: <checkpoint_path>.<layer>.sae.pt
            base = checkpoint_path
            sae_files = [f"{base}.{layer}.sae.pt" for layer in layer_list]

        # load SAEs, support common variants: direct pickled object, trainer.SAETrainingResult.sae,
        # or state-dict saved dicts.
        from world_model_lens.sae import SAEFeatureCircuitAnalyzer

        saes = {}
        cache_keys = {}
        for layer, sf in zip(layer_list, sae_files):
            try:
                loaded = torch.load(sf)
            except Exception as e:
                console.print(f"[red]Failed to load SAE file {sf}: {e}[/red]")
                raise typer.Exit(1)

            sae = None
            # pattern: SAETrainingResult or similar wrapper
            if hasattr(loaded, "sae"):
                sae = getattr(loaded, "sae")
            # direct SAE implementation
            elif hasattr(loaded, "encode") and hasattr(loaded, "decode"):
                sae = loaded
            # dict with state_dict
            elif isinstance(loaded, dict) and "state_dict" in loaded:
                try:
                    from world_model_lens.sae.trainer import SAETrainer

                    # best-effort reconstruct trainer (user may need to supply sizes)
                    d_input = loaded.get("d_input", None) or loaded.get("input_dim", None) or 0
                    n_boj = loaded.get("n_boj", None) or loaded.get("n_features", None) or 0
                    k = loaded.get("k", 1)
                    trainer = SAETrainer(d_input=d_input, n_boj=n_boj, k=k)
                    trainer.sae.load_state_dict(loaded["state_dict"])  # type: ignore[arg-type]
                    sae = trainer.sae
                except Exception:
                    sae = None

            if sae is None:
                console.print(
                    f"[red]Could not interpret SAE file {sf} for layer {layer}. Provide a trained SAE object or trainer output.[/red]"
                )
                raise typer.Exit(1)

            saes[layer] = sae
            cache_keys[layer] = layer  # default key = layer name; user can adjust later

        analyzer = SAEFeatureCircuitAnalyzer(
            wm=wm, saes=saes, cache_keys=cache_keys, threshold=threshold, topk_per_source=topk
        )

        graph = analyzer.build_graph(obs)

        # export to networkx and save
        nx = None
        try:
            import networkx as nx
        except Exception:
            console.print(
                "[red]networkx is required to save the graph. Install networkx and retry.[/red]"
            )
            raise typer.Exit(1)

        G = graph.to_networkx()
        nx.write_gml(G, output)

        # persist compact JSON metadata alongside GML for easy programmatic loading
        import json

        meta = {
            "edges": [
                {
                    "src_layer": e.src_layer,
                    "src_feat": e.src_feat,
                    "tgt_layer": e.tgt_layer,
                    "tgt_feat": e.tgt_feat,
                    "weight": e.weight,
                }
                for e in graph.edges
            ],
            "layer_feature_counts": graph.layer_feature_counts,
        }
        meta_path = output + ".json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        console.print(f"[green]Saved circuits graph to:[/green] {output}")
        console.print(f"[green]Saved circuits metadata to:[/green] {meta_path}")

    except Exception as e:
        console.print(f"[red]Error while building circuits:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def benchmark(
    checkpoint_path: str = typer.Argument(..., help="Path to model checkpoint"),
    backend: str = typer.Option("dreamerv3", help="Backend architecture"),
    tasks: str = typer.Option("all", help="Benchmark tasks: all, imagination, latent, geometry"),
    output_dir: str = typer.Option("./benchmark_results", help="Output directory"),
    mode: str = typer.Option("auto", help="Benchmark mode: auto, rl, non-rl"),
):
    """Run benchmark suite on a world model.

    RL mode benchmarks:
    - Imagination reward prediction
    - Value estimation accuracy
    - Planning horizon analysis

    Non-RL mode benchmarks:
    - Latent trajectory coherence
    - Dynamics prediction accuracy
    - Manifold structure preservation
    """
    console.print(f"[bold]Benchmarking:[/bold] {checkpoint_path}")
    console.print(f"[bold]Backend:[/bold] {backend}")
    console.print(f"[bold]Mode:[/bold] {mode}")
    console.print("[yellow]Benchmarking coming soon...[/yellow]")


@app.command()
def capabilities(
    checkpoint_path: str = typer.Argument(..., help="Path to model checkpoint"),
    backend: str = typer.Option("dreamerv3", help="Backend architecture"),
):
    """Show the capabilities of a world model.

    Displays which optional features (decoder, reward head, critic, etc.)
    are available in the model.
    """

    console.print(f"[bold]Inspecting:[/bold] {checkpoint_path}")
    console.print(f"[bold]Backend:[/bold] {backend}")

    try:
        cfg = WorldModelConfig(d_action=4, d_obs=12288, backend=cast(Any, backend))
        wm = HookedWorldModel.from_checkpoint(checkpoint_path, backend=backend, config=cfg)

        caps = wm.capabilities if hasattr(wm, "capabilities") else None

        if caps is None:
            console.print("[yellow]Capabilities not available[/yellow]")
            return

        table = Table(title="World Model Capabilities")
        table.add_column("Capability", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Has Decoder", str(caps.has_decoder))
        table.add_row("Has Reward Head", str(caps.has_reward_head))
        table.add_row("Has Continue Head", str(caps.has_continue_head))
        table.add_row("Has Actor", str(caps.has_actor))
        table.add_row("Has Critic", str(caps.has_critic))
        table.add_row("Uses Actions", str(caps.uses_actions))
        table.add_row("Is RL Trained", str(caps.is_rl_trained))

        console.print(table)

        model_type = "RL" if caps.is_rl_model() else "Non-RL"
        console.print(f"\n[bold]Model Type:[/bold] {model_type}")

        if caps.is_rl_model():
            console.print(
                "[green]This is an RL-trained world model with full analysis support.[/green]"
            )
        else:
            console.print(
                "[yellow]This is a non-RL world model. RL-specific analyses will be skipped.[/yellow]"
            )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def ui(
    port: int = typer.Option(8080, help="Port for web UI"),
    debug: bool = typer.Option(False, help="Enable debug mode"),
):
    """Launch interactive web UI."""
    console.print(f"[bold]Starting World Model Lens UI on port {port}...[/bold]")
    console.print("[yellow]Web UI coming soon...[/yellow]")


@app.command()
def run(
    experiment_yaml: str,
):
    """Run experiment from YAML file."""
    console.print(f"[bold]Running experiment:[/bold] {experiment_yaml}")
    console.print("[yellow]YAML runner coming soon...[/yellow]")


if __name__ == "__main__":
    app()
