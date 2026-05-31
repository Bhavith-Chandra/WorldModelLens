"""Simple graph plotting utilities for SAE circuits.

Produces a matplotlib plot using the hierarchical layout exported by
`intervention_plots.hierarchical_layout_from_graph`.
"""

from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt


def _rgba_to_hex(rgba: Tuple[float, float, float, float]) -> str:
    r, g, b, a = rgba
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def plot_circuit_graph_matplotlib(
    nx_graph,
    ax=None,
    node_size: int = 60,
    edge_cmap: str = "plasma",
    edge_width_scale: float = 5.0,
    show_labels: bool = True,
):
    """Plot circuit graph with richer styles using matplotlib.

    - Edge color maps to weight; edge width scales with weight.
    - Nodes are placed via hierarchical layout (x=layer, y=normalized index).
    """
    from world_model_lens.visualization.intervention_plots import hierarchical_layout_from_graph
    import networkx as nx
    import matplotlib as mpl

    pos = hierarchical_layout_from_graph(nx_graph)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # collect weights and normalize
    edge_data = [(u, v, d.get("weight", 0.0)) for u, v, d in nx_graph.edges(data=True)]
    weights = [w for _, _, w in edge_data]
    if weights:
        max_w = max(weights)
        norm = mpl.colors.Normalize(vmin=0.0, vmax=max_w)
        cmap = mpl.cm.get_cmap(edge_cmap)
    else:
        max_w = 1.0
        norm = None
        cmap = mpl.cm.get_cmap(edge_cmap)

    # draw edges
    for u, v, w in edge_data:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        if norm is not None:
            rgba = cmap(norm(w))
        else:
            rgba = (0.5, 0.5, 0.5, 0.7)
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=rgba,
            linewidth=max(0.5, (w / (max_w + 1e-12)) * edge_width_scale),
            alpha=rgba[3],
        )

    # nodes
    nodes = list(nx_graph.nodes())
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]

    # color nodes by layer
    layer_to_idx: Dict[str, int] = {}
    colors = []
    for n in nodes:
        layer = n[0] if isinstance(n, tuple) else str(n)
        if layer not in layer_to_idx:
            layer_to_idx[layer] = len(layer_to_idx)
        colors.append(layer_to_idx[layer])

    sc = ax.scatter(xs, ys, s=node_size, c=colors, cmap="tab10", zorder=3)

    if show_labels:
        for n in nodes:
            x, y = pos[n]
            ax.text(x, y - 0.02, str(n), fontsize=7, ha="center", va="top")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    return fig, ax


def plot_circuit_graph_plotly(nx_graph, title: str = "SAE Feature Circuit"):
    """Create an interactive Plotly figure for the circuit graph.

    Edges are colored by weight and have hover information. Nodes are interactive
    with hover text showing (layer, index).
    """
    try:
        import plotly.graph_objs as go
    except Exception as e:
        raise RuntimeError("plotly is required for interactive plotting") from e

    from world_model_lens.visualization.intervention_plots import hierarchical_layout_from_graph

    pos = hierarchical_layout_from_graph(nx_graph)

    edge_x = []
    edge_y = []
    edge_colors = []
    edge_widths = []
    edge_texts = []

    weights = [d.get("weight", 0.0) for _, _, d in nx_graph.edges(data=True)]
    max_w = max(weights) if weights else 1.0

    for u, v, d in nx_graph.edges(data=True):
        w = d.get("weight", 0.0)
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        # color via normalized weight mapped to a hex
        frac = min(1.0, w / (max_w + 1e-12))
        import matplotlib.cm as cm

        rgba = cm.plasma(frac)
        edge_colors.append(_rgba_to_hex(rgba))
        edge_widths.append(max(0.5, frac * 6))
        edge_texts.append(f"{u} → {v}<br>weight: {w:.4f}")

    # create edge trace (one per edge for hover/color control)
    edge_traces = []
    ei = 0
    for u, v, d in nx_graph.edges(data=True):
        w = d.get("weight", 0.0)
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        frac = min(1.0, w / (max_w + 1e-12))
        import matplotlib.cm as cm

        rgba = cm.plasma(frac)
        color = _rgba_to_hex(rgba)
        width = max(0.5, frac * 6)
        trace = go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="text",
            text=f"{u} → {v}<br>weight: {w:.4f}",
            showlegend=False,
        )
        edge_traces.append(trace)
        ei += 1

    # node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    layer_to_idx = {}
    for n in nx_graph.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(n))
        layer = n[0] if isinstance(n, tuple) else str(n)
        if layer not in layer_to_idx:
            layer_to_idx[layer] = len(layer_to_idx)
        node_color.append(layer_to_idx[layer])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="bottom center",
        marker=dict(size=12, color=node_color, colorscale="Viridis", showscale=False),
        hoverinfo="text",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
    )
    return fig


def plot_circuit_graph(nx_graph, interactive: bool = False, **kwargs):
    """Backward-compatible wrapper: choose matplotlib or interactive plotly output.

    Call with `interactive=True` to get a Plotly figure object.
    """
    if interactive:
        return plot_circuit_graph_plotly(nx_graph)
    return plot_circuit_graph_matplotlib(nx_graph, **kwargs)
