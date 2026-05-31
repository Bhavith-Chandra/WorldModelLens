"""Simple graph plotting utilities for SAE circuits.

Produces a matplotlib plot using the hierarchical layout exported by
`intervention_plots.hierarchical_layout_from_graph`.
"""

from typing import Any
import matplotlib.pyplot as plt


def plot_circuit_graph(nx_graph, ax=None, node_size: int = 40, cmap: str = "viridis"):
    from world_model_lens.visualization.intervention_plots import hierarchical_layout_from_graph

    pos = hierarchical_layout_from_graph(nx_graph)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # draw edges with alpha proportional to weight
    weights = [d.get("weight", 0.1) for _, _, d in nx_graph.edges(data=True)]
    max_w = max(weights) if weights else 1.0
    alphas = [min(1.0, w / max_w) for w in weights]

    import networkx as nx

    for (u, v, data), a in zip(nx_graph.edges(data=True), alphas):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="gray", alpha=a)

    xs = [pos[n][0] for n in nx_graph.nodes()]
    ys = [pos[n][1] for n in nx_graph.nodes()]
    ax.scatter(xs, ys, s=node_size, c=range(len(xs)), cmap=cmap)

    # annotate nodes with (layer,index)
    for n in nx_graph.nodes():
        x, y = pos[n]
        ax.text(x, y, str(n), fontsize=6, ha="center", va="center")

    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax
