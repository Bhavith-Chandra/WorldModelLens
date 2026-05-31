import networkx as nx
from world_model_lens.visualization.graph_viz import plot_circuit_graph


def test_plotly_fig_creation():
    G = nx.DiGraph()
    G.add_node(("encoder", 0), layer="encoder", index=0)
    G.add_node(("dynamics", 0), layer="dynamics", index=0)
    G.add_edge(("encoder", 0), ("dynamics", 0), weight=0.5)

    fig = plot_circuit_graph(G, interactive=True)
    # Plotly fig should be returned
    assert fig is not None
