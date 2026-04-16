"""Circuit discovery for world model mechanistic analysis."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from world_model_lens.core.activation_cache import ActivationCache
from world_model_lens.hooked_world_model import HookedWorldModel
from world_model_lens.patching.causal_tracer import CausalTracer


def _get_device() -> torch.device:
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class CircuitNode:
    """A node in the discovered circuit."""

    name: str
    importance: float
    in_degree: int
    out_degree: int
    attributions: dict[str, float] = field(default_factory=dict)


@dataclass
class CircuitEdge:
    """An edge in the discovered circuit."""

    source: str
    target: str
    strength: float
    edge_type: str = "causal"


@dataclass
class Circuit:
    """A discovered circuit subgraph."""

    nodes: list[CircuitNode]
    edges: list[CircuitEdge]
    source_nodes: list[str]
    target_nodes: list[str]
    faithfulness_score: float
    completeness: float

    def to_adjacency_list(self) -> dict[str, list[str]]:
        """Convert to adjacency list representation."""
        adj: dict[str, list[str]] = {node.name: [] for node in self.nodes}
        for edge in self.edges:
            if edge.source in adj:
                adj[edge.source].append(edge.target)
        return adj

    def get_path(self, source: str, target: str) -> list[str] | None:
        """Find a path from source to target."""
        if source not in self.to_adjacency_list():
            return None

        visited = set()
        queue = [(source, [source])]

        while queue:
            current, path = queue.pop(0)

            if current == target:
                return path

            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.to_adjacency_list().get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return None


class CircuitDiscovery:
    """Discovery of computational circuits in world models."""

    def __init__(self, wm: HookedWorldModel, device: torch.device | None = None):
        """Initialize circuit discovery.

        Args:
            wm: The hooked world model.
            device: Device for computations.
        """
        self.wm = wm
        self.device = device or _get_device()
        self.tracer = CausalTracer(wm, self.device)

    def greedy_prune(
        self,
        cache: ActivationCache,
        metric_fn: Callable[[ActivationCache], float],
        min_importance: float = 0.05,
    ) -> Circuit:
        """Greedy circuit pruning to find important subgraph.

        Args:
            cache: Activation cache.
            metric_fn: Metric function to optimize.
            min_importance: Minimum importance threshold.

        Returns:
            Discovered Circuit.
        """
        components = self.tracer.get_component_order(cache)
        attribution = self.tracer.compute_attribution(cache, metric_fn)

        nodes = []
        edges = []

        for comp in components:
            importance = attribution.component_scores.get(comp, 0.0)
            if importance >= min_importance:
                node = CircuitNode(
                    name=comp,
                    importance=importance,
                    in_degree=0,
                    out_degree=0,
                    attributions={comp: importance},
                )
                nodes.append(node)

        source_nodes = []
        target_nodes = []
        if nodes:
            source_nodes = [nodes[0].name]
            target_nodes = [nodes[-1].name] if len(nodes) > 1 else [nodes[0].name]

        for i, source_node in enumerate(nodes):
            for target_node in nodes[i + 1 :]:
                strength = self.tracer.compute_single_attribution(
                    cache, source_node.name, target_node.name, metric_fn
                )
                if strength > min_importance:
                    edge = CircuitEdge(
                        source=source_node.name,
                        target=target_node.name,
                        strength=strength,
                    )
                    edges.append(edge)
                    source_node.out_degree += 1
                    target_node.in_degree += 1

        faithfulness = sum(n.importance for n in nodes) / max(len(components), 1)
        completeness = len(nodes) / max(len(components), 1)

        return Circuit(
            nodes=nodes,
            edges=edges,
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            faithfulness_score=faithfulness,
            completeness=completeness,
        )

    def compute_edge_attribution(
        self,
        cache: ActivationCache,
        metric_fn: Callable[[ActivationCache], float],
    ) -> dict[tuple[str, str], float]:
        """Compute attribution for all potential edges.

        Args:
            cache: Activation cache.
            metric_fn: Metric function.

        Returns:
            Dictionary mapping (source, target) to attribution strength.
        """
        components = self.tracer.get_component_order(cache)
        edge_attributions: dict[tuple[str, str], float] = {}

        for i, source in enumerate(components):
            for target in components[i + 1 :]:
                strength = self.tracer.compute_single_attribution(cache, source, target, metric_fn)
                edge_attributions[(source, target)] = strength

        return edge_attributions

    def find_attention_heads(
        self,
        cache: ActivationCache,
        threshold: float = 0.1,
    ) -> list[tuple[str, float]]:
        """Find components with high cross-attention (important for outputs).

        Args:
            cache: Activation cache.
            threshold: Importance threshold.

        Returns:
            List of (component, attention_score) sorted by score.
        """
        components = list(cache.component_names)
        attention_scores = []

        for comp in components:

            def metric(c):
                return float((c.get(comp, 0) or torch.tensor(0.0)).abs().sum())

            score = self.tracer.compute_single_attribution(cache, comp, "z_posterior", metric)

            if score > threshold:
                attention_scores.append((comp, score))

        attention_scores.sort(key=lambda x: x[1], reverse=True)
        return attention_scores

    def extract_subgraph(
        self,
        nodes: list[str],
        edges: dict[tuple[str, str], float],
    ) -> Circuit:
        """Extract a subgraph from given nodes and edges.

        Args:
            nodes: List of node names.
            edges: Edge dictionary with (source, target) -> strength.

        Returns:
            Circuit subgraph.
        """
        circuit_nodes = []
        for node_name in nodes:
            circuit_nodes.append(
                CircuitNode(
                    name=node_name,
                    importance=0.0,
                    in_degree=0,
                    out_degree=0,
                )
            )

        circuit_edges = []
        node_set = set(nodes)

        for (source, target), strength in edges.items():
            if source in node_set and target in node_set:
                circuit_edges.append(
                    CircuitEdge(
                        source=source,
                        target=target,
                        strength=strength,
                    )
                )

        return Circuit(
            nodes=circuit_nodes,
            edges=circuit_edges,
            source_nodes=[nodes[0]] if nodes else [],
            target_nodes=[nodes[-1]] if nodes else [],
            faithfulness_score=0.0,
            completeness=0.0,
        )

    def analyze_bottleneck(
        self,
        cache: ActivationCache,
        metric_fn: Callable[[ActivationCache], float],
    ) -> list[dict[str, Any]]:
        """Analyze bottleneck nodes in the circuit.

        Args:
            cache: Activation cache.
            metric_fn: Metric function.

        Returns:
            List of bottleneck analysis results.
        """
        components = list(cache.component_names)
        bottlenecks: list[dict[str, Any]] = []

        clean_metric = metric_fn(cache)

        for i, comp in enumerate(components):
            test_cache = ActivationCache()
            for c in components:
                for t in cache.timesteps:
                    val = cache.get(c, t, None)
                    if val is not None:
                        test_cache[c, t] = val.to(self.device)

            for t in cache.timesteps:
                val = cache.get(comp, t, None)
                if val is not None:
                    test_cache[comp, t] = torch.zeros_like(val).to(self.device)

            metric_without = metric_fn(test_cache)
            importance = (clean_metric - metric_without) / (clean_metric + 1e-8)

            bottlenecks.append(
                {
                    "component": comp,
                    "importance": float(importance),
                    "position": i,
                    "total_components": len(components),
                }
            )

        bottlenecks.sort(key=lambda x: x["importance"], reverse=True)
        return bottlenecks

    def greedy_ablation_attention_heads(
        self,
        cache: ActivationCache,
        prediction_types: dict[str, Callable[[ActivationCache], float]],
        head_pattern: str = "attn_head",
        threshold: float = 0.01,
    ) -> dict[str, Circuit]:
        """Greedy ablation over attention heads to find minimal subgraph per prediction type.

        Args:
            cache: Activation cache.
            prediction_types: Dict of prediction type names to metric functions.
            head_pattern: Pattern to identify attention head components.
            threshold: Minimum drop threshold to stop ablation.

        Returns:
            Dict of prediction type to minimal Circuit.
        """
        all_heads = [comp for comp in cache.component_names if head_pattern in comp]
        if not all_heads:
            raise ValueError(f"No attention head components found with pattern '{head_pattern}'")

        results = {}
        for pred_type, metric_fn in prediction_types.items():
            active_heads = set(all_heads)
            clean_metric = metric_fn(cache)

            while active_heads:
                effects = []
                for head in active_heads:
                    effect = compute_lesion_effect(cache, head, metric_fn)
                    effects.append((head, effect))

                if not effects:
                    break

                effects.sort(key=lambda x: x[1])
                least_important_head, smallest_effect = effects[0]
                drop = smallest_effect / (clean_metric + 1e-8)
                if drop > threshold:
                    break
                active_heads.remove(least_important_head)

            nodes = [
                CircuitNode(
                    name=head,
                    importance=0.0,
                    in_degree=0,
                    out_degree=0,
                )
                for head in sorted(active_heads)
            ]
            edges = []
            for i in range(len(nodes) - 1):
                edges.append(
                    CircuitEdge(
                        source=nodes[i].name,
                        target=nodes[i + 1].name,
                        strength=1.0,
                    )
                )
            faithfulness = len(active_heads) / len(all_heads)
            circuit = Circuit(
                nodes=nodes,
                edges=edges,
                source_nodes=[nodes[0].name] if nodes else [],
                target_nodes=[nodes[-1].name] if nodes else [],
                faithfulness_score=faithfulness,
                completeness=1.0,
            )
            results[pred_type] = circuit

        return results


class CircuitComparator:
    """Compare circuits across different world model types."""

    def __init__(self):
        self.circuits: dict[str, Circuit] = {}

    def add_circuit(self, name: str, circuit: Circuit) -> None:
        """Add a circuit to the comparator.

        Args:
            name: Circuit identifier.
            circuit: Circuit to add.
        """
        self.circuits[name] = circuit

    def compare_structure(self) -> dict[str, Any]:
        """Compare structural properties of circuits.

        Returns:
            Comparison results.
        """
        results = {}

        for name, circuit in self.circuits.items():
            results[name] = {
                "n_nodes": len(circuit.nodes),
                "n_edges": len(circuit.edges),
                "faithfulness": circuit.faithfulness_score,
                "completeness": circuit.completeness,
                "density": len(circuit.edges) / max(len(circuit.nodes), 1),
            }

        return results

    def find_common_subcircuit(
        self,
        names: list[str] | None = None,
    ) -> Circuit | None:
        """Find common subgraph across circuits.

        Args:
            names: Circuit names to compare. If None, compare all.

        Returns:
            Common circuit if found, None otherwise.
        """
        if names is None:
            names = list(self.circuits.keys())

        if len(names) < 2:
            return None

        circuits_to_compare = [self.circuits[name] for name in names if name in self.circuits]
        if len(circuits_to_compare) < 2:
            return None

        common_nodes_set = set(circuits_to_compare[0].nodes[0].name)

        for circuit in circuits_to_compare[1:]:
            circuit_node_names = {node.name for node in circuit.nodes}
            common_nodes_set = common_nodes_set.intersection(circuit_node_names)

        if not common_nodes_set:
            return None

        common_nodes = [
            node for node in circuits_to_compare[0].nodes if node.name in common_nodes_set
        ]

        common_edges = []
        edge_counts: dict[tuple[str, str], int] = {}

        for circuit in circuits_to_compare:
            for edge in circuit.edges:
                key = (edge.source, edge.target)
                if edge.source in common_nodes_set and edge.target in common_nodes_set:
                    edge_counts[key] = edge_counts.get(key, 0) + 1

        for (source, target), count in edge_counts.items():
            if count == len(circuits_to_compare):
                strength = sum(
                    next(
                        (e.strength for e in c.edges if e.source == source and e.target == target),
                        0.0,
                    )
                    for c in circuits_to_compare
                ) / len(circuits_to_compare)
                common_edges.append(CircuitEdge(source, target, strength))

        return Circuit(
            nodes=common_nodes,
            edges=common_edges,
            source_nodes=[common_nodes[0].name] if common_nodes else [],
            target_nodes=[common_nodes[-1].name] if common_nodes else [],
            faithfulness_score=1.0,
            completeness=1.0,
        )

    def compute_circuit_similarity(self, name1: str, name2: str) -> float:
        """Compute Jaccard similarity between two circuits.

        Args:
            name1: First circuit name.
            name2: Second circuit name.

        Returns:
            Similarity score between 0 and 1.
        """
        if name1 not in self.circuits or name2 not in self.circuits:
            return 0.0

        nodes1 = {node.name for node in self.circuits[name1].nodes}
        nodes2 = {node.name for node in self.circuits[name2].nodes}

        intersection = len(nodes1.intersection(nodes2))
        union = len(nodes1.union(nodes2))

        if union == 0:
            return 0.0

        return intersection / union


class SubgraphAnalyzer:
    """Analyze subgraph properties of circuits."""

    @staticmethod
    def compute_modularity(circuit: Circuit) -> float:
        """Compute modularity of the circuit.

        Args:
            circuit: Circuit to analyze.

        Returns:
            Modularity score.
        """
        if len(circuit.nodes) <= 1:
            return 0.0

        adj = circuit.to_adjacency_list()
        degrees = {node: len(adj[node]) for node in adj}

        m = len(circuit.edges)
        if m == 0:
            return 0.0

        modularity = 0.0
        for edge in circuit.edges:
            if edge.source in degrees and edge.target in degrees:
                ki = degrees[edge.source]
                kj = degrees[edge.target]
                expected = (ki * kj) / (2 * m)
                modularity += edge.strength - expected

        modularity /= 2 * m
        return float(modularity)

    @staticmethod
    def find_communities(circuit: Circuit) -> list[list[str]]:
        """Find communities in the circuit using greedy approach.

        Args:
            circuit: Circuit to analyze.

        Returns:
            List of communities (each is a list of node names).
        """
        if len(circuit.nodes) <= 1:
            return [[node.name] for node in circuit.nodes]

        adj = circuit.to_adjacency_list()
        communities = []
        unassigned = set(adj.keys())

        while unassigned:
            if not unassigned:
                break

            seed = next(iter(unassigned))
            community = {seed}
            frontier = {seed}
            unassigned.discard(seed)

            while frontier:
                next_frontier = set()
                for node in frontier:
                    for neighbor in adj.get(node, []):
                        if neighbor in unassigned:
                            edge_strength = next(
                                (
                                    e.strength
                                    for e in circuit.edges
                                    if (e.source == node and e.target == neighbor)
                                    or (e.source == neighbor and e.target == node)
                                ),
                                0.0,
                            )
                            if edge_strength > 0.1:
                                community.add(neighbor)
                                next_frontier.add(neighbor)
                                unassigned.discard(neighbor)
                frontier = next_frontier

            communities.append(list(community))

        return communities


def default_circuit_metric(cache: ActivationCache) -> float:
    """Default metric for circuit discovery: z_posterior activation sum.

    Args:
        cache: Activation cache.

    Returns:
        Metric value.
    """
    if "z_posterior" in cache.component_names and 0 in cache.timesteps:
        return float(cache["z_posterior", 0].abs().sum().item())
    return 0.0


def compute_lesion_effect(
    cache: ActivationCache,
    component: str,
    metric_fn: Callable[[ActivationCache], float],
) -> float:
    """Compute effect of lesioning a component.

    Args:
        cache: Activation cache.
        component: Component to lesion.
        metric_fn: Metric function.

    Returns:
        Lesion effect (negative is important).
    """
    clean_metric = metric_fn(cache)

    lesioned_cache = ActivationCache()
    for comp in cache.component_names:
        for t in cache.timesteps:
            val = cache.get(comp, t, None)
            if val is not None:
                if comp == component:
                    lesioned_cache[comp, t] = torch.zeros_like(val)
                else:
                    lesioned_cache[comp, t] = val.clone()

    lesioned_metric = metric_fn(lesioned_cache)

    return clean_metric - lesioned_metric
