"""Cross-layer SAE Feature Circuits analyzer.

This implements a lightweight pipeline to build a directed, weighted graph
between sparse features learned by SAEs at multiple layers. The implementation
follows the phases described in the project notes: collect clean feature
activations, run an ablation loop, accumulate weighted edges and export a
networkx graph for visualization.

Notes:
- This implementation intentionally keeps runtime assumptions conservative:
  * It requires a HookedWorldModel and an ActivationCache produced by
    `wm.run_with_cache()` so layer hidden activations can be read.
  * SAEs must be provided as already-trained objects (they must implement
    `encode(x) -> (features, indices)` and `decode(features) -> hidden`).
  * When SAE decode output dimensions differ between source and target
    layers the simple residual-addition shortcut is skipped (see docs).
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
import torch
import numpy as np


@dataclass
class CircuitEdge:
    src_layer: str
    src_feat: int
    tgt_layer: str
    tgt_feat: int
    weight: float


class FeatureCircuitGraph:
    """Directed weighted graph of (layer,feature) -> (layer,feature).

    Nodes are represented implicitly as tuples (layer:str, feat_idx:int).
    Edges are stored as a list of CircuitEdge dataclasses.
    """

    def __init__(self):
        self.edges: List[CircuitEdge] = []
        # track node feature counts per layer for layout/normalization
        self.layer_feature_counts: Dict[str, int] = {}

    def add_edge(self, src_layer: str, src_feat: int, tgt_layer: str, tgt_feat: int, weight: float):
        self.edges.append(CircuitEdge(src_layer, src_feat, tgt_layer, tgt_feat, float(weight)))
        # maintain approximate counts
        self.layer_feature_counts.setdefault(src_layer, 0)
        self.layer_feature_counts.setdefault(tgt_layer, 0)
        self.layer_feature_counts[src_layer] = max(
            self.layer_feature_counts[src_layer], src_feat + 1
        )
        self.layer_feature_counts[tgt_layer] = max(
            self.layer_feature_counts[tgt_layer], tgt_feat + 1
        )

    def prune_topk(self, k: int = 20):
        """Keep only top-k outgoing edges per source node (by weight)."""
        from collections import defaultdict

        outgoing: Dict[Tuple[str, int], List[CircuitEdge]] = defaultdict(list)
        for e in self.edges:
            outgoing[(e.src_layer, e.src_feat)].append(e)

        new_edges: List[CircuitEdge] = []
        for src, elist in outgoing.items():
            elist.sort(key=lambda e: e.weight, reverse=True)
            new_edges.extend(elist[:k])

        self.edges = new_edges

    def to_networkx(self):
        """Export to networkx.DiGraph. Returns (nx.DiGraph, node_metadata)

        Node attributes: layer (str), index (int), normalized_index (float).
        Edge attribute: weight (float).
        """
        try:
            import networkx as nx
        except Exception as e:
            raise RuntimeError("networkx is required to export graph; install networkx") from e

        G = nx.DiGraph()
        # add nodes
        for layer, count in self.layer_feature_counts.items():
            for idx in range(count):
                G.add_node(
                    (layer, idx),
                    layer=layer,
                    index=idx,
                    normalized_index=(idx / max(1, count - 1) if count > 1 else 0.0),
                )

        for e in self.edges:
            if not G.has_node((e.src_layer, e.src_feat)):
                G.add_node(
                    (e.src_layer, e.src_feat),
                    layer=e.src_layer,
                    index=e.src_feat,
                    normalized_index=0.0,
                )
            if not G.has_node((e.tgt_layer, e.tgt_feat)):
                G.add_node(
                    (e.tgt_layer, e.tgt_feat),
                    layer=e.tgt_layer,
                    index=e.tgt_feat,
                    normalized_index=0.0,
                )
            G.add_edge((e.src_layer, e.src_feat), (e.tgt_layer, e.tgt_feat), weight=e.weight)

        return G


class SAEFeatureCircuitAnalyzer:
    """Analyzer that constructs cross-layer SAE feature circuits.

    Basic usage:

        analyzer = SAEFeatureCircuitAnalyzer(wm, saes, cache_keys)
        graph = analyzer.build_graph(observations)

    where `saes` is a dict mapping layer name -> SAE instance, and
    `cache_keys` maps layer name -> ActivationCache key used by HookedWorldModel.
    """

    def __init__(
        self,
        wm: Any,
        saes: Dict[str, Any],
        cache_keys: Dict[str, str],
        device: Optional[torch.device] = None,
        threshold: float = 1e-3,
        topk_per_source: int = 50,
        max_features_per_layer: Optional[int] = None,
        max_timesteps_per_feature: Optional[int] = None,
    ):
        self.wm = wm
        self.saes = saes
        self.cache_keys = cache_keys
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.threshold = threshold
        self.topk_per_source = topk_per_source
        # performance knobs for the expensive exact re-run path
        self.max_features_per_layer = max_features_per_layer
        self.max_timesteps_per_feature = max_timesteps_per_feature

    def _collect_clean_acts(self, cache: Any) -> Dict[str, torch.Tensor]:
        """Encode cached hidden activations through their respective SAEs.

        Returns dict layer -> clean_acts [N, dict_size]
        """
        clean_acts: Dict[str, torch.Tensor] = {}
        for layer, sae in self.saes.items():
            key = self.cache_keys.get(layer, layer)
            # collect all timesteps available from ActivationCache
            try:
                acts = []
                for t in cache.timesteps:
                    a = cache[key, t]
                    if a is None:
                        continue
                    if a.dim() > 1:
                        a = a.flatten(1)
                    acts.append(a.detach().cpu())
                if not acts:
                    raise KeyError(f"No activations found for layer {layer} using key {key}")
                acts = torch.cat(acts, dim=0)
                sae = sae.to(self.device)
                with torch.no_grad():
                    feats, _ = sae.encode(acts.to(self.device))
                clean_acts[layer] = feats.detach().cpu()
            except Exception as e:
                # Provide more help: list available cache keys if ActivationCache present
                try:
                    avail = cache.component_names
                except Exception:
                    avail = None
                msg = f"Skipping layer {layer}: {e}"
                if avail:
                    msg += f"; available cache components: {avail}"
                    msg += f"; suggested cache key variants: {[layer, layer + '_out', layer + '.out', 'h', layer + '_hidden']}"
                warnings.warn(msg)
        return clean_acts

    def build_graph(
        self, observations: torch.Tensor, actions: Optional[torch.Tensor] = None
    ) -> FeatureCircuitGraph:
        """Run the full pipeline (run model, encode features, ablate, build graph).

        This is a relatively expensive operation; it runs `wm.run_with_cache()`
        on the provided observations and performs ablations per active feature.
        """
        traj, cache = self.wm.run_with_cache(observations, actions)

        clean_acts = self._collect_clean_acts(cache)

        # precompute decoded hidden for each layer
        clean_decoded: Dict[str, torch.Tensor] = {}
        for layer, feats in clean_acts.items():
            sae = self.saes[layer]
            with torch.no_grad():
                decoded = sae.decode(feats.to(self.device)).detach().cpu()
            clean_decoded[layer] = decoded

        # compute activity rates and active features
        active_features: Dict[str, List[int]] = {}
        for layer, feats in clean_acts.items():
            act_rate = (feats > 0).float().mean(dim=0)
            alive = (act_rate > 0).nonzero(as_tuple=True)[0].tolist()
            active_features[layer] = alive

        graph = FeatureCircuitGraph()

        # Ablation loop
        layers = list(self.saes.keys())
        for i_src, src_layer in enumerate(layers[:-1]):
            sae_src = self.saes[src_layer]
            feats_src = clean_acts.get(src_layer)
            if feats_src is None:
                continue
            decoded_src = clean_decoded[src_layer]

            for f in active_features.get(src_layer, []):
                # ablate feature f across all examples
                ablated_feats = feats_src.clone()
                if f >= ablated_feats.shape[1]:
                    continue
                ablated_feats[:, f] = 0.0

                with torch.no_grad():
                    ablated_decoded_src = (
                        sae_src.decode(ablated_feats.to(self.device)).detach().cpu()
                    )

                delta_hidden = ablated_decoded_src - decoded_src

                # For each downstream layer, attempt residual add if dims match
                for tgt_layer in layers[i_src + 1 :]:
                    sae_tgt = self.saes[tgt_layer]
                    feats_tgt = clean_acts.get(tgt_layer)
                    if feats_tgt is None:
                        continue
                    decoded_tgt = clean_decoded[tgt_layer]

                    # If dims match we can use the cheap residual-addition shortcut.
                    if delta_hidden.shape[1] == decoded_tgt.shape[1]:
                        perturbed_hidden = decoded_tgt + delta_hidden
                        # encode perturbed hidden through target SAE
                        with torch.no_grad():
                            ablated_tgt_feats, _ = sae_tgt.encode(perturbed_hidden.to(self.device))
                            ablated_tgt_feats = ablated_tgt_feats.detach().cpu()

                        # compute mean absolute change per target feature
                        edge_weights = (feats_tgt - ablated_tgt_feats).abs().mean(dim=0)

                        for j, w in enumerate(edge_weights.tolist()):
                            if w >= self.threshold:
                                graph.add_edge(src_layer, f, tgt_layer, j, w)
                    else:
                        # Non-residual case: we perform a batched exact re-run per
                        # source feature: choose a subset of timesteps (configurable)
                        # and install hooks for all those timesteps, then run once
                        # per source feature to obtain target activations.
                        try:
                            from world_model_lens.core.hooks import HookPoint

                            ablated_tgt_feats_list = []
                            N = decoded_tgt.shape[0]

                            # choose timesteps to test: all or sampled up to max_timesteps_per_feature
                            if (
                                self.max_timesteps_per_feature is None
                                or self.max_timesteps_per_feature >= N
                            ):
                                timesteps = list(range(N))
                            else:
                                # sample uniformly
                                idx = np.linspace(
                                    0, N - 1, num=self.max_timesteps_per_feature, dtype=int
                                )
                                timesteps = idx.tolist()

                            # create hooks for selected timesteps
                            hooks = []
                            for t in timesteps:
                                repl = ablated_decoded_src[t].to(self.device)

                                def make_hook(repl_tensor: torch.Tensor):
                                    def fn(_tensor, _ctx):
                                        return repl_tensor

                                    return fn

                                hooks.append(
                                    HookPoint(
                                        name=self.cache_keys.get(src_layer, src_layer),
                                        fn=make_hook(repl),
                                        timestep=t,
                                    )
                                )

                            # run model once with all hooks for this source feature
                            try:
                                traj2, cache2 = self.wm.run_with_hooks(
                                    observations, fwd_hooks=hooks, return_cache=True
                                )
                            except TypeError:
                                # fall back to running hooks and then a cache read
                                _ = self.wm.run_with_hooks(observations, fwd_hooks=hooks)
                                _, cache2 = self.wm.run_with_cache(observations)

                            tgt_key = self.cache_keys.get(tgt_layer, tgt_layer)
                            feats_list = []
                            for t in timesteps:
                                try:
                                    hidden_t = cache2[tgt_key, t]
                                except Exception:
                                    continue
                                with torch.no_grad():
                                    feats_enc, _ = sae_tgt.encode(hidden_t.to(self.device))
                                    feats_list.append(feats_enc.detach().cpu())

                            if not feats_list:
                                continue

                            ablated_tgt_feats = torch.stack(feats_list, dim=0)
                            # select corresponding rows from feats_tgt if available
                            try:
                                feats_tgt_sel = feats_tgt[timesteps]
                            except Exception:
                                feats_tgt_sel = feats_tgt

                            edge_weights = (feats_tgt_sel - ablated_tgt_feats).abs().mean(dim=0)

                            for j, w in enumerate(edge_weights.tolist()):
                                if w >= self.threshold:
                                    graph.add_edge(src_layer, f, tgt_layer, j, w)
                        except Exception:
                            # if anything fails in the expensive path, skip
                            continue

        # prune outgoing edges per source
        graph.prune_topk(self.topk_per_source)

        return graph
