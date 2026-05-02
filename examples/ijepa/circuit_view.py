#examples/ijepa/circuit_view.py
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from image_utils import get_sample_image, preprocess_image, get_ijepa_masks
import os
from typing import cast, Any
import matplotlib as mpl

class ThresholdCircuitVisualizer:
    def __init__(self, threshold_pct=0.005):
        self.threshold_pct = threshold_pct
        self.raw_img = get_sample_image()
        self.img_tensor = preprocess_image(self.raw_img)
        
        config = WorldModelConfig(backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384)
        self.adapter = IJEPAAdapter(config)
        
        checkpoint_path = "ijepa_mini.pth"
        if os.path.exists(checkpoint_path):
            print(f"Loading weights from {checkpoint_path}")
            self.adapter.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)
            
        self.wm = HookedWorldModel(self.adapter, config)
        self.wm.adapter.eval()

    def get_patch_rect(self, patch_id):
        row, col = patch_id // 14, patch_id % 14
        p_size = 224 // 14
        return col * p_size, row * p_size, p_size, p_size

    def visualize(self, target_id=97, layer_idx=-1):
        # 1. Run Analysis
        context_ids, _ = get_ijepa_masks(num_context=100)
        if target_id in context_ids: context_ids.remove(target_id)
        
        self.adapter.last_context_ids = context_ids
        self.adapter.last_target_ids = [target_id]
        
        with torch.no_grad():
            self.wm.run_with_cache(self.img_tensor)
            attn = self.adapter.predictor.blocks[layer_idx].attn.last_attn_weights
            attributions = attn[0].mean(0)[-1, :len(context_ids)].cpu().numpy()

        # 2. Dynamic Thresholding
        # Identify all patches exceeding the threshold percentage of total energy
        total_energy = attributions.sum()
        circuit_indices = np.where(attributions > (total_energy * self.threshold_pct))[0]
        
        circuit_patches = [context_ids[i] for i in circuit_indices]
        circuit_weights = [attributions[i] for i in circuit_indices]
        
        print(f"Found circuit of {len(circuit_patches)} patches using {self.threshold_pct*100}% threshold.")

        # 3. Setup Layout
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
        ax_graph = fig.add_subplot(gs[0, 0])
        ax_img = fig.add_subplot(gs[0, 1])
        
        # 4. Draw Graph
        G = nx.DiGraph()
        target_node = f"target_{target_id}"
        for pid, w in zip(circuit_patches, circuit_weights):
            G.add_edge(f"patch_{pid}", target_node, weight=float(w))
            
        pos = self.compute_radial_layout(G, target_node)
        self.draw_graph(ax_graph, G, pos, target_node)
        
        # 5. Draw Image Grounding
        self.draw_image(ax_img, target_id, circuit_patches)
        
        plt.suptitle(f"I-JEPA Dynamic Circuit View (Threshold > {self.threshold_pct*100}% energy)", fontsize=16)
        plt.tight_layout()
        
        if os.environ.get("SAVE_PLOT"):
            plt.savefig("circuit_view.png")
            print("Circuit view saved to circuit_view.png")
        else:
            plt.show()

    def compute_radial_layout(self, G, target_node):
        pos = {target_node: np.array([0, 0])}
        context_nodes = [n for n in G.nodes() if n != target_node]
        for i, node in enumerate(context_nodes):
            angle = 2 * np.pi * i / max(1, len(context_nodes))
            radius = 1.0
            pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        return pos

    def draw_graph(self, ax, G, pos, target_node):
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        if weights:
            max_w = max(weights)
            widths = [1 + 10 * (w / max_w) for w in weights]
            draw_edges = cast(Any, nx.draw_networkx_edges)

            draw_edges(G, pos, width=widths, 
                                   edge_color=weights, edge_cmap=mpl.colormaps["Greens"], ax=ax, alpha=0.6)
            
        nx.draw_networkx_nodes(G, pos, nodelist=[target_node], node_color="#F44336", node_size=1500, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes() if n != target_node], 
                               node_color="#4CAF50", node_size=800, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", ax=ax)
        ax.axis('off')

    def draw_image(self, ax, target_id, circuit_patches):
        ax.imshow(self.raw_img.resize((224, 224)))
        tx, ty, tw, th = self.get_patch_rect(target_id)
        ax.add_patch(patches.Rectangle((tx, ty), tw, th, linewidth=3, edgecolor='#F44336', facecolor='none'))
        
        for pid in circuit_patches:
            cx, cy, cw, ch = self.get_patch_rect(pid)
            ax.add_patch(patches.Rectangle((cx, cy), cw, ch, linewidth=2, edgecolor='#4CAF50', facecolor='none', alpha=0.7))
        ax.axis('off')
        ax.set_title("Spatial Grounding")

if __name__ == "__main__":
    visualizer = ThresholdCircuitVisualizer()
    visualizer.visualize(target_id=97)
