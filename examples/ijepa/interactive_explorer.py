#examples/ijepa/interactive_explorer.py
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, RadioButtons
import numpy as np
from PIL import Image
from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from image_utils import get_sample_image, preprocess_image
from matplotlib import cm

class AttributionExplorer:
    def __init__(self, k=6, threshold=0.01, layout="radial"):
        print("Initializing official I-JEPA Interactive Explorer...")
        self.k = k
        self.threshold = threshold
        self.layout_mode = layout
        self.target_id = 97 
        self.hovered_node = None
        
        # 1. Load Model & Data
        self.raw_img = get_sample_image()
        self.img_tensor = preprocess_image(self.raw_img)
        
        config = WorldModelConfig(backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384, predictor_depth=4, predictor_heads=6)
        self.adapter = IJEPAAdapter(config)
        
        # Load weights if they exist (best effort)
        import os
        checkpoint_path = "ijepa_mini.pth"
        if os.path.exists(checkpoint_path):
            print(f"Loading weights from {checkpoint_path}")
            try:
                self.adapter.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)
            except Exception as e:
                print(f"Could not load weights: {e}. Using random initialization.")
            
        self.wm = HookedWorldModel(self.adapter, config)
        self.wm.adapter.eval()
        
        # 2. Pre-compute All-to-All Attention matrix
        print("Pre-computing full context attention matrix...")
        with torch.no_grad():
            # Run with cache. 
            traj, cache = self.wm.run_with_cache(self.img_tensor)
            
            # Extract attention from the last context encoder block
            # In IJEPAAdapter, Attention weights are DETACHED and stored in .last_attn_weights
            attn = self.adapter.context_encoder.blocks[-1].attn.last_attn_weights[0]
            self.attn_matrix = attn.mean(0).cpu().numpy() # [196, 196]
            self.num_patches = self.attn_matrix.shape[0]

            if self.target_id >= self.num_patches:
                print(f"Adjusting target_id from {self.target_id} → {self.num_patches - 1}")
                self.target_id = self.num_patches - 1

        # 3. Setup Figure
        self.fig = plt.figure(figsize=(16, 10))
        self.gs = self.fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[1.2, 1])
        
        self.ax_graph = self.fig.add_subplot(self.gs[0, 0])
        self.ax_image = self.fig.add_subplot(self.gs[0, 1])
        self.ax_ctrl = self.fig.add_subplot(self.gs[1, :])
        self.ax_ctrl.axis('off')
        
        self.setup_widgets()
        self.setup_hover()
        self.update()

    def get_patch_rect(self, patch_id):
        row, col = patch_id // 14, patch_id % 14
        p_size = 224 // 14
        return col * p_size, row * p_size, p_size, p_size

    def setup_widgets(self):
        ax_target = self.fig.add_axes((0.15, 0.15, 0.3, 0.03))
        #self.s_target = Slider(ax_target, 'Target Patch', 0, 195, valinit=self.target_id, valstep=1)
        self.s_target = Slider(ax_target, 'Target Patch', 0, self.num_patches - 1, valinit=self.target_id, valstep=1)
        ax_k = self.fig.add_axes((0.15, 0.1, 0.3, 0.03))
        self.s_k = Slider(ax_k, 'Top-K', 1, 25, valinit=self.k, valstep=1)
        
        ax_thresh = self.fig.add_axes((0.15, 0.05, 0.3, 0.03))
        self.s_thresh = Slider(ax_thresh, 'Threshold', 0.0, 0.1, valinit=self.threshold)
        
        ax_layout = self.fig.add_axes((0.6, 0.05, 0.1, 0.1))
        self.r_layout = RadioButtons(ax_layout, ('radial', 'spatial', 'bipartite'), active=0)
        
        self.s_target.on_changed(self.on_widget_change)
        self.s_k.on_changed(self.on_widget_change)
        self.s_thresh.on_changed(self.on_widget_change)
        self.r_layout.on_clicked(self.on_widget_change)

    def setup_hover(self):
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)

    def on_widget_change(self, val):
        self.target_id = int(self.s_target.val)
        self.k = int(self.s_k.val)
        self.threshold = self.s_thresh.val
        self.layout_mode = self.r_layout.value_selected
        self.update()

    def update(self):
        self.ax_graph.clear()
        self.ax_image.clear()
        
        attributions = self.attn_matrix[self.target_id]
        indices = np.argsort(attributions)[::-1]
        
        top_indices = [i for i in indices if i != self.target_id and attributions[i] > self.threshold][:self.k]
        top_weights = [attributions[i] for i in top_indices]
        
        self.G = nx.DiGraph()
        target_node = f"patch_{self.target_id}"
        for i, weight in zip(top_indices, top_weights):
            self.G.add_edge(f"patch_{i}", target_node, weight=float(weight))
            
        self.pos = self.compute_layout(target_node)
        self.draw_graph(target_node, top_indices, top_weights)
        self.draw_image(top_indices)
        self.fig.canvas.draw_idle()

    def compute_layout(self, target_node):
        pos = {}
        context_nodes = [n for n in self.G.nodes() if n != target_node]
        if self.layout_mode == "radial":
            pos[target_node] = np.array([0, 0])
            for i, node in enumerate(context_nodes):
                w = self.G.get_edge_data(node, target_node)['weight']
                angle = np.pi/2 - (2 * np.pi * i / len(context_nodes))
                radius = 1.2 - (w * 10) 
                pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        elif self.layout_mode == "spatial":
            for node in self.G.nodes():
                pid = int(node.split("_")[1])
                pos[node] = np.array([pid % 14, 14 - (pid // 14)])
        else: # bipartite
            pos[target_node] = np.array([1, 0.5])
            for i, node in enumerate(context_nodes):
                pos[node] = np.array([0, i / max(1, len(context_nodes)-1)])
        return pos

    def draw_graph(self, target_node, top_indices, top_weights):
        edges = list(self.G.edges())
        cmap = plt.get_cmap('viridis')

        if edges:
            weights = [self.G[u][v]['weight'] for u, v in edges]
            max_w = max(weights) if weights else 0.001
            for i, (u, v) in enumerate(edges):
                w = weights[i]
                is_hovered = (u == self.hovered_node)
                color = "#FFD700" if is_hovered else cmap(w / max_w)
                nx.draw_networkx_edges(self.G, self.pos, edgelist=[(u,v)], width=1+8*(w/max_w), 
                                       edge_color=color, ax=self.ax_graph, alpha=0.9 if is_hovered else 0.6)

        context_nodes = [n for n in self.G.nodes() if n != target_node]
        target_color = "#F44336"
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[target_node], node_color=target_color, node_size=1000, ax=self.ax_graph)
        
        for n in context_nodes:
            color = "#FFD700" if n == self.hovered_node else "#2196F3"
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[n], node_color=color, node_size=600, ax=self.ax_graph)
        
        nx.draw_networkx_labels(self.G, self.pos, font_size=7, font_weight="bold", ax=self.ax_graph, font_color="white")
        self.ax_graph.set_title(f"I-JEPA Activation Topology ({self.layout_mode})")
        self.ax_graph.axis('off')

    def draw_image(self, top_indices):
        self.ax_image.imshow(self.raw_img.resize((224, 224)))
        tx, ty, tw, th = self.get_patch_rect(self.target_id)
        self.ax_image.add_patch(patches.Rectangle((tx, ty), tw, th, linewidth=3, edgecolor='#F44336', facecolor='none'))
        
        for pid in top_indices:
            cx, cy, cw, ch = self.get_patch_rect(pid)
            is_hovered = (f"patch_{pid}" == self.hovered_node)
            color = "#FFD700" if is_hovered else "#2196F3"
            lw = 3 if is_hovered else 1.5
            alpha = 1.0 if is_hovered else 0.6
            self.ax_image.add_patch(patches.Rectangle((cx, cy), cw, ch, linewidth=lw, edgecolor=color, facecolor='none', alpha=alpha))
            
        self.ax_image.set_title("I-JEPA Representation Grounding")
        self.ax_image.axis('off')

    def on_hover(self, event):
        if event.inaxes != self.ax_graph and event.inaxes != self.ax_image:
            return
        new_hover = None
        if event.inaxes == self.ax_graph:
            for node, (nx_x, nx_y) in self.pos.items():
                dist = np.sqrt((event.xdata - nx_x)**2 + (event.ydata - nx_y)**2)
                if dist < 0.15:
                    new_hover = node
                    break
        elif event.inaxes == self.ax_image:
            px, py = event.xdata, event.ydata
            grid_row, grid_col = int(py // (224/14)), int(px // (224/14))
            hover_pid = grid_row * 14 + grid_col
            if f"patch_{hover_pid}" in self.G.nodes():
                new_hover = f"patch_{hover_pid}"

        if new_hover != self.hovered_node:
            self.hovered_node = new_hover
            self.update()

if __name__ == "__main__":
    explorer = AttributionExplorer()
    plt.show()
