#examples/ijepa/progressive_build.py
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from image_utils import get_sample_image, preprocess_image, get_ijepa_masks
from world_model_lens.core.config import WorldModelConfig

config = WorldModelConfig(
    backend="ijepa",
    d_embed=192,
    n_layers=6,
    n_heads=3,
    predictor_embed_dim=384
)




def animate_ijepa_progressive(seed: int = 42):
    """
    Animates the progressive build of an I-JEPA prediction,
    using actual attention weights and latent similarities.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Setup Data & Model
    print("Loading image and model...")
    raw_img = get_sample_image()
    img_tensor = preprocess_image(raw_img)
    model = IJEPAAdapter(config)
    model.eval()
    
    # 2. Generate Masks
    num_patches = 196
    context_ids, target_ids = get_ijepa_masks(num_patches=num_patches, num_context=40, num_target=1)
    target_id = target_ids[0]
    
    # 3. Get Ground Truth and Full Attention
    model.last_context_ids = context_ids
    model.last_target_ids = target_ids

    with torch.no_grad():
        #_, target_gt = model.predict(img_tensor, context_ids, target_ids)
        target_reps = model.target_encode(img_tensor)
        target_gt = target_reps[:, target_ids, :]
        context_latents, _ = model.encode(img_tensor)
        _ = model.predictor(context_latents, context_ids, target_ids)
        # Re-run once to get attention for ALL context patches
        attn = model.predictor.get_last_self_attention()
        # [B, H, L, L] -> Avg over heads -> Target to Context slice
        target_to_context = attn[0].mean(0)[-1, :len(context_ids)].cpu().numpy()
        
    # Sort context patches by their attention importance (attribution)
    sorted_indices = np.argsort(target_to_context)[::-1]
    sorted_context_ids = [context_ids[i] for i in sorted_indices]
    
    # 4. Animation Setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [2, 1]})
    
    # Track similarity as we add patches
    similarities = []

    def update(frame):
        ax1.clear()
        ax2.clear()

        current_context = sorted_context_ids[: frame + 1]

        with torch.no_grad():
            x = model.context_encoder.patch_embed(img_tensor)
            x = x + model.context_encoder.pos_embed
            subset_context_latents = x[:, current_context, :]

            pred_latents = model.predictor(subset_context_latents, current_context, [target_id])

            sim = F.cosine_similarity(pred_latents[0, 0], target_gt[0, 0], dim=0).item()
            similarities.append(sim)

        G = nx.DiGraph()
        target_node = f"target_{target_id}"
        for pid in current_context:
            G.add_edge(f"patch_{pid}", target_node)

        def get_coord(pid_str):
            pid = int(pid_str.split("_")[1])
            return (pid % 14, 14 - (pid // 14))

        pos = {node: get_coord(node) for node in G.nodes()}

        nx.draw_networkx_nodes(
            nx.Graph(),
            pos={f"patch_{pid}": get_coord(f"patch_{pid}") for pid in context_ids},
            nodelist=[f"patch_{pid}" for pid in context_ids],
            node_color="#eeeeee",
            node_size=300,
            alpha=0.2,
            ax=ax1,
        )

        node_colors = ["#F44336" if "target" in n else "#2196F3" for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, edgecolors="white", ax=ax1)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax1)
        nx.draw_networkx_edges(G, pos, edge_color="#2196F3", alpha=0.5, ax=ax1)

        ax1.set_title(f"Step {frame+1}: Adding patch {sorted_context_ids[frame]}\nSimilarity: {sim:.4f}")
        ax1.axis("off")

        ax2.plot(similarities, color="#F44336", marker="o", linewidth=2)
        ax2.set_xlim(-0.5, len(context_ids))
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_xlabel("Informative Patches Added")
        ax2.set_ylabel("Latent Cosine Similarity")
        ax2.set_title("Prediction Faithfulness")
        ax2.grid(True, alpha=0.3)

        return ()

    num_frames = min(20, len(sorted_context_ids)) # Limit frames for demonstration
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=500, repeat=False)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    animate_ijepa_progressive()
