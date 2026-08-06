#example/ijepa/formal_circuits.py
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from image_utils import get_sample_image, preprocess_image, get_ijepa_masks


class FormalCircuitDiscoverer:
    def __init__(self):
        self.raw_img = get_sample_image()
        self.img_tensor = preprocess_image(self.raw_img)

        self.config = WorldModelConfig(
            backend="ijepa",
            d_embed=192,
            n_layers=6,
            n_heads=3,
            predictor_embed_dim=384,
        )

        self.adapter = IJEPAAdapter(self.config)

        checkpoint_path = "ijepa_mini.pth"
        if os.path.exists(checkpoint_path):
            print(f"Loading weights from {checkpoint_path}")
            sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            elif isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]

            self.adapter.load_state_dict(sd, strict=True)
            print("Checkpoint loaded successfully with strict=True")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")

        self.wm = HookedWorldModel(self.adapter, self.config)
        self.wm.adapter.eval()

    def calculate_performance(self, target_gt, pred, k=None):
        """Normalized performance: 1 - mse / zero-baseline-mse."""
        mse = F.mse_loss(pred, target_gt).item()
        baseline_mse = torch.mean(target_gt**2).item()
        return  1 - (mse / (baseline_mse + 1e-6))


    def discover_minimal_set(self, target_id, threshold=0.90):
        print(
            f"Formalizing circuit for Target {target_id} "
            f"(Target threshold: {threshold*100}%)..."
        )

        # 1. Get attributions to rank patches
        context_ids, _ = get_ijepa_masks(num_context=100)
        if target_id in context_ids:
            context_ids.remove(target_id)

        self.adapter.last_context_ids = context_ids
        self.adapter.last_target_ids = [target_id]

        with torch.no_grad():
            _, cache = self.wm.run_with_cache(self.img_tensor)

            attn = self.adapter.predictor.blocks[-1].attn.last_attn_weights
            attributions = attn[0].mean(0)[-1, : len(context_ids)].cpu().numpy()

            target_reps = cache["target_encoding", 0]
            target_gt = (
                target_reps[0, [target_id], :]
                if target_reps.dim() == 3
                else target_reps[[target_id], :]
            )

            # Debug 1: is attention ranking even non-trivial?
            print("Attribution variance:", float(np.var(attributions)))

            # Debug 2: sanity check before greedy circuit loop
            h_full, _ = self.adapter.encode(self.img_tensor)
            pred_full = self.adapter.predictor(h_full, self.adapter.last_context_ids, [target_id])
            sanity_mse = F.mse_loss(pred_full[0], target_gt).item()
            print("Sanity MSE:", sanity_mse)

            for name, p in self.adapter.predictor.named_parameters():
                print(f"{name}: mean|w|={p.abs().mean().item():.6f}")
                break

        # 2. Iterative Addition Loop
        ranked_indices = np.argsort(attributions)[::-1]

        performance_history = []
        circuit_patches = []
        minimal_set = []
        found_k = -1

        for i, idx in enumerate(ranked_indices):
            current_patch = context_ids[idx]
            circuit_patches.append(current_patch)

            self.adapter.last_context_ids = circuit_patches
            self.adapter.last_target_ids = [target_id]

            with torch.no_grad():
                h, _ = self.adapter.encode(self.img_tensor)
                pred = self.adapter.predictor(h, self.adapter.last_context_ids, [target_id])
                pred = pred[0]

                perf = self.calculate_performance(target_gt, pred, k=i + 1)

            performance_history.append(perf)

            if perf >= threshold and found_k == -1:
                found_k = i + 1
                minimal_set = list(circuit_patches)
                print(f"Threshold reached at K={found_k} patches (Perf: {perf:.4f})")
                # Optional early stop:
                # break

        return {
            "target_id": target_id,
            "found_k": found_k,
            "minimal_set": minimal_set,
            "history": performance_history,
            "threshold": threshold,
        }

    def visualize_result(self, result):
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(1, 2)

        ax_curve = fig.add_subplot(gs[0, 0])
        ax_img = fig.add_subplot(gs[0, 1])

        history = result["history"]
        ax_curve.plot(
            range(1, len(history) + 1),
            history,
            color="#2196F3",
            linewidth=3,
            label="Top-K Performance",
        )
        ax_curve.axhline(
            y=result["threshold"],
            color="#F44336",
            linestyle="--",
            label=f"Threshold ({result['threshold']*100}%)",
        )

        if result["found_k"] != -1:
            ax_curve.axvline(
                x=result["found_k"],
                color="#4CAF50",
                linestyle=":",
                linewidth=2,
            )
            ax_curve.scatter(
                [result["found_k"]],
                [history[result["found_k"] - 1]],
                color="#4CAF50",
                s=100,
                zorder=5,
            )
            ax_curve.annotate(
                f"Minimal Circuit (K={result['found_k']})",
                xy=(result["found_k"], history[result["found_k"] - 1]),
                xytext=(result["found_k"] + 5, history[result["found_k"] - 1] - 0.15),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1),
            )

        ax_curve.set_xlabel("Number of Context Patches (Added by Attribution Rank)")
        ax_curve.set_ylabel("Reconstruction Performance (1 - Norm MSE)")
        ax_curve.set_title(f"Circuit Formalization Loop | Target {result['target_id']}")
        ax_curve.set_ylim(-0.35, 1.05)
        ax_curve.grid(True, alpha=0.3)
        ax_curve.legend()

        ax_img.imshow(self.raw_img.resize((224, 224)))
        p_size = 224 // 14

        tr, tc = result["target_id"] // 14, result["target_id"] % 14
        ax_img.add_patch(
            patches.Rectangle(
                (tc * p_size, tr * p_size),
                p_size,
                p_size,
                linewidth=4,
                edgecolor="#F44336",
                facecolor="none",
                label="Target",
            )
        )

        for pid in result["minimal_set"]:
            r, c = pid // 14, pid % 14
            ax_img.add_patch(
                patches.Rectangle(
                    (c * p_size, r * p_size),
                    p_size,
                    p_size,
                    linewidth=2,
                    edgecolor="#4CAF50",
                    facecolor="none",
                    alpha=0.8,
                )
            )

        ax_img.set_title(f"Minimal Sufficient Set (K={len(result['minimal_set'])})")
        ax_img.axis("off")

        plt.tight_layout()
        if os.environ.get("SAVE_PLOT"):
            plt.savefig("formal_circuit.png")
            print("Formal circuit analysis saved to formal_circuit.png")
        else:
            plt.show()


if __name__ == "__main__":
    discoverer = FormalCircuitDiscoverer()
    res = discoverer.discover_minimal_set(target_id=114, threshold=0.85)
    discoverer.visualize_result(res)
