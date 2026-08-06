import os
import torch
from typing import Any, Optional, Dict
from PIL import Image

try:
    import wandb
except ImportError:
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from world_model_lens import HookedWorldModel
from world_model_lens.analysis.layer_cka import LayerCKAAnalyzer


class ConvergenceAuditorCallback:
    """Callback for automated Layer-wise Predictor Convergence reporting during training.
    
    This auditor computes Centered Kernel Alignment (CKA) between layers to track
    if the model's representations are converging semantically over epochs.
    It can log to Weights & Biases or TensorBoard.
    """
    
    def __init__(
        self, 
        world_model: HookedWorldModel,
        validation_batch: torch.Tensor,
        layer_hook_pattern: str = "context_encoder.blocks.{}.hook_resid_post",
        log_freq: int = 1,
        use_wandb: bool = True,
        tensorboard_dir: Optional[str] = None
    ):
        """Initialize the convergence auditor.
        
        Args:
            world_model: The HookedWorldModel instance to evaluate.
            validation_batch: A fixed batch of images [B, C, H, W] to track across epochs.
            layer_hook_pattern: The hook pattern used to extract layer outputs.
            log_freq: Frequency of epochs to run the expensive CKA analysis.
            use_wandb: If True, log plots and metrics to Weights & Biases (if initialized).
            tensorboard_dir: If provided, log to TensorBoard in this directory.
        """
        self.wm = world_model
        self.val_batch = validation_batch
        self.layer_hook_pattern = layer_hook_pattern
        self.log_freq = log_freq
        self.analyzer = LayerCKAAnalyzer(self.wm)
        
        self.use_wandb = use_wandb and wandb is not None and wandb.run is not None
        
        self.tb_writer = None
        if tensorboard_dir is not None and SummaryWriter is not None:
            self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)
            
    def on_epoch_end(self, epoch: int, step: Optional[int] = None) -> Dict[str, float]:
        """Call this method at the end of a training epoch.
        
        Args:
            epoch: Current epoch number.
            step: Optional global step counter.
            
        Returns:
            Dict containing convergence metrics.
        """
        if epoch % self.log_freq != 0:
            return {}
            
        # 1. Compute Layer-wise CKA
        try:
            result = self.analyzer.analyze_layers(
                self.val_batch,
                layer_hook_pattern=self.layer_hook_pattern
            )
        except Exception as e:
            print(f"ConvergenceAuditorCallback failed to analyze layers: {e}")
            return {}
            
        metrics = {
            "eval/semantic_convergence_score": result.semantic_convergence_score,
            "eval/final_layer_cka": float(result.avg_cka_per_layer[-1]) if len(result.avg_cka_per_layer) > 0 else 0.0,
            "eval/initial_layer_cka": float(result.avg_cka_per_layer[0]) if len(result.avg_cka_per_layer) > 0 else 0.0,
        }
        
        # 2. Generate and log convergence plot
        fig = self.analyzer.plot_convergence(result)
        
        if self.use_wandb and fig is not None:
            wandb.log({
                **metrics,
                "eval/convergence_curve": wandb.Image(fig),
                "epoch": epoch
            }, step=step if step is not None else epoch)
            
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("eval/semantic_convergence_score", metrics["eval/semantic_convergence_score"], step or epoch)
            if fig is not None:
                self.tb_writer.add_figure("eval/convergence_curve", fig, step or epoch)
                
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)
            
        return metrics

    def close(self):
        """Cleanup resources like TensorBoard writer."""
        if self.tb_writer is not None:
            self.tb_writer.close()
