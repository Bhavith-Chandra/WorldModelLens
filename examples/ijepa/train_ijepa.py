import torch
import torch.nn.functional as F
import torch.optim as optim
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from image_utils import get_sample_image, preprocess_image
import time
import os

def train_ijepa_mini(epochs=300, steps_per_epoch=4, lr=1e-4):
    print("Starting I-JEPA Mini-Training with official IJEPAAdapter...")
    
    config = WorldModelConfig(
        backend="ijepa",
        d_embed=192,
        n_layers=6,
        n_heads=3,
        predictor_embed_dim=384,
        predictor_depth=4,
        predictor_heads=6
    )
    
    model = IJEPAAdapter(config)
    model.train()
    
    # We only optimize the context encoder and predictor
    # target_encoder is updated via EMA from context_encoder
    params = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    optimizer = optim.Adam(params, lr=lr)
    
    # We'll use a few "scenes" for training
    raw_img = get_sample_image()
    img_tensor = preprocess_image(raw_img)
    
    losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            
            # Forward pass: compute_loss handles structured masking internally
            # It uses the target encoder for GT and predicts from the context encoder
            loss = model.compute_loss(img_tensor)
            
            loss.backward()
            optimizer.step()
            
            # EMA Update for Target Encoder
            model.update_target_encoder(momentum=0.999)
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / steps_per_epoch
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Elapsed: {time.time()-start_time:.1f}s")

    # Save weights
    checkpoint_path = "ijepa_mini.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nTraining Complete. Model saved to {checkpoint_path}")
    
    return losses

if __name__ == "__main__":
    train_ijepa_mini(epochs=300)
