# Getting Started

## Install

Use a local editable install from this repository:

```bash
pip install -e .
```

## Minimal Example

```python
import torch
from world_model_lens import WorldModelConfig
from world_model_lens.backends.base_adapter import BaseModelAdapter

class MyWorldModel(BaseModelAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = torch.nn.Linear(config.d_obs, config.d_h)
        self.transition_layer = torch.nn.Linear(config.d_h, config.d_h)

    def encode(self, obs, context=None):
        state = self.encoder(obs)
        return state, state

    def dynamics(self, state, action=None):
        return self.transition_layer(state)

config = WorldModelConfig(d_obs=128, d_h=64)
model = MyWorldModel(config)

obs = torch.randn(1, 128)
posterior, encoding = model.encode(obs)
```
