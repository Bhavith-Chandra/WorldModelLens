# Backends API

Adapters for connecting any world model architecture to the World Model Lens analysis tools.
All adapters extend `BaseModelAdapter` and only require `encode()` and `dynamics()` to be implemented.

## Base

### BaseModelAdapter

```{eval-rst}
.. automodule:: world_model_lens.backends.base_adapter
   :members:
   :undoc-members:
   :show-inheritance:
```

### GenericAdapter

```{eval-rst}
.. automodule:: world_model_lens.backends.base_adapter
   :members:
   :undoc-members:
   :show-inheritance:
```

### Registry

```{eval-rst}
.. automodule:: world_model_lens.backends.registry
   :members:
   :undoc-members:
   :show-inheritance:
```

---

## RL World Models

### DreamerV3

```{eval-rst}
.. automodule:: world_model_lens.backends.dreamerv3
   :members:
   :undoc-members:
   :show-inheritance:
```

### DreamerV2

```{eval-rst}
.. automodule:: world_model_lens.backends.dreamerv2
   :members:
   :undoc-members:
   :show-inheritance:
```

### DreamerV1

```{eval-rst}
.. automodule:: world_model_lens.backends.dreamerv1
   :members:
   :undoc-members:
   :show-inheritance:
```

### RSSM

```{eval-rst}
.. automodule:: world_model_lens.backends.rssm
   :members:
   :undoc-members:
   :show-inheritance:
```

### IRIS

```{eval-rst}
.. automodule:: world_model_lens.backends.iris
   :members:
   :undoc-members:
   :show-inheritance:
```

### TD-MPC2

```{eval-rst}
.. automodule:: world_model_lens.backends.tdmpc2
   :members:
   :undoc-members:
   :show-inheritance:
```

### PlaNet

```{eval-rst}
.. automodule:: world_model_lens.backends.planet
   :members:
   :undoc-members:
   :show-inheritance:
```

### Ha & Schmidhuber

```{eval-rst}
.. automodule:: world_model_lens.backends.ha_schmidhuber
   :members:
   :undoc-members:
   :show-inheritance:
```

### Decision Transformer

```{eval-rst}
.. automodule:: world_model_lens.backends.decision_transformer
   :members:
   :undoc-members:
   :show-inheritance:
```

### Contrastive Predictive

```{eval-rst}
.. automodule:: world_model_lens.backends.contrastive_predictive
   :members:
   :undoc-members:
   :show-inheritance:
```

---

## Video / Non-RL

### Video Adapter

```{eval-rst}
.. automodule:: world_model_lens.backends.video_adapter
   :members:
   :undoc-members:
   :show-inheritance:
```

### Video World Model

```{eval-rst}
.. automodule:: world_model_lens.backends.video_world_model
   :members:
   :undoc-members:
   :show-inheritance:
```

### Toy Video Model

```{eval-rst}
.. automodule:: world_model_lens.backends.toy_video_model
   :members:
   :undoc-members:
   :show-inheritance:
```

### Toy Scientific Model

```{eval-rst}
.. automodule:: world_model_lens.backends.toy_scientific_model
   :members:
   :undoc-members:
   :show-inheritance:
```

### Planning Adapter

```{eval-rst}
.. automodule:: world_model_lens.backends.planning_adapter
   :members:
   :undoc-members:
   :show-inheritance:
```

### Diffusion

```{eval-rst}
.. automodule:: world_model_lens.backends.diffusion
   :members:
   :undoc-members:
   :show-inheritance:
```

---

## Specialized

### Transformer World Model

```{eval-rst}
.. automodule:: world_model_lens.backends.transformer_world_model
   :members:
   :undoc-members:
   :show-inheritance:
```

### Robotics

```{eval-rst}
.. automodule:: world_model_lens.backends.robotics
   :members:
   :undoc-members:
   :show-inheritance:
```

### Autonomous Driving

```{eval-rst}
.. automodule:: world_model_lens.backends.autonomous_driving
   :members:
   :undoc-members:
   :show-inheritance:
```

---

## Custom Adapters

### Template

```{eval-rst}
.. automodule:: world_model_lens.backends.custom_adapter_template
   :members:
   :undoc-members:
   :show-inheritance:
```
