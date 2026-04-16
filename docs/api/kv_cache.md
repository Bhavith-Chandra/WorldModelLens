# KV Cache Hook

Transformer-style adapters often use an explicit key/value (KV) memory that
accumulates entries over timesteps (keys and values per layer). World Model
Lens exposes a dedicated hook point named `kv_cache` that runs once per
timestep and receives the full `ActivationCache`. This allows users to inspect
and mutate a model's KV memory (for example, to "forget" a key) without
re-running the forward pass up to that point.

The hook is available via the existing `HookPoint` abstraction. Unlike the
usual tensor hooks, `kv_cache` hooks receive `(ActivationCache, HookContext)`.

Usage

```python
from world_model_lens import HookPoint

def my_kv_hook(cache, ctx):
    # Prefer helper methods instead of mutating _store directly.
    # Delete the key for layer 0 at this timestep.
    cache.delete_kv(0, "k", ctx.timestep)

# Register for a single timestep
wm.add_hook(HookPoint(name="kv_cache", fn=my_kv_hook, timestep=10))

# Or register across a time slice
wm.add_hook(HookPoint(name="kv_cache", fn=my_kv_hook, time_slice=[5, 12]))
```

Best practices

- Prefer creating small, focused hooks that perform a single mutation (e.g., delete a key).
- Hooks are executed in registration order; use `prepend=True` via the `temp_hooks`
  context manager if you need to ensure ordering relative to other hooks.
- The current implementation calls hook functions with the cache object and
  allows in-place mutation of `ActivationCache._store`. Consider adding
  helper methods on `ActivationCache` (`get_kv`, `set_kv`, `delete_kv`) for a
  cleaner API (see README examples).

Notes

- `HookPoint.fn` remains the same dataclass but is allowed to accept
  `(ActivationCache, HookContext)` for `kv_cache` hooks. The system will call
  the function with these arguments when the `kv_cache` component fires.
