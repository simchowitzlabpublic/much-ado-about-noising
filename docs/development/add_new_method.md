# Add New Training Method

Add new training method:
1. Add loss function to [mip/losses.py](../../mip/losses.py)
2. Add corresponding sampler to [mip/samplers.py](../../mip/samplers.py)
3. Register both in `get_loss_fn()` and `get_sampler()`
4. Test with unit tests

## Samplers

Samplers in [mip/samplers.py](../../mip/samplers.py) implement inference-time sampling strategies that correspond to each loss type:

```python
def sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,  # Initial noise
    obs: torch.Tensor,    # Observations
) -> torch.Tensor:
    """
    Returns:
        action: Predicted action trajectory
    """
```

## Losses

All loss functions in [mip/losses.py](../../mip/losses.py) follow a consistent signature:

```python
def loss_fn(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Returns:
        loss: Scalar loss tensor
        info: Dict with additional metrics for logging
    """
```
