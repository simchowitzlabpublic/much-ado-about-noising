# Add New Network Architecture

1. Create network file in [mip/networks/](../../mip/networks/)
2. Inherit from `BaseNetwork`
3. Implement `forward(x, s, t, condition)` returning `(action, scalar)`
4. Register in `network_utils.get_network()`
5. Create config file in `examples/configs/network/`

## Base Network Interface

All networks in [mip/networks/](../../mip/networks/) inherit from `BaseNetwork` and implement:

```python
class BaseNetwork(nn.Module):
    def forward(
        self,
        x: torch.Tensor,      # Action trajectory (B, T, action_dim)
        s: torch.Tensor,      # Source time (B,)
        t: torch.Tensor,      # Target time (B,)
        condition: torch.Tensor,  # Encoded observations (B, cond_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action: Predicted action (B, T, action_dim)
            scalar: Auxiliary scalar prediction (B,)
        """
```

**Dual Output Design**:
- Primary output: Action prediction
- Secondary output: Scalar value (can be used for value estimation, uncertainty, etc.)
- Enhances learning signal without additional architectural complexity

## Available Networks

1. **MLP** ([mip/networks/mlp.py](../../mip/networks/mlp.py)):
   - Multi-layer perceptron with timestep embeddings
   - Simple, fast, works well for state observations
   - Configuration: `emb_dim`, `num_layers`, `dropout`

2. **Vanilla MLP** ([mip/networks/mlp.py](../../mip/networks/mlp.py)):
   - Even simpler MLP without special embeddings
   - Baseline for ablation studies

3. **ChiUNet** ([mip/networks/chiunet.py](../../mip/networks/chiunet.py)):
   - U-Net architecture from Diffusion Policy (Chi et al.)
   - Temporal convolutions with skip connections
   - Excellent for action trajectory modeling

4. **JannerUNet** ([mip/networks/jannerunet.py](../../mip/networks/jannerunet.py)):
   - U-Net from Decision Diffuser (Janner et al.)
   - Alternative U-Net design

5. **ChiTransformer** ([mip/networks/chitfm.py](../../mip/networks/chitfm.py)):
   - Transformer architecture from Diffusion Policy
   - Self-attention over action sequence
   - Better for long horizons

6. **SudeepDiT** ([mip/networks/sudeepdit.py](../../mip/networks/sudeepdit.py)):
   - Diffusion Transformer (DiT) architecture
   - State-of-the-art generative modeling
   - Scalable to large models

7. **RNN** ([mip/networks/rnn.py](../../mip/networks/rnn.py)):
   - LSTM/GRU-based recurrent networks
   - Sequential processing of action trajectories
   - Configuration: `rnn_type` (LSTM or GRU)

## Network Selection

Networks are instantiated via `network_utils.get_network()`:

```python
from mip.network_utils import get_network

net = get_network(
    network_config=config.network,
    task_config=config.task,
)
```

The function automatically handles:
- Input/output dimension inference from task config
- Conditional dimension calculation based on encoder
- Network-specific parameter initialization
