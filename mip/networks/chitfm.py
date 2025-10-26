"""The file contains ported transformer models from original Chi's diffusion policy paper.

Author: Chaoyi Pan
Data: 2025-07-23
"""

import copy

import torch
import torch.nn as nn

from mip.embeddings import SUPPORTED_TIMESTEP_EMBEDDING
from mip.networks.base import BaseNetwork


class _SimpleTransformerEncoder(nn.Module):
    """Simple Transformer Encoder without nn.TransformerEncoder for compile compatibility."""

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
        return output


class _SimpleTransformerDecoder(nn.Module):
    """Simple Transformer Decoder without nn.TransformerDecoder for compile compatibility."""

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return output


def _init_weights(module):
    """Weight initialization for transformer components."""
    ignore_types = (
        nn.Dropout,
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
        nn.TransformerEncoder,
        nn.TransformerDecoder,
        nn.ModuleList,
        nn.Mish,
        nn.Sequential,
    )
    if isinstance(module, (nn.Linear, nn.Embedding)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        weight_names = [
            "in_proj_weight",
            "q_proj_weight",
            "k_proj_weight",
            "v_proj_weight",
        ]
        for name in weight_names:
            weight = getattr(module, name)
            if weight is not None:
                torch.nn.init.normal_(weight, mean=0.0, std=0.02)

        bias_names = ["in_proj_bias", "bias_k", "bias_v"]
        for name in bias_names:
            bias = getattr(module, name)
            if bias is not None:
                torch.nn.init.zeros_(bias)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)
    elif isinstance(module, ignore_types):
        # no param
        pass


# Using SinusoidalEmbedding from embeddings module instead of duplicating


class ChiTransformer(BaseNetwork):
    """ChiTransformer with flow matching time inputs and a scalar prediction head.
    condition: (1 + To) | x: (Ta)
    Outputs predicted action sequence and a scalar value.
    """

    def __init__(
        self,
        act_dim: int,
        obs_dim: int,
        Ta: int,
        To: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 8,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
        n_cond_layers: int = 0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: dict | None = None,
        disable_time_embedding: bool = False,
    ):
        # Initialize BaseNetwork with proper parameters
        super().__init__(act_dim, Ta, obs_dim, To, d_model, num_layers)

        # compute number of tokens for main trunk and condition encoder
        T = Ta
        T_cond = 1 + To  # time + observations

        self.Ta = Ta
        self.To = To
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.disable_time_embedding = disable_time_embedding

        # input embedding stem
        self.input_emb = nn.Linear(act_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, d_model))
        self.drop = nn.Dropout(p_drop_emb)

        # condition encoder components
        self.cond_obs_emb = nn.Linear(obs_dim, d_model)

        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, d_model))

        # encoder for conditioning
        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            # Use simple encoder for torch.compile compatibility
            self.encoder = _SimpleTransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_cond_layers
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.Mish(),
                nn.Linear(4 * d_model, d_model),
            )

        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # important for stability
        )
        # Use simple decoder for torch.compile compatibility
        self.decoder = _SimpleTransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_layers
        )

        # Store mask dimensions for dynamic creation
        # We'll create masks dynamically in forward() for better torch.compile compatibility
        self.T = T
        self.T_cond = T_cond

        # decoder head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, act_dim)

        # Mapping for time embeddings (s and t)
        self.timestep_emb_type = timestep_emb_type
        timestep_emb_params = timestep_emb_params or {}
        if not disable_time_embedding:
            # Create embeddings for both s and t
            self.map_s = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                d_model // 2, **timestep_emb_params
            )
            self.map_t = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                d_model // 2, **timestep_emb_params
            )
        else:
            self.map_s = None
            self.map_t = None

        # --- Components for Scalar Head ---
        # Process mean of input action sequence
        self.input_processor = nn.Linear(act_dim, d_model // 4)
        # Process mean of final decoder hidden state
        self.final_processor = nn.Linear(d_model, d_model // 4)
        # Scalar head taking processed input, processed output, and mean condition embedding
        self.scalar_head = nn.Linear(d_model // 4 + d_model // 4 + d_model, 1)

        # init
        self.apply(_init_weights)

        # Initialize positional embeddings
        torch.nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.cond_pos_emb, mean=0.0, std=0.02)

        # Zero-out scalar head for stable training
        nn.init.constant_(self.scalar_head.weight, 0)
        nn.init.constant_(self.scalar_head.bias, 0)

    def _create_masks(self, device):
        """Create attention masks dynamically for better torch.compile compatibility.

        Returns:
            mask: Causal mask for decoder self-attention
            memory_mask: Mask for decoder cross-attention
        """
        # Causal mask for decoder self-attention
        sz = self.T
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, 0.0)
        )

        # Memory mask for decoder cross-attention
        S = self.T_cond
        t_idx, s_idx = torch.meshgrid(
            torch.arange(self.T, device=device),
            torch.arange(S, device=device),
            indexing="ij",
        )
        memory_mask = t_idx >= (s_idx - 1)
        memory_mask = (
            memory_mask.float()
            .masked_fill(memory_mask == 0, float("-inf"))
            .masked_fill(memory_mask == 1, 0.0)
        )

        return mask, memory_mask

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
    ):
        """Forward pass for ChiTransformer with flow matching.

        Input:
            x:          (b, Ta, act_dim) - Target action sequence
            s:          (b, ) - Source time parameter
            t:          (b, ) - Target time parameter
            condition:  (b, To, obs_dim) - Observation sequence condition

        Output:
            y:          (b, Ta, act_dim) - Predicted action sequence
            scalar:     (b, 1) - Predicted scalar value
        """
        b = x.shape[0]
        device = x.device

        if condition is None:
            # Create zero condition if none provided
            condition = torch.zeros((b, self.To, self.obs_dim), device=device)

        # 1. Process input for scalar head
        # Calculate mean over the sequence length (Ta)
        processed_input = self.input_processor(x.mean(dim=1))  # (b, d_model // 4)

        # 2. Prepare time embeddings for s and t
        # Ensure timesteps are properly formatted
        if not torch.is_tensor(s):
            s = torch.tensor([s], dtype=torch.float32, device=device)
        elif len(s.shape) == 0:
            s = s[None].to(device)
        s = s.expand(b)

        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32, device=device)
        elif len(t.shape) == 0:
            t = t[None].to(device)
        t = t.expand(b)

        # Compute and combine time embeddings
        if not self.disable_time_embedding:
            s_emb = self.map_s(s)  # (b, d_model // 2)
            t_emb = self.map_t(t)  # (b, d_model // 2)
            # Concatenate embeddings
            time_emb = torch.cat([s_emb, t_emb], dim=-1).unsqueeze(1)  # (b, 1, d_model)
        else:
            # Use zero embedding when time embedding is disabled
            time_emb = torch.zeros((b, 1, self.d_model), device=device)

        # 3. Process input action sequence
        input_emb = self.input_emb(x)  # (b, Ta, d_model)

        # 4. encoder - process condition sequence
        cond_obs_emb = self.cond_obs_emb(condition)  # (b, To, d_model)
        cond_embeddings = torch.cat(
            [time_emb, cond_obs_emb], dim=1
        )  # (b, 1+To, d_model)
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[
            :, :tc, :
        ]  # each position maps to a (learnable) vector
        cond_input = self.drop(cond_embeddings + position_embeddings)
        memory = self.encoder(cond_input)  # (b, T_cond, d_model)

        # 5. decoder - process action sequence
        token_embeddings = input_emb
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        decoder_input = self.drop(
            token_embeddings + position_embeddings
        )  # (b, T, d_model)

        # Create masks dynamically for torch.compile compatibility
        mask, memory_mask = self._create_masks(device)

        decoder_output = self.decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=mask,
            memory_mask=memory_mask,
        )  # (b, T, d_model)

        # 6. Predict action sequence (main head)
        y = self.ln_f(decoder_output)  # (b, Ta, d_model)
        y = self.head(y)  # (b, Ta, act_dim)

        # 7. Prepare features for scalar head
        # Process mean of the decoder's output sequence
        processed_final_output = self.final_processor(
            decoder_output.mean(dim=1)
        )  # (b, d_model // 4)

        # Use the mean of the encoded condition sequence as the embedding feature
        scalar_emb = memory.mean(dim=1)  # (b, d_model)

        # Concatenate features for the scalar head
        combined_features = torch.cat(
            [processed_input, processed_final_output, scalar_emb], dim=1
        )  # (b, d_model // 4 + d_model // 4 + d_model)

        # 8. Predict scalar value
        scalar_output = self.scalar_head(combined_features)  # (b, 1)

        return y, scalar_output


def test_chitransformer():
    # Define dimensions
    act_dim = 10
    obs_dim = 5
    Ta = 20  # Action sequence length
    To = 5  # Observation sequence length
    d_model = 128
    nhead = 4
    num_layers = 4
    batch_size = 16

    # Instantiate the model
    model = ChiTransformer(
        act_dim=act_dim,
        obs_dim=obs_dim,
        Ta=Ta,
        To=To,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        timestep_emb_type="positional",  # Make sure this matches an entry in SUPPORTED_TIMESTEP_EMBEDDING
    )

    # Create dummy input tensors
    x = torch.randn(batch_size, Ta, act_dim)
    s = torch.rand(batch_size)  # Source time parameter
    t = torch.rand(batch_size)  # Target time parameter
    condition = torch.randn(batch_size, To, obs_dim)

    # Perform forward pass
    predicted_action, predicted_scalar = model(x, s, t, condition)

    # Check output shapes
    print("Input action shape:", x.shape)
    print("Input condition shape:", condition.shape)
    print("Predicted action shape:", predicted_action.shape)
    print("Predicted scalar shape:", predicted_scalar.shape)

    # Expected output shapes:
    # Input action shape: torch.Size([16, 20, 10])
    # Input condition shape: torch.Size([16, 5, 5])
    # Predicted action shape: torch.Size([16, 20, 10])
    # Predicted scalar shape: torch.Size([16, 1])


def test_disable_time_embedding():
    """Test ChiTransformer with disable_time_embedding=True"""
    print("=" * 50)
    print("Testing disable_time_embedding for ChiTransformer")
    print("=" * 50)

    # Test parameters
    act_dim = 4
    obs_dim = 3
    Ta = 8
    To = 2
    batch_size = 2

    print("\n1. Testing ChiTransformer with disable_time_embedding=True")
    try:
        model = ChiTransformer(
            act_dim=act_dim,
            obs_dim=obs_dim,
            Ta=Ta,
            To=To,
            d_model=64,
            nhead=2,
            num_layers=2,
            disable_time_embedding=True,
        )

        x = torch.randn(batch_size, Ta, act_dim)
        s = torch.randn(batch_size)
        t = torch.randn(batch_size)
        condition = torch.randn(batch_size, To, obs_dim)

        y1, scalar1 = model(x, s, t, condition)
        # Test with different time values - should give same output
        y2, scalar2 = model(
            x, torch.randn(batch_size), torch.randn(batch_size), condition
        )

        print(f"   Output shapes: y={y1.shape}, scalar={scalar1.shape}")
        print(
            f"   Noise invariant: {torch.allclose(y1, y2, atol=1e-6) and torch.allclose(scalar1, scalar2, atol=1e-6)}"
        )

        # Test without condition
        y_no_cond, scalar_no_cond = model(x, s, t, None)
        print(
            f"   Works without condition: y={y_no_cond.shape}, scalar={scalar_no_cond.shape}"
        )

    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n" + "=" * 50)
    print("disable_time_embedding test completed!")
    print("=" * 50)


if __name__ == "__main__":
    test_chitransformer()
    test_disable_time_embedding()
