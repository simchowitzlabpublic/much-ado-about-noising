"""DiT network used in DiT Policy by Sudeep et al.

The file contains ported DiT models from original DiT paper.

Author: Chaoyi Pan
Data: 2025-07-23
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mip.embeddings import SUPPORTED_TIMESTEP_EMBEDDING
from mip.networks.base import BaseNetwork


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def _with_pos_embed(tensor, pos=None):
    return tensor if pos is None else tensor + pos


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Args:
            x: Tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor of shape (seq_len, batch_size, d_model) with positional encodings added
        """
        pe = self.pe[: x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()


class _TimeNetwork(nn.Module):
    def __init__(self, time_dim, out_dim, learnable_w=False):
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        assert len(x.shape) == 1, "assumes 1d input timestep array"
        x = x[:, None] * self.w[None]
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=1)
        return self.out_net(x)


class _SelfAttnEncoder(nn.Module):
    def __init__(
        self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos):
        q = k = _with_pos_embed(src, pos)
        src2, _ = self.self_attn(q, k, value=src, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None] + self.shift(c)[None]

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.scale.weight)
        nn.init.xavier_uniform_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class _DiTDecoder(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # create modulation layers
        self.attn_mod1 = _ShiftScaleMod(d_model)
        self.attn_mod2 = _ZeroScaleMod(d_model)
        self.mlp_mod1 = _ShiftScaleMod(d_model)
        self.mlp_mod2 = _ZeroScaleMod(d_model)

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = torch.mean(cond, axis=0)
        cond = cond + t

        x2 = self.attn_mod1(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = self.attn_mod2(self.dropout1(x2), cond) + x

        x2 = self.mlp_mod1(self.norm2(x), cond)
        x2 = self.linear2(self.dropout2(self.activation(self.linear1(x2))))
        x2 = self.mlp_mod2(self.dropout3(x2), cond)
        return x + x2

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for s in (self.attn_mod1, self.attn_mod2, self.mlp_mod1, self.mlp_mod2):
            s.reset_parameters()


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = torch.mean(cond, axis=0)
        cond = cond + t

        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = x * scale[None] + shift[None]
        x = self.linear(x)
        return x.transpose(0, 1)

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


class _TransformerEncoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_module) for _ in range(num_layers)]
        )

        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, src, pos):
        x, outputs = src, []
        for layer in self.layers:
            x = layer(x, pos)
            outputs.append(x)
        return outputs


class _TransformerDecoder(_TransformerEncoder):
    def forward(self, src, t, all_conds):
        x = src
        for layer, cond in zip(self.layers, all_conds, strict=False):
            x = layer(x, t, cond)
        return x


class SudeepDiT(BaseNetwork):
    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: dict | None = None,
        disable_time_embedding: bool = False,
    ):
        # BaseNetwork expects: act_dim, Ta, obs_dim, To, emb_dim, n_layers
        emb_dim = d_model  # Use d_model as embedding dimension
        super().__init__(act_dim, Ta, obs_dim, To, emb_dim, depth)

        self.act_dim = act_dim
        self.Ta = Ta
        self.obs_dim = obs_dim
        self.To = To
        self.d_model = d_model
        self.disable_time_embedding = disable_time_embedding

        # Time embeddings for s and t
        timestep_emb_params = timestep_emb_params or {}
        if not disable_time_embedding:
            # Use SUPPORTED_TIMESTEP_EMBEDDING for s and t
            self.map_s = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                d_model // 2, **timestep_emb_params
            )
            self.map_t = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                d_model // 2, **timestep_emb_params
            )
        else:
            self.map_s = None
            self.map_t = None

        # Input projection
        self.x_proj = nn.Sequential(
            nn.Linear(act_dim, act_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(act_dim, d_model),
        )

        # Condition projection (from obs_dim * To to d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(obs_dim * To, d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_model, d_model),
        )

        # Positional encoding
        self.pos_enc = _PositionalEncoding(d_model)
        self.register_parameter(
            "pos_emb",
            nn.Parameter(torch.empty(Ta, 1, d_model), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.pos_emb.data)

        # Encoder blocks
        encoder_module = _SelfAttnEncoder(
            d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = _TransformerEncoder(encoder_module, depth)

        # Decoder blocks
        decoder_module = _DiTDecoder(
            d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
        )
        self.decoder = _TransformerDecoder(decoder_module, depth)

        # Output layers
        self.final_layer = _FinalLayer(d_model, act_dim)

        # Scalar output head
        self.scalar_head = nn.Sequential(
            nn.Linear(act_dim * 2 + d_model, d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_model, 1),
        )

        print(
            f"number of DiT parameters: {sum(p.numel() for p in self.parameters()):e}"
        )

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
    ):
        """Input:
            x:          (b, Ta, act_dim)
            s:          (b, ) - source time parameter
            t:          (b, ) - target time parameter
            condition:  (b, To, obs_dim) or None
        Output:
            y:          (b, Ta, act_dim) - predicted action sequence
            scalar:     (b, 1) - predicted scalar value
        """
        batch_size, Ta, act_dim = x.shape
        device = x.device

        # Process time embeddings
        if self.map_s is not None and self.map_t is not None:
            s_emb = self.map_s(s)  # (b, d_model // 2)
            t_emb = self.map_t(t)  # (b, d_model // 2)
            time_emb = torch.cat([s_emb, t_emb], dim=-1)  # (b, d_model)
        else:
            time_emb = torch.zeros(batch_size, self.d_model, device=device)

        # Process condition
        if condition is not None:
            cond_flat = torch.flatten(condition, 1)  # (b, To * obs_dim)
            cond_emb = self.cond_proj(cond_flat)  # (b, d_model)
        else:
            # Use zero condition if none provided
            cond_emb = torch.zeros(batch_size, self.d_model, device=device)

        # Combine time and condition embeddings
        combined_emb = time_emb + cond_emb  # (b, d_model)

        # Project input and add positional embeddings
        x_tokens = self.x_proj(x)  # (b, Ta, d_model)
        x_tokens = x_tokens.transpose(0, 1)  # (Ta, b, d_model)
        x_tokens = x_tokens + self.pos_emb[:Ta]  # Add positional embeddings

        # Create condition sequence for encoder (time embedding as "observation")
        time_emb_expanded = combined_emb.unsqueeze(0)  # (1, b, d_model)
        cond_seq = time_emb_expanded  # (1, b, d_model)

        # Encode condition
        pos = self.pos_enc(cond_seq)
        enc_cache = self.encoder(cond_seq, pos)

        # Decode
        y_tokens = self.decoder(x_tokens, combined_emb, enc_cache)

        # Final output layer
        y = self.final_layer(y_tokens, combined_emb, enc_cache[-1])  # (b, Ta, act_dim)

        # Compute scalar output
        # Use input mean, output mean, and time embedding
        x_mean = x.mean(dim=1)  # (b, act_dim)
        y_mean = y.mean(dim=1)  # (b, act_dim)
        scalar_features = torch.cat(
            [x_mean, y_mean, combined_emb], dim=1
        )  # (b, 2*act_dim + d_model)
        scalar = self.scalar_head(scalar_features)  # (b, 1)

        return y, scalar


def test_sudeepdit():
    """Test SudeepDiT network"""
    print("=" * 50)
    print("Testing SudeepDiT")
    print("=" * 50)

    act_dim = 2
    Ta = 4
    obs_dim = 3
    To = 2
    batch_size = 4

    model = SudeepDiT(
        act_dim=act_dim,
        Ta=Ta,
        obs_dim=obs_dim,
        To=To,
        d_model=128,
        n_heads=4,
        depth=2,  # Reduced for testing
        dropout=0.1,
        timestep_emb_type="positional",
    )

    x = torch.randn(batch_size, Ta, act_dim)
    s = torch.randn(batch_size)
    t = torch.randn(batch_size)
    condition = torch.randn(batch_size, To, obs_dim)

    y, scalar_out = model(x, s, t, condition)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Scalar output shape: {scalar_out.shape}")

    # Test without condition
    y_no_cond, scalar_no_cond = model(x, s, t, None)
    print(f"Output without condition shape: {y_no_cond.shape}")
    print(f"Scalar without condition shape: {scalar_no_cond.shape}")

    # Test with disable_time_embedding=True
    print("\nTesting with disable_time_embedding=True:")
    model_no_time = SudeepDiT(
        act_dim=act_dim,
        Ta=Ta,
        obs_dim=obs_dim,
        To=To,
        d_model=128,
        n_heads=4,
        depth=2,
        disable_time_embedding=True,
    )

    y1, s1 = model_no_time(x, s, t, condition)
    # Test with different time values - should give same output
    y2, s2 = model_no_time(
        x, torch.randn(batch_size), torch.randn(batch_size), condition
    )

    print(
        f"Time invariant: {torch.allclose(y1, y2, atol=1e-6) and torch.allclose(s1, s2, atol=1e-6)}"
    )

    print("=" * 50)
    print("SudeepDiT test completed!")
    print("=" * 50)


if __name__ == "__main__":
    test_sudeepdit()
