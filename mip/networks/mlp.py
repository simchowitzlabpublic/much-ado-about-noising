"""MLP network."""

import torch
import torch.nn as nn

from mip.networks.base import BaseNetwork


class VanillaMLP(BaseNetwork):
    """Simplified MLP without fancy time embedding, LayerNorm, or ApproxGELU.

    Uses s and t directly by concatenating them with the input.
    Can optionally use expansion_factor to match original MLP capacity.
    """

    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        emb_dim: int = 512,
        n_layers: int = 6,
        dropout: float = 0.1,
        expansion_factor: int = 1,  # Set to 4 to match original MLP
    ):
        """Args:
            act_dim: (int) - action dimension
            Ta: (int) - action sequence length
            obs_dim: (int) - observation dimension
            To: (int) - observation sequence length
            emb_dim: (int) - embedding dimension
            n_layers: (int) - number of layers
            dropout: (float) - dropout rate
            expansion_factor: (int) - expansion factor for hidden layers.

        Returns:
            None
        """
        super().__init__(act_dim, Ta, obs_dim, To, emb_dim, n_layers)
        self.dropout = dropout
        self.expansion_factor = expansion_factor

        # Input: action sequence + observation sequence + s + t (direct concatenation)
        input_dim = act_dim * Ta + obs_dim * To + 2  # +2 for s and t
        output_dim = act_dim * Ta

        # Simple MLP layers with ReLU activation
        layers = []
        layers.append(nn.Linear(input_dim, emb_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers with optional expansion
        if expansion_factor > 1:
            # Use expansion like original MLP: expand → contract
            for _ in range(self.n_layers):
                layers.append(nn.Linear(emb_dim, emb_dim * expansion_factor))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(emb_dim * expansion_factor, emb_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        else:
            # No expansion, simple layers
            for _ in range(self.n_layers):
                layers.append(nn.Linear(emb_dim, emb_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

        # Output layers: one for the main output and one for the scalar
        self.main_output = nn.Linear(emb_dim, output_dim)
        self.scalar_output = nn.Linear(emb_dim, 1)

        # Simple Xavier initialization
        self._init_weights()

    def _init_weights(self):
        """Apply Xavier initialization to all linear layers."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

        nn.init.xavier_uniform_(self.main_output.weight)
        nn.init.constant_(self.main_output.bias, 0)

        # Zero-out scalar head
        nn.init.constant_(self.scalar_output.weight, 0)
        nn.init.constant_(self.scalar_output.bias, 0)

    def forward(
        self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ):
        """Args:
            x: (b, Ta, act_dim)
            s: (b, ) - concatenated directly
            t: (b, ) - concatenated directly
            condition: (b, To, obs_dim).

        Returns:
            output_data: (b, Ta, act_dim)
            scalar_output: (b, 1)
        """
        # Flatten inputs and concatenate s, t directly
        x_flat = torch.flatten(x, 1)  # (b, Ta * act_dim)
        condition_flat = torch.flatten(condition, 1)  # (b, To * obs_dim)
        s_expanded = s.unsqueeze(-1)  # (b, 1)
        t_expanded = t.unsqueeze(-1)  # (b, 1)
        input_data = torch.cat([x_flat, condition_flat, s_expanded, t_expanded], dim=-1)

        # Pass through MLP
        features = self.mlp(input_data)

        # Generate both outputs
        main_output = self.main_output(features)  # (b, Ta * act_dim)
        scalar_output = self.scalar_output(features)  # (b, 1)

        # Reshape main output to (b, Ta, act_dim)
        main_output = main_output.view(x.shape[0], x.shape[1], self.act_dim)

        return main_output, scalar_output


class MLPResidualBlock(nn.Module):
    """Residual block with LayerNorm and ApproxGELU activation."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 4)  # Expand to 4x dimension
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)  # Contract back to original dimension
        self.dropout2 = nn.Dropout(dropout)

        # Orthogonal initialization
        self._init_weights()

    def _init_weights(self):
        """Apply orthogonal initialization to linear layers."""
        nn.init.orthogonal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.orthogonal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x):
        # First sub-layer with residual connection
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.norm2(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        return x + residual


class MLP(BaseNetwork):
    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        emb_dim: int = 512,  # Increased default size for larger networks
        n_layers: int = 6,  # Increased default depth for larger networks
        timestep_emb_dim: int = 128,
        max_freq: float = 100.0,
        disable_time_embedding: bool = False,
        dropout: float = 0.1,  # Added dropout parameter
    ):
        super().__init__(act_dim, Ta, obs_dim, To, emb_dim, n_layers)
        self.timestep_emb_dim = timestep_emb_dim
        self.disable_time_embedding = disable_time_embedding
        self.dropout = dropout

        # Create uniform frequencies between 0 and max_freq for time embeddings
        # timestep_emb_dim should be even so we can split between sin and cos
        assert timestep_emb_dim % 2 == 0, "timestep_emb_dim must be even"
        num_frequencies = timestep_emb_dim // 2

        # Create uniformly spaced frequencies from 0 to max_freq
        frequencies = torch.linspace(0, max_freq, num_frequencies)
        self.register_buffer(
            "frequencies", frequencies
        )  # Register as buffer so it moves with device

        # Input size calculation depends on whether time embedding is disabled
        time_emb_size = 0 if disable_time_embedding else 2 * timestep_emb_dim
        input_dim = act_dim * Ta + time_emb_size + obs_dim * To
        output_dim = act_dim * Ta

        # Input projection layer with LayerNorm and orthogonal initialization
        self.input_proj = nn.Linear(input_dim, emb_dim)
        self.input_norm = nn.LayerNorm(emb_dim)
        self.input_activation = nn.GELU()

        # Stack of residual blocks
        self.residual_blocks = nn.ModuleList(
            [MLPResidualBlock(emb_dim, dropout) for _ in range(self.n_layers)]
        )

        # Final layer norm before output heads
        self.final_norm = nn.LayerNorm(emb_dim)

        # Output layers: one for the main output and one for the 1D vector
        self.main_output = nn.Linear(emb_dim, output_dim)
        self.scalar_output = nn.Linear(emb_dim, 1)  # Additional head for 1D output

        # Apply orthogonal initialization
        self._init_weights()

    def _init_weights(self):
        """Apply orthogonal initialization to all linear layers."""
        # Input projection
        nn.init.orthogonal_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)

        # Output layers
        nn.init.orthogonal_(self.main_output.weight)
        nn.init.constant_(self.main_output.bias, 0)

        # Zero-out scalar head (as before)
        nn.init.constant_(self.scalar_output.weight, 0)
        nn.init.constant_(self.scalar_output.bias, 0)

    def _embed_time(self, t):
        """Embed time values using uniform frequencies and sin/cos functions.

        Args:
            t: (batch_size,) tensor of time values

        Returns:
            embedded: (batch_size, timestep_emb_dim) tensor of embedded time values
        """
        # t: (batch_size,) -> (batch_size, 1)
        t = t.unsqueeze(-1)

        # Compute t * frequencies: (batch_size, 1) * (num_frequencies,) -> (batch_size, num_frequencies)
        angles = t * self.frequencies.unsqueeze(0)

        # Apply sin and cos
        sin_embed = torch.sin(angles)  # (batch_size, num_frequencies)
        cos_embed = torch.cos(angles)  # (batch_size, num_frequencies)

        # Concatenate sin and cos embeddings
        embedded = torch.cat(
            [sin_embed, cos_embed], dim=-1
        )  # (batch_size, timestep_emb_dim)

        return embedded

    def forward(
        self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ):
        """Args:
            x: (b, Ta, act_dim)
            s: (b, )
            t: (b, )
            condition: (b, To, obs_dim).

        Returns:
            output_data: (b, Ta, act_dim)
            scalar_output: (b, 1)
        """
        x_flat = torch.flatten(x, 1)  # (b, Ta * act_dim)

        if not self.disable_time_embedding:
            # Apply uniform frequency embeddings to time inputs
            s_embedded = self._embed_time(s)  # (b, timestep_emb_dim)
            t_embedded = self._embed_time(t)  # (b, timestep_emb_dim)

            condition_flat = torch.flatten(condition, 1)  # (b, To * obs_dim)
            input_data = torch.cat(
                [x_flat, s_embedded, t_embedded, condition_flat], dim=-1
            )  # (b, Ta * act_dim + 2 * timestep_emb_dim + To * obs_dim)
        else:
            # Skip time embeddings when disabled
            condition_flat = torch.flatten(condition, 1)  # (b, To * obs_dim)
            input_data = torch.cat(
                [x_flat, condition_flat], dim=-1
            )  # (b, Ta * act_dim + To * obs_dim)

        # Input projection with LayerNorm and ApproxGELU
        features = self.input_proj(input_data)  # (b, emb_dim)
        features = self.input_norm(features)
        features = self.input_activation(features)

        # Pass through residual blocks
        for block in self.residual_blocks:
            features = block(features)

        # Final layer normalization
        features = self.final_norm(features)

        # Generate both outputs
        main_output = self.main_output(features)  # (b, Ta * act_dim)
        scalar_output = self.scalar_output(features)  # (b, 1)

        # reshape main output to (b, Ta, act_dim)
        main_output = main_output.view(x.shape[0], x.shape[1], self.act_dim)

        return main_output, scalar_output
