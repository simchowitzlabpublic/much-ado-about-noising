"""RNN network for policy learning"""

import numpy as np
import torch
import torch.nn as nn

from mip.networks.base import BaseNetwork


class ApproxGELU(nn.Module):
    """Approximate GELU activation function for better performance."""

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
            )
        )


class MLPResidualBlock(nn.Module):
    """Residual block with LayerNorm and ApproxGELU activation."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 4)  # Expand to 4x dimension
        self.activation = ApproxGELU()
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


class RNN(BaseNetwork):
    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 2,
        rnn_type: str = "LSTM",  # [LSTM, GRU]
        timestep_emb_dim: int = 128,
        max_freq: float = 100.0,
        disable_time_embedding: bool = False,
        dropout: float = 0.1,
        mlp_layer_dims: list = None,  # Hidden dims for final MLP layers
    ):
        # BaseNetwork expects: act_dim, Ta, obs_dim, To, emb_dim, n_layers
        super().__init__(act_dim, Ta, obs_dim, To, rnn_hidden_dim, rnn_num_layers)
        self.act_dim = act_dim
        self.Ta = Ta
        self.obs_dim = obs_dim
        self.To = To
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.rnn_type = rnn_type
        self.timestep_emb_dim = timestep_emb_dim
        self.disable_time_embedding = disable_time_embedding
        self.dropout = dropout

        # Create uniform frequencies between 0 and max_freq for time embeddings
        assert timestep_emb_dim % 2 == 0, "timestep_emb_dim must be even"
        num_frequencies = timestep_emb_dim // 2
        frequencies = torch.linspace(0, max_freq, num_frequencies)
        self.register_buffer("frequencies", frequencies)

        # Input size calculation depends on whether time embedding is disabled
        time_emb_size = 0 if disable_time_embedding else 2 * timestep_emb_dim
        # For RNN, we process each timestep individually
        input_dim_per_timestep = act_dim + time_emb_size + obs_dim * To

        # RNN layer
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_dim_per_timestep,
                hidden_size=rnn_hidden_dim,
                num_layers=rnn_num_layers,
                dropout=dropout if rnn_num_layers > 1 else 0,
                batch_first=True,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_dim_per_timestep,
                hidden_size=rnn_hidden_dim,
                num_layers=rnn_num_layers,
                dropout=dropout if rnn_num_layers > 1 else 0,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        # MLP layers after RNN
        if mlp_layer_dims is None:
            mlp_layer_dims = [rnn_hidden_dim, rnn_hidden_dim // 2]

        mlp_layers = []
        prev_dim = rnn_hidden_dim
        for dim in mlp_layer_dims:
            mlp_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.LayerNorm(dim),
                    ApproxGELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = dim

        self.mlp = nn.Sequential(*mlp_layers)

        # Output layers: one for the main output and one for the scalar
        self.main_output = nn.Linear(prev_dim, act_dim)
        self.scalar_output = nn.Linear(prev_dim, 1)

        # Apply orthogonal initialization
        self._init_weights()

    def _init_weights(self):
        """Apply orthogonal initialization to linear layers."""
        # RNN initialization
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)

        # MLP initialization
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0)

        # Output layers
        nn.init.orthogonal_(self.main_output.weight)
        nn.init.constant_(self.main_output.bias, 0)

        # Zero-out scalar head
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
            condition: (b, To, obs_dim)

        Returns:
            output_data: (b, Ta, act_dim)
            scalar_output: (b, 1)
        """
        batch_size, Ta, act_dim = x.shape

        if not self.disable_time_embedding:
            # Apply uniform frequency embeddings to time inputs
            s_embedded = self._embed_time(s)  # (b, timestep_emb_dim)
            t_embedded = self._embed_time(t)  # (b, timestep_emb_dim)
            time_emb = torch.cat(
                [s_embedded, t_embedded], dim=-1
            )  # (b, 2 * timestep_emb_dim)
        else:
            # Use zero embedding when disabled
            time_emb = torch.zeros((batch_size, 0), device=x.device)

        # Handle condition - use zeros if None
        if condition is not None:
            condition_flat = torch.flatten(condition, 1)  # (b, To * obs_dim)
        else:
            # Create zeros for missing condition to maintain consistent input size
            condition_flat = torch.zeros(
                batch_size, self.To * self.obs_dim, device=x.device, dtype=x.dtype
            )

        # Prepare input for RNN: for each timestep, concatenate action, time_emb, and condition
        rnn_inputs = []
        for i in range(Ta):
            timestep_input = [x[:, i, :]]  # (b, act_dim)

            if not self.disable_time_embedding:
                timestep_input.append(time_emb)  # (b, 2 * timestep_emb_dim)

            timestep_input.append(condition_flat)  # (b, To * obs_dim)

            # Concatenate all inputs for this timestep
            timestep_input = torch.cat(
                timestep_input, dim=-1
            )  # (b, input_dim_per_timestep)
            rnn_inputs.append(timestep_input)

        # Stack to create sequence: (b, Ta, input_dim_per_timestep)
        rnn_input_seq = torch.stack(rnn_inputs, dim=1)

        # Pass through RNN
        rnn_output, _ = self.rnn(rnn_input_seq)  # (b, Ta, rnn_hidden_dim)

        # Pass through MLP for each timestep
        # Reshape to (b * Ta, rnn_hidden_dim) for efficient processing
        rnn_output_flat = rnn_output.reshape(-1, self.rnn_hidden_dim)
        mlp_output_flat = self.mlp(rnn_output_flat)  # (b * Ta, prev_dim)

        # Generate main output for each timestep
        main_output_flat = self.main_output(mlp_output_flat)  # (b * Ta, act_dim)
        main_output = main_output_flat.reshape(
            batch_size, Ta, act_dim
        )  # (b, Ta, act_dim)

        # Generate scalar output from the final timestep
        final_timestep_features = mlp_output_flat.reshape(batch_size, Ta, -1)[
            :, -1, :
        ]  # (b, prev_dim)
        scalar_output = self.scalar_output(final_timestep_features)  # (b, 1)

        return main_output, scalar_output


class VanillaRNN(BaseNetwork):
    """Simplified RNN without fancy time embedding, LayerNorm, or ApproxGELU.

    Uses s and t directly by concatenating them with the input.
    """

    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 2,
        rnn_type: str = "LSTM",  # [LSTM, GRU]
        dropout: float = 0.1,
        mlp_layer_dims: list = None,  # Hidden dims for final MLP layers
    ):
        # BaseNetwork expects: act_dim, Ta, obs_dim, To, emb_dim, n_layers
        super().__init__(act_dim, Ta, obs_dim, To, rnn_hidden_dim, rnn_num_layers)
        self.act_dim = act_dim
        self.Ta = Ta
        self.obs_dim = obs_dim
        self.To = To
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.rnn_type = rnn_type
        self.dropout = dropout

        # Input size: action + observation + s + t (direct concatenation)
        input_dim_per_timestep = act_dim + obs_dim * To + 2  # +2 for s and t

        # RNN layer
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_dim_per_timestep,
                hidden_size=rnn_hidden_dim,
                num_layers=rnn_num_layers,
                dropout=dropout if rnn_num_layers > 1 else 0,
                batch_first=True,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_dim_per_timestep,
                hidden_size=rnn_hidden_dim,
                num_layers=rnn_num_layers,
                dropout=dropout if rnn_num_layers > 1 else 0,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        # Simple MLP layers after RNN (with ReLU instead of ApproxGELU)
        if mlp_layer_dims is None:
            mlp_layer_dims = [rnn_hidden_dim, rnn_hidden_dim // 2]

        mlp_layers = []
        prev_dim = rnn_hidden_dim
        for dim in mlp_layer_dims:
            mlp_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = dim

        self.mlp = nn.Sequential(*mlp_layers)

        # Output layers: one for the main output and one for the scalar
        self.main_output = nn.Linear(prev_dim, act_dim)
        self.scalar_output = nn.Linear(prev_dim, 1)

        # Simple Xavier initialization
        self._init_weights()

    def _init_weights(self):
        """Apply Xavier initialization to linear layers."""
        # RNN initialization
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)

        # MLP initialization
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

        # Output layers
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
            condition: (b, To, obs_dim)

        Returns:
            output_data: (b, Ta, act_dim)
            scalar_output: (b, 1)
        """
        batch_size, Ta, act_dim = x.shape

        # Handle condition - use zeros if None
        if condition is not None:
            condition_flat = torch.flatten(condition, 1)  # (b, To * obs_dim)
        else:
            # Create zeros for missing condition to maintain consistent input size
            condition_flat = torch.zeros(
                batch_size, self.To * self.obs_dim, device=x.device, dtype=x.dtype
            )

        s_expanded = s.unsqueeze(-1)  # (b, 1)
        t_expanded = t.unsqueeze(-1)  # (b, 1)

        # Prepare input for RNN: for each timestep, concatenate action, condition, s, and t
        rnn_inputs = []
        for i in range(Ta):
            timestep_input = torch.cat(
                [x[:, i, :], condition_flat, s_expanded, t_expanded], dim=-1
            )  # (b, input_dim_per_timestep)
            rnn_inputs.append(timestep_input)

        # Stack to create sequence: (b, Ta, input_dim_per_timestep)
        rnn_input_seq = torch.stack(rnn_inputs, dim=1)

        # Pass through RNN
        rnn_output, _ = self.rnn(rnn_input_seq)  # (b, Ta, rnn_hidden_dim)

        # Pass through MLP for each timestep
        # Reshape to (b * Ta, rnn_hidden_dim) for efficient processing
        rnn_output_flat = rnn_output.reshape(-1, self.rnn_hidden_dim)
        mlp_output_flat = self.mlp(rnn_output_flat)  # (b * Ta, prev_dim)

        # Generate main output for each timestep
        main_output_flat = self.main_output(mlp_output_flat)  # (b * Ta, act_dim)
        main_output = main_output_flat.reshape(
            batch_size, Ta, act_dim
        )  # (b, Ta, act_dim)

        # Generate scalar output from the final timestep
        final_timestep_features = mlp_output_flat.reshape(batch_size, Ta, -1)[
            :, -1, :
        ]  # (b, prev_dim)
        scalar_output = self.scalar_output(final_timestep_features)  # (b, 1)

        return main_output, scalar_output
