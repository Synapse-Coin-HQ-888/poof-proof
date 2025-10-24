import torch
import torch.nn as nn

from models.retention import MultiScaleRetention


class SynapseNet(nn.Module):
    """SynapseNet — layered retention-based architecture for multi-scale temporal encoding."""

    def __init__(self, num_layers, hidden_dim, ffn_dim, num_heads, expand_v_dim=False):
        super(SynapseNet, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.v_dim = hidden_dim * 2 if expand_v_dim else hidden_dim

        # Multi-scale retention layers (temporal memory modules)
        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, num_heads, expand_v_dim)
            for _ in range(num_layers)
        ])

        # Feed-forward transformation layers
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, hidden_dim)
            )
            for _ in range(num_layers)
        ])

        # Layer normalization (pre-norm for stability)
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Forward pass through all layers.

        Args:
            x: Tensor of shape (batch_size, seq_length, hidden_dim)
        Returns:
            Tensor of same shape representing processed output.
        """
        for i in range(self.num_layers):
            y = self.retentions[i](self.layer_norms_1[i](x)) + x
            x = self.ffns[i](self.layer_norms_2[i](y)) + y
        return x

    def forward_recurrent(self, x_t, prev_states, step_idx):
        """
        Recurrent forward pass (for streaming or online updates).

        Args:
            x_t: (batch_size, hidden_dim)
            prev_states: list of per-layer state tensors
            step_idx: current step index
        Returns:
            (x_t_next, new_states)
        """
        next_states = []
        for i in range(self.num_layers):
            out_t, new_state = self.retentions[i].forward_recurrent(
                self.layer_norms_1[i](x_t), prev_states[i], step_idx
            )
            y_t = out_t + x_t
            next_states.append(new_state)
            x_t = self.ffns[i](self.layer_norms_2[i](y_t)) + y_t
        return x_t, next_states

    def forward_chunkwise(self, chunk, prev_retentions, chunk_idx):
        """
        Chunkwise forward pass (for sequence segments).

        Args:
            chunk: Tensor (batch_size, seq_length, hidden_dim)
            prev_retentions: list of previous retention states
            chunk_idx: index of current chunk
        Returns:
            (updated_chunk, new_retentions)
        """
        new_retentions = []
        for i in range(self.num_layers):
            out_chunk, new_ret = self.retentions[i].forward_chunkwise(
                self.layer_norms_1[i](chunk), prev_retentions[i], chunk_idx
            )
            y_chunk = out_chunk + chunk
            new_retentions.append(new_ret)
            chunk = self.ffns[i](self.layer_norms_2[i](y_chunk)) + y_chunk
        return chunk, new_retentions
