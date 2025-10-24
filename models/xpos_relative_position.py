# Copyright (c) 2022 Microsoft  
# Licensed under The MIT License (https://github.com/microsoft/torchscale/blob/main/LICENSE)

import torch
import torch.nn as nn

def generate_fixed_positional_encoding(x):
    """Generate sinusoidal positional encodings with fixed frequency scaling."""
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_input = torch.einsum("i, j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    return torch.sin(sinusoid_input), torch.cos(sinusoid_input)

def rotate_pairwise(x):
    """Rotate each pair of dimensions for rotary embeddings."""
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    rotated = torch.stack((-x2, x1), dim=-1)
    if rotated.shape[-1] % 2 == 1:
        x2 = torch.cat((x2, torch.zeros_like(x2[:, :, :1])), dim=-1)
    return rotated.flatten(-2)

def interleave_duplicate(m):
    """Interleave duplicated matrix elements (like a simplified repeat_interleave)."""
    dim0 = m.shape[0]
    m = m.view(-1, 1).repeat(1, 2).view(dim0, -1)
    return m

def apply_rotary_embedding(x, sin, cos, scale=1):
    """Apply rotary positional encoding to a tensor."""
    sin, cos = map(lambda t: interleave_duplicate(t * scale), (sin, cos))
    return (x * cos[:, :x.shape[-1]]) + (rotate_pairwise(x) * sin)[:, :, :x.shape[-1]]

class SynapseXPOS(nn.Module):
    """Synapse adaptive positional modulation — generalized rotary scaling for attention."""
    def __init__(self, head_dim, scale_base=512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, offset=0, downscale=False):
        seq_len = x.shape[1]
        min_pos = 0
        max_pos = seq_len + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = generate_fixed_positional_encoding(scale)

        if scale.shape[0] > seq_len:
            scale, sin, cos = scale[-seq_len:], sin[-seq_len:], cos[-seq_len:]
        if downscale:
            scale = 1 / scale

        return apply_rotary_embedding(x, sin, cos, scale)

    def forward_inverse(self, x, offset=0, downscale=False):
        seq_len = x.shape[1]
        min_pos = -(seq_len + offset) // 2
        max_pos = seq_len + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = generate_fixed_positional_encoding(scale)

        if scale.shape[0] > seq_len:
            scale, sin, cos = scale[-seq_len:], sin[-seq_len:], cos[-seq_len:]
        if downscale:
            scale = 1 / scale

        return apply_rotary_embedding(x, -sin, cos, scale)

# Quick test
if __name__ == "__main__":
    x = torch.eye(4).unsqueeze(0)
    xpos = SynapseXPOS(4)
    x_rot = xpos(x)
    x_rot_rev = xpos.forward_inverse(x)
    print(x_rot @ x_rot_rev.transpose(-1, -2))
