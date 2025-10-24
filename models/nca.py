import math

import torch
import torch.nn as nn

from models.xpos_relative_position import XPOS


class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma, head_size=None, double_v_dim=False):
        """
        Simplified retention mechanism inspired by 
        "Retentive Network: A Successor to Transformer for Large Language Models" 
        [https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(SimpleRetention, self).__init__()

        self.hidden_size = hidden_size
        self.head_size = head_size if head_size is not None else hidden_size
        self.v_dim = self.head_size * 2 if double_v_dim else self.head_size
        self.gamma = gamma

        self.W_Q = nn.Parameter(torch.randn(hidden_size, self.head_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, self.head_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)

        self.xpos = XPOS(self.head_size)

    def forward(self, X):
        """
        Parallel form of the retention mechanism.
        X: (batch_size, seq_length, hidden_size)
        """
        seq_length = X.shape[1]
        D = self._get_decay_matrix(seq_length).to(self.W_Q.device)

        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        Q = self.xpos(Q)
        K = self.xpos(K, downscale=True)

        retention = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)
        return retention @ V

    def forward_recurrent(self, x_n, s_prev, n):
        """
        Recurrent representation of the retention mechanism.
        x_n: (batch_size, 1, hidden_size)
        s_prev: (batch_size, hidden_size, v_dim)
        """
        Q = x_n @ self.W_Q
        K = x_n @ self.W_K
        V = x_n @ self.W_V

        Q = self.xpos(Q, n + 1)
        K = self.xpos(K, n + 1, downscale=True)

        s_next = self.gamma * s_prev + (K.transpose(-1, -2) @ V)
        return (Q @ s_next), s_next

    def forward_chunkwise(self, x_i, r_prev, i):
        """
        Chunk-based computation of the retention mechanism.
        x_i: (batch_size, chunk_size, hidden_size)
        r_prev: (batch_size, hidden_size, v_dim)
        """
        batch, chunk_size, _ = x_i.shape
        D = self._get_decay_matrix(chunk_size)

        Q = x_i @ self.W_Q
        K = x_i @ self.W_K
        V = x_i @ self.W_V

        Q = self.xpos(Q, i * chunk_size)
        K = self.xpos(K, i * chunk_size, downscale=True)

        r_i = (K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_prev
        inner_chunk = ((Q @ K.transpose(-1, -2)) * D.unsqueeze(0)) @ V

        decay_exp = torch.zeros(batch, chunk_size, 1)
        for idx in range(chunk_size):
            decay_exp[:, idx, :] = self.gamma ** (idx + 1)

        cross_chunk = (Q @ r_prev) * decay_exp
        return inner_chunk + cross_chunk, r_i

    def _get_decay_matrix(self, seq_length):
        n = torch.arange(seq_length).unsqueeze(1)
        m = torch.arange(seq_length).unsqueeze(0)
        D = (self.gamma ** (n - m)) * (n >= m).float()
        D[D != D] = 0  # Replace NaNs with zeros
        return D


class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, double_v_dim=False):
        """
        Multi-scale retention mechanism based on
        "Retentive Network: A Successor to Transformer for Large Language Models"
        """
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by number of heads"
        self.head_size = hidden_size // heads

        self.gammas = (1 - torch.exp(torch.linspace(math.log(1 / 32), math.log(1 / 512), heads))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, self.v_dim)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.hidden_size, gamma, self.head_size, double_v_dim)
            for gamma in self.gammas
        ])

    def forward(self, X):
        """
        Parallel execution of the multi-scale retention mechanism.
        """
        Y = [ret(X) for ret in self.retentions]
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)
        return (self.swish(X @ self.W_G) * Y) @ self.W_O

    def forward_recurrent(self, x_n, s_prev_list, n):
        """
        Recurrent execution of the multi-scale retention mechanism.
        x_n: (batch_size, 1, hidden_size)
        s_prev_list: list of (batch_size, hidden_size, v_dim)
        """
        Y, s_next_list = [], []
        for i in range(self.heads):
            y, s_next = self.retentions[i].forward_recurrent(x_n, s_prev_list[i], n)
            Y.append(y)
            s_next_list.append(s_next)

        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)
        return (self.swish(x_n @ self.W_G) * Y) @ self.W_O, s_next_list

    def forward_chunkwise(self, x_i, r_prev_list, i):
        """
        Chunk-based version of the multi-scale retention mechanism.
        x_i: (batch_size, chunk_size, hidden_size)
        r_prev_list: list of previous retention states
        """
        Y, r_next_list = [], []
        for j in range(self.heads):
            y, r_next = self.retentions[j].forward_chunkwise(x_i, r_prev_list[j], i)
            Y.append(y)
            r_next_list.append(r_next)

        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)
        return (self.swish(x_i @ self.W_G) * Y) @ self.W_O, r_next_list
