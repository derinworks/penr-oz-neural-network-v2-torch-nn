import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as nnf


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, block_size: int, bias=True, dropout: float=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        # Query, Key and Value in one matrix
        self.query_key_value = nn.Linear(embedding_dim, embedding_dim * 3, bias)
        # Output projection
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias)
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.projection_dropout = nn.Dropout(dropout)
        # Causal mask for self attention decoding
        self.register_buffer("causal_mask", torch.tril(torch.ones(1, 1, block_size, block_size)))

    def forward(self, input_data: Tensor) -> Tensor:
        batch_size, block_size, embedding_dim = input_data.size()
        # extract query, key and value splits
        qkv_splits = self.query_key_value(input_data).split(self.embedding_dim, dim=2)
        # prep query, key and value for matrix multiply shaped batch size, block size, head size, # of heads
        head_size = embedding_dim // self.num_heads
        q, k, v = (s.view(batch_size, block_size, self.num_heads, head_size).transpose(1, 2) for s in qkv_splits)
        # apply attention formula
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.num_heads)
        causal_mask = self.causal_mask[:, :, :block_size, :block_size]
        weights = nnf.softmax(scores.masked_fill(causal_mask == 0, float('-inf')), dim=-1)
        weights = self.dropout(weights)
        # (batch size, # of heads, block size, block size) x (batch size, # of heads, block size, head size) ->
        # (batch size, # of heads, block size, head size)
        output = weights @ v
        # combine head outputs -> (batch size, block size, embedding dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, block_size, embedding_dim)
        # project output
        output_projected = self.projection(output)
        output_projected = self.projection_dropout(output_projected)
        return output_projected


class PositionEmbedding(nn.Embedding):
    def forward(self, input_data: Tensor) -> Tensor:
        _, num_positions = input_data.shape
        positions = torch.arange(num_positions, dtype=torch.long, device=input_data.device)
        forwarded = super().forward(positions)
        return forwarded


class Summation(nn.Sequential):
    def forward(self, input_data: Tensor) -> Tensor:
        forwarded = self[0].forward(input_data)
        for layer in self[1:]:
            # please note: torch autograd fails with += in-place op, so use a = a + b instead
            forwarded = forwarded + layer(input_data)
        return forwarded


class ResidualConnection(nn.Sequential):
    def forward(self, forwarded: Tensor) -> Tensor:
        for layer in self:
            # please note: torch autograd fails with += in-place op, so use a = a + b instead
            forwarded = forwarded + layer(forwarded)
        return forwarded


class SoftmaxOnLast(nn.Softmax):
    def forward(self, logits: Tensor) -> Tensor:
        probs = super().forward(logits[:,-1,:])
        return probs
