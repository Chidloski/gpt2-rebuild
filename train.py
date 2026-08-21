from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):
    bias: torch.Tensor

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Holds W_q, W_k, W_v, to allow for one big matmul
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # creates a matrix capabale of holding query, key, and value by dims * 3
        # Output projection matrix
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # register buffer to show it holds state not a parameter, no gradient and ignored by optimiser
        # tril gives lower-triangular matrix of 1s which is broadcasted across batch and head due to leading singleton dims
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size)) # historically called bias in paper, though it is a mask

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2) # splits qkv into chunks of size 384 (chunk for each q, k, v)

        # transpose makes batch and head the leading dimensions so matmul batches over them automatically
        # view splits the 384 channels into 6 contiguous blocks of 64, one for each attention head
        # gives shape (B, 6, T, 64), thus batch and attention n_head comes first
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # k.transpose(-2, -1) -> (B, 6, 64, T) thus we have matmul (B, 6, T, 64) @ (B, 6, 64, T)
        # only last two dims participate giving every query i dotted with key j, in head h
        # scaled as in the paper
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # crops the mask to sequence length, if j > i (gives 0) then a future token is influencing a past token
        # the score then becomes -inf as tokens only influenced by their predecessors
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        # (B, 6, T, T) @ (B, 6, T, 64) -> (B, 6, T, 64)
        # Convex combination of vectors weighted by attention
        y = att @ v

        # Undoes earlier permutation, back to (B, T, 6, 64) then flattens 6x64 into 384
        # Contiguous is called to allocate a new buffer with elements in row-major order so that view is compatible
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y) # Combines the outputs from all heads

        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # tanh approximation used within gpt2 paper, GELU is a smoother RELU
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    # projects up to 4* dims of n_embd, through gelu and then back down the n_embd dims
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) # layer normal 1, before attention
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd) # layer normal 2, before mlp
        self.mlp = MLP(config)

    # forward prop, adds both the attention and mlp back into the token
    # path is purely additive to allow for easier gradient flow
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # gives blocks for each of the layers within the transformer
            ln_f = nn.LayerNorm(config.n_embd), # final normalisation after final self-attention block as stated in gpt2-paper
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # linear map converts embedding space into vocab space