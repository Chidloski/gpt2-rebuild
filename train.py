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

        # crops the mask to sequence length, if j > i (mask is 0) then a future token is influencing a past token
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
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

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

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}"

        # forward the position and token embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # shape (T, n_emb)
        tok_emb = self.transformer.wte(idx) # shape (B, T, n_emb)
        x = tok_emb + pos_emb

        # forward the transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

num_return_sequences = 5
max_length = 30

model = GPT(GPTConfig())
model.eval()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")

model.to(device)

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a dumb language model")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

# generate tokens
torch.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get logits
    # no grad states backprop won't be run so less data to cache
    with torch.no_grad():
        logits = model(x)
        # only care about logits at last position
        logits = logits[:, -1, :]
        # get probabilities
        probs = F.softmax(logits, dim=-1)
        # only sample top 50 most probable tokens
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select token
        ix = torch.multinomial(topk_probs, 1)
        # gather the corresponding indices and append to sequence
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

# print generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)