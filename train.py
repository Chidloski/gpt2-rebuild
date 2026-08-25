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
        self.c_proj.NANOGPT_SCALE_INIT = 1

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

        '''# k.transpose(-2, -1) -> (B, 6, 64, T) thus we have matmul (B, 6, T, 64) @ (B, 6, 64, T)
        # only last two dims participate giving every query i dotted with key j, in head h
        # scaled as in the paper
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # crops the mask to sequence length, if j > i (mask is 0) then a future token is influencing a past token
        # the score then becomes -inf as tokens only influenced by their predecessors
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        # (B, 6, T, T) @ (B, 6, T, 64) -> (B, 6, T, 64)
        # Convex combination of vectors weighted by attention
        y = att @ v'''

        # replaces the above commented code, allows torch.compile to realise flashattention should be called
        # TODO read flash-attention paper
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
        self.c_proj.NANOGPT_SCALE_INIT = 1

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

# --------------------------------------------------------------

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

        # linear map and wte matrix are the same matrix in gpt2
        # -> Done as both have similar goals, wte takes tokens to embeddings, lm wants to find a token closest to the embedding it wants to say
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # initialised in accordance with gpt2
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # scale in gpt2 paper to keep std at 1
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
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
        loss = None
        if targets is not None:
            # flattening B and T to BxT to get two dims (BxT, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

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

# -------------------------------------------------------
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # input data
        y = (buf[1:]).view(B, T) # label / target data
        self.current_position += B*T
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# --------------------------------------------------------

import time

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")
device = "cpu" # override

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=4, T=256)

# TODO - enable this for cuda
#torch.set_float32_matmul_precision('high')

# get logits
# artificially increase the number of tokens to go from ugly 50257 to nice 50304, cuda has kernels that work in chunks of nice numbers so special case handling needed
# this leads to larger but nice computation which in the long run is faster, harmless as adds tokens which aren't found by tokeniser which only has 50257 tokens
# these extra tokens will never be used and their probability will drop to zero
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model) # does what it says on the tin, compiles the program so pytorch doesnt have to run in "eager" mode

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad() # set gradients to 0, gradients deposited via +=
    # TODO enable for cuda - with torch.autocast(device_type=device, dtype=torch.bfloat16): # cast to lower precision for faster runtime on ampere
    logits, loss = model(x, y) # TODO tab in so nested in autocast for cuda
    loss.backward()
    optimizer.step()
    torch.cpu.synchronize() # TODO needs to be changed to .cuda before training on nvidia
    t1 = time.time()
    dt = (t1 - t0)*1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1-t0)
    print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tokens_sec: {tokens_per_sec:.2f}hz")

print(loss)

import sys; sys.exit(0)

model.eval()
num_return_sequences = 5
max_length = 30

tokens = enc.encode("Hello, I'm a dumb language model")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

# get logits, get tokens
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