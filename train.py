from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import os

# taken from meta's llama 3 repo
class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # params initialised to one means RMSNorm is initially a pure normalisation

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x) # computes in fp32 and casts back, bf16 only has 8 bit mantissa so squaring in it makes precision worse
        return output * self.weight

# precomputers the rotations for each position
def precompute_freqs_cis(dim, end, theta=10000.0):
    # for each even index (rotation happens in pairs), get base^(-2i/d)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    # outer product pairs every position with every frequency, allowing angle to be computed easily
    freqs = torch.outer(t, freqs)
    # torch.polar returns cos + isin from polar coordinates
    return torch.polar(torch.ones_like(freqs), freqs)

# freqs_cis is (T, 32), xq_ is (B, T, n_head, 32) thus we need to reshape
def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # for each of the 4 dimensions, keep the size (T) at position 1 and at the last position, everything else is collapsed to 1
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

# rope applies the positional embeddings only where necessary, just before attention in the query and key matrices
# it does this by rotating each vector, it chops the vector into 2d pairs and rotates each by a different frequency
# this means that positions are relative rather than absolute
def apply_rotary_emb(xq, xk, freqs_cis):
    # reshapes from (B, T, nh, 64) -> (B, t, nh, 32, 2) for complex numbers
    # viewing as complex thus gives (B, t, nh, 32)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # xq_ * freqs_cis give the rotation, view_as_real gets back to (B, T, nh, 32, 2), flatten gets back to (B, T, nh, 64)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Holds W_q, W_k, W_v, to allow for one big matmul
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False) # creates a matrix capabale of holding query, key, and value by dims * 3
        # Output projection matrix
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, freqs_cis):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2) # splits qkv into chunks of size 384 (chunk for each q, k, v)

        # transpose makes batch and head the leading dimensions so matmul batches over them automatically
        # view splits the 768 channels into 12 contiguous blocks of 64, one for each attention head
        # gives shape (B, 12, T, 64), thus batch and attention n_head comes first
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q,k = q.transpose(1, 2), k.transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

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
        hidden = 4 * config.n_embd
        hidden = int(2 * hidden / 3) # shrink width to keep param count the same as using gelu (without dim_multiplier)
        if config.ffn_dim_multiplier is not None:
            hidden = int(config.ffn_dim_multiplier * hidden)
        hidden = config.multiple_of * ((hidden + config.multiple_of - 1) // config.multiple_of) # round up to nearest multiple

        self.w1 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=False)
        self.w2.NANOGPT_SCALE_INIT = 1

    # projects up to 4* dims of n_embd, through swiglu and then back down the n_embd dims
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.n_embd, config.norm_eps) # normal before attention
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd, config.norm_eps) # normal before mlp
        self.mlp = MLP(config)

    # forward prop, adds both the attention and mlp back into the token
    # path is purely additive to allow for easier gradient flow
    def forward(self, x, freqs_cis):
        x = x + self.attn(self.attention_norm(x), freqs_cis)
        x = x + self.mlp(self.ffn_norm(x))
        return x

# --------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    norm_eps: float = 1e-5
    ffn_dim_multiplier: float | None = 1.3
    multiple_of: int = 1024
    rope_theta: float = 500000.0

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # gives blocks for each of the layers within the transformer
            norm = RMSNorm(config.n_embd, config.norm_eps), # final normalisation after final self-attention block as stated in gpt2-paper
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # linear map converts embedding space into vocab space

        self.register_buffer("freqs_cis", precompute_freqs_cis(config.n_embd // config.n_head, config.block_size, config.rope_theta), persistent=False,)

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # initialised in accordance with gpt2
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # scale in gpt2 paper to keep std at 1
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}"

        x = self.transformer.wte(idx)
        freqs_cis = self.freqs_cis[:T]

        for block in self.transformer.h:
            x = block(x, freqs_cis)

        # forward the final layernorm and classifier
        x = self.transformer.norm(x)
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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # all parameters which require grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # only weight decay parameters which should be, e.g biases and layernorms aren't weight decayed
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create the AdamW optimizer and used fused versions if it is available
        # fused is a speedup which instead of launching many gpu kernels to update parameter tensors, it does it in a singular kernel handling many tensors at once
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -------------------------------------------------------
import tiktoken
enc = tiktoken.get_encoding('gpt2') # used by the sampling block in the training loop
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get shard names
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, initalised to shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank # strides out the different processes

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # input data
        y = (buf[1:]).view(B, T) # label / target data
        self.current_position += B * T * self.num_processes
        # advance to the next shard if loading next batch would be out of bounds
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y

# --------------------------------------------------------

import time
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# setting up distributed data parallel (ddp)
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    # TODO needs CUDA
    assert torch.cuda.is_available(), "ddp needs cuda"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # master process (arbitrarily 0) will do logging etc
else:
    # non-ddp
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"using device: {device}")
    # device = 'cpu' # override

# device is "cuda:0" under ddp but "cuda" otherwise; device_type is the kind of device,
# which is what autocast wants. defined outside the if/else so both paths have it.
device_type = "cuda" if device.startswith("cuda") else device
use_amp = (device_type == "cuda") # used to gate autocast, autocast buys little time on cpu runs of small models

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Due to such a large batch size, we must run gradient accumulation to run the batch partly sequentially
total_batch_size = 524288 # batch size of 0.5M tokens in accordance with gpt3 paper
B = 16
T = 1024
""" CPU VALUES
total_batch_size = 4096
B = 16
T = 128"""
assert total_batch_size % (B * T * ddp_world_size) == 0, "batch size divisible by B*T"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# create model
# artificially increase the number of tokens to go from ugly 50257 to nice 50304, cuda has kernels that work in chunks of nice numbers so special case handling needed
# this leads to larger but nice computation which in the long run is faster, harmless as adds tokens which aren't found by tokeniser which only has 50257 tokens
# these extra tokens will never be used and their probability will drop to zero
model = GPT(GPTConfig(vocab_size=50304))

# model = GPT(GPTConfig(vocab_size=50304, n_layer=6, n_head=6, n_embd=384, block_size=128)) # model shrunk for cpu run

model.to(device)
if device_type == 'cuda':
    model = torch.compile(model) # does what it says on the tin, compiles the program so pytorch doesnt have to run in "eager" mode
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the unwrapped model

# learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # derived as 375e6 / 524288, linear warmup over first 375M tokens
max_steps = 19073 # derived as 10e9 / 524288, one pass over 10B token dataset
""" CPU VALUES
warmup_steps = 32
max_steps = 4096"""
# according to gpt3 paper we have:
# 1. Linear warmup over first 375 million tokens
# 2. Cosine decay to 10% of original lr value over 260 billion tokens
# 3. Training continues after this at 10% of original lr
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # lr follows the slope of a cosine graph from 0 to pi
    return min_lr + coeff * (max_lr - min_lr)


#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) # hyperparams according to gpt3
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()

    # check validation loss
    if step % 50 == 0 or step == max_steps - 1:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                if use_amp:
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                else:
                    logits, loss = model(x, y)

                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
                
    # generate samples, apparently throws a scary error when used with torch.compile()
    if (step > 0 and step % 50 == 0) or step == max_steps - 1: # and False: TODO uncomment when using torch.compile() and change 10 to 100
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("To be or not to be ")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # training loop
    model.train()
    optimizer.zero_grad() # set gradients to 0, gradients deposited via +=
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        if use_amp:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # cast to lower precision for faster runtime on ampere
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        # the loss in each step is averaged and thus if we simply added the loss of each micro-step we would be summing averages
        # to get the true average we divide each micro-step's loss by number of micro-steps to re-average the loss
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # clipping the norm according to gpt3's paper, the norm is the length of the vector containing the gradient of all parameters
    # clipping this preserves the direction but stops large magnitude updates from shocking the model, potentially due to bad data within a batch
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # get learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups: # sets the learning rate for all parameter groups within the optimiser
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # needs switching to cpu for cpu runs
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step}, loss: {loss_accum.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, dt: {dt*1000:.2f}ms, tokens_sec: {tokens_per_sec:.2f}hz")

if ddp:
    destroy_process_group()
