from dataclasses import dataclass, fields, asdict, replace
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import os
import argparse
from hellaswag import load_examples as load_hellaswag, evaluate as evaluate_hellaswag
import contextlib

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
        assert config.n_head % config.n_kv_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head

        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False) # more queries than k or v due to gqa
        self.wk = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)
        self.wo.NANOGPT_SCALE_INIT = 1

        self.cache_k = None
        self.cache_v = None

    def setup_cache(self, max_B, max_T, dtype, device):
        shape = (max_B, max_T, self.n_kv_head, self.head_dim)
        self.cache_k = torch.zeros(shape, dtype=dtype, device=device)
        self.cache_v = torch.zeros(shape, dtype=dtype, device=device)

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, freqs_cis, start_pos=0):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        q = self.wq(x).view(B, T, self.n_head, self.head_dim) # transforms (4, 64, 768) -> (4, 64, 12, 64)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim) # k and v both do (4, 64, 256) -> (4, 64, 4, 64)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        if self.cache_k is not None:
            self.cache_k[:B, start_pos:start_pos+T] = k
            self.cache_v[:B, start_pos:start_pos+T] = v
            k = self.cache_k[:B, :start_pos+T]
            v = self.cache_v[:B, :start_pos+T]

        # transform to (4, 12, 64, 64) so heads are at the front
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # scaled dot product attention is able to handle gqa internally
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1), enable_gqa=True)
        # revert to (4, 64, 768)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)

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
    def forward(self, x, freqs_cis, start_pos=0):
        x = x + self.attn(self.attention_norm(x), freqs_cis, start_pos)
        x = x + self.mlp(self.ffn_norm(x))
        return x

# --------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 8192
    vocab_size: int = 128256
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: int = 8
    n_embd: int = 4096
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

    def forward(self, idx, targets=None, start_pos=0):
        B, T = idx.size()
        assert start_pos + T <= self.config.block_size, f"Cannot forward sequence of length {T}"

        x = self.transformer.wte(idx)
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]

        for block in self.transformer.h:
            x = block(x, freqs_cis, start_pos)

        # forward the final layernorm and classifier
        x = self.transformer.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # flattening B and T to BxT to get two dims (BxT, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # caches are allocated per generation based on sequence length and B
    def setup_caches(self, max_B, max_T, dtype, device):
        for block in self.transformer.h:
            block.attn.setup_cache(max_B, max_T, dtype, device)

    def reset_caches(self):
        for block in self.transformer.h:
            block.attn.reset_cache()

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

    @staticmethod
    def _sample_next(logits, temperature, top_k, generator):
        if temperature == 0.0:
            return logits.argmax(dim=-1, keepdim=True) # greedy, used for parity testing
        logits = logits / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            vals, idxs = torch.topk(logits, k, dim=-1)
            probs = F.softmax(vals, dim=-1)
            return torch.gather(idxs, -1, torch.multinomial(probs, 1, generator=generator))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1, generator=generator)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50,
                 eos_token=None, generator=None, use_cache=True, autocast_dtype=None):
        was_training = self.training
        self.eval()

        B, T = idx.size()
        max_T = T + max_new_tokens
        assert max_T <= self.config.block_size, f"prompt ({T}) + new tokens ({max_new_tokens}) exceeds block_size"

        # cache must hold same dtype as forward pass
        cache_dtype = autocast_dtype or next(self.parameters()).dtype
        if use_cache:
            self.setup_caches(B, max_T, cache_dtype, idx.device)
        ctx = (torch.autocast(device_type=idx.device.type, dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext())

        finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
        cur, start = idx, 0

        with ctx:
            for _ in range(max_new_tokens):
                logits, _ = self(cur, start_pos=start)
                start += cur.size(1)
                next_tok = self._sample_next(logits[:, -1, :], temperature, top_k, generator)

                if eos_token is not None:
                    # rows that have already stopped will keep emitting eos to keep rectangular shape
                    next_tok = torch.where(finished.unsqueeze(1), torch.full_like(next_tok, eos_token), next_tok)
                    finished |= next_tok.squeeze(1) == eos_token

                idx = torch.cat((idx, next_tok), dim=1)
                if finished.all():
                    break

                # with cache we only have to feed new token, else we re-feed everything
                cur = next_tok if use_cache else idx
                if not use_cache:
                    start = 0

        self.reset_caches()
        if was_training:
            self.train()
        return idx

# -------------------------------------------------------
import tiktoken
enc = tiktoken.get_encoding('gpt2') # used by the sampling block in the training loop
import numpy as np

def load_tokens(filename):
    # now stays in uint16
    return np.load(filename)

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root, verbose=False):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get shard names
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if verbose:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, initalised to shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank # strides out the different processes

    def state_dict(self):
        return {"current_shard": self.current_shard,
                "current_position": self.current_position,
                "B": self.B, "T": self.T, "num_processes": self.num_processes}

    def load_state_dict(self, sd):
        # rank 0 gpu writes the checkpoints, so what is stored is in rank 0's position
        # must re-add the stride for other ranks
        if (sd["B"], sd["T"], sd["num_processes"]) != (self.B, self.T, self.num_processes):
            print(f"WARNING: dataloader geometry has been changed"
                  f"(ckpt {sd['B']}/{sd['T']}/{sd['num_processes']},"
                  f"now {self.B}/{self.T}/{self.num_processes})")
            self.reset()
            return
        self.current_shard = sd["current_shard"]
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = sd["current_position"] + self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        # only widen B*T+1 tokens to int64, this only widens the tokens that need it rather than the whole shard
        # each rank only uses its portion of each shard so rather than making each rank widen the whole shard it only widens what is necessary
        buf = torch.from_numpy(self.tokens[self.current_position : self.current_position+B*T+1].astype(np.int64))
        x = (buf[:-1]).view(B, T) # input data
        y = (buf[1:]).view(B, T) # label / target data
        self.current_position += B * T * self.num_processes
        # advance to the next shard if loading next batch would be out of bounds
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y

@dataclass
class TrainConfig:
    run_name: str = "run"
    out_dir: str = "log"
    data_root: str = "edu_fineweb10B"
    # shape
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50304
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    # batching
    total_batch_size: int = 524288
    B: int = 16
    T: int = 1024
    # optimisation
    max_lr: float = 1.8e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 19073
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # cadence
    eval_every: int = 250
    eval_steps: int = 20
    sample_every: int = 250
    checkpoint_every: int = 1000
    # misc
    seed: int = 1337
    compile: bool = True
    resume: str = ""
    # hellaswag
    hellaswag_every: int = 1000
    hellaswag_limit: int = 0 # 0 means all examples

PRESETS = {
    "llama-124m": {}, # default TrainConfig values
    "smoke": dict(run_name="smoke", B=8, total_batch_size=8*1024*4, warmup_steps=10,
                  max_steps=200, eval_every=50, sample_every=0, checkpoint_every=50,
                  hellaswag_every=0),
    "cpu": dict(run_name="cpu", n_layer=6, n_head=6, n_kv_head=2, n_embd=384,
                block_size=256, multiple_of=256, total_batch_size=4096, B=8, T=256,
                warmup_steps=20, max_steps=8000, eval_every=50, sample_every=50,
                checkpoint_every=100, compile=False, hellaswag_every=0),
}

def _opt_float(s):
    return None if s.lower() in ("none", "null", "") else float(s)

def build_config_from_cli(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="llama-124m")
    for f in fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        elif f.type in (int, float, str):
            parser.add_argument(flag, type=f.type, default=None)
        else:
            parser.add_argument(flag, type=_opt_float, default=None)
    args = parser.parse_args(argv)

    cfg = replace(TrainConfig(), **PRESETS[args.preset])
    # overrides any preset fields which have been changed in CLI
    # knows which are changed as flags are all defaulted to None
    overrides = {f.name: getattr(args, f.name) for f in fields(TrainConfig) if getattr(args, f.name) is not None}
    return replace(cfg, **overrides), overrides

def checkpoint_paths(cfg):
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    return run_dir, os.path.join(run_dir, "ckpt_last.pt"), os.path.join(run_dir, "log.txt")

def save_checkpoint(path, *, orig_model, optimizer, step, cfg, model_cfg, train_loader, world_size, val_loss, hellaswag_acc):
    payload = {
        "step": step,
        "model": orig_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_config": asdict(cfg),
        "model_config": asdict(model_cfg),
        "train_loader": train_loader.state_dict(),
        "world_size": world_size,
        "val_loss": val_loss,
        "hellaswag_acc": hellaswag_acc,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)

def resolve_resume(cfg):
    if not cfg.resume:
        return None
    if cfg.resume == "auto":
        _, last, _ = checkpoint_paths(cfg)
        return last if os.path.exists(last) else None
    if not os.path.exists(cfg.resume):
        raise FileNotFoundError(f"--resume {cfg.resume} does not exist")
    return cfg.resume

def log_line(path, s):
    with open(path, "a") as f:
        f.write(s + "\n")

# --------------------------------------------------------
if __name__ == '__main__':
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

    cfg, overrides = build_config_from_cli()

    resume_path = resolve_resume(cfg)
    ckpt = torch.load(resume_path, map_location="cpu") if resume_path else None
    run_dir, ckpt_path, log_path = checkpoint_paths(cfg)
    if master_process:
        os.makedirs(run_dir, exist_ok=True)

    hella_df = None
    if cfg.hellaswag_every:
        if master_process:
            load_hellaswag() # only rank 0 downloads the file
        if ddp:
            dist.barrier() # all other ranks wait until rank 0 finishes downloading
        hella_df = load_hellaswag() # all other ranks then read the file from cache

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    total_batch_size = cfg.total_batch_size
    B = cfg.B
    T = cfg.T
    assert total_batch_size % (B * T * ddp_world_size) == 0, "batch size divisible by B*T"

    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"Total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", data_root=cfg.data_root, verbose=master_process)
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", data_root=cfg.data_root, verbose=master_process)

    torch.set_float32_matmul_precision('high')

    # create model
    # artificially increase the number of tokens to go from ugly 50257 to nice 50304, cuda has kernels that work in chunks of nice numbers so special case handling needed
    # this leads to larger but nice computation which in the long run is faster, harmless as adds tokens which aren't found by tokeniser which only has 50257 tokens
    # these extra tokens will never be used and their probability will drop to zero
    gpt_field_names = {f.name for f in fields(GPTConfig)}
    if ckpt is not None:
        model_cfg = GPTConfig(**ckpt["model_config"])
        cli_cfg = GPTConfig(**{k: v for k, v in asdict(cfg).items() if k in gpt_field_names})
        if master_process and model_cfg != cli_cfg:
            print(f"WARNING: model shape from checkpoint overrides CLI shape\n"
                  f"    ckpt {model_cfg}\n"
                  f"    cli: {cli_cfg}")
    else:
        model_cfg = GPTConfig(**{k: v for k, v in asdict(cfg).items() if k in gpt_field_names})
    model = GPT(model_cfg)
    model.to(device)
    orig_model = model
    if ckpt is not None:
        orig_model.load_state_dict(ckpt["model"])

    if master_process:
        print(f"config: {cfg}")
        if overrides:
            print(f"CLI overrides: {overrides}")
        n_params = sum(p.numel() for p in model.parameters())
        n_emb = model_cfg.vocab_size * model_cfg.n_embd * 2   # wte + untied lm_head
        print(f"model: {n_params/1e6:.1f}M params ({(n_params-n_emb)/1e6:.1f}M non-embedding), "
              f"ffn={model.transformer.h[0].mlp.w1.out_features}, "
              f"grad_accum={grad_accum_steps}")

    if cfg.compile and device_type == 'cuda':
        model = torch.compile(model) # does what it says on the tin, compiles the program so pytorch doesnt have to run in "eager" mode
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], broadcast_buffers=False)

    # learning rate scheduler
    max_lr = cfg.max_lr
    min_lr = max_lr * cfg.min_lr_ratio
    warmup_steps = cfg.warmup_steps
    max_steps = cfg.max_steps
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
    optimizer = orig_model.configure_optimizers(weight_decay=cfg.weight_decay, learning_rate=cfg.max_lr, device=device)

    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

        train_loader.load_state_dict(ckpt["train_loader"])
        if ckpt["world_size"] != ddp_world_size and master_process:
            print(f"WARNING: resuming an {ckpt['world_size']}-process run on {ddp_world_size}")
        if ckpt["train_config"]["max_steps"] != cfg.max_steps and master_process:
            print(f"WARNING: mismatch in max steps between cfg ({cfg.max_steps}) and ckpt ({ckpt['train_config']['max_steps']})")
    start_step = ckpt["step"] + 1 if ckpt is not None else 0

    last_val_loss = None
    last_hella_acc = None

    for step in range(start_step, max_steps):
        t0 = time.time()

        # check validation loss
        if cfg.eval_every and (step % cfg.eval_every == 0 or step == max_steps - 1):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = cfg.eval_steps
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
                last_val_loss = val_loss_accum.item()
                msg = f"step {step}, validation loss: {last_val_loss:.4f}"
                print(msg)
                log_line(log_path, msg)

        # hellaswag eval
        if cfg.hellaswag_every and (step % cfg.hellaswag_every == 0 or step == max_steps - 1):
            orig_model.eval()
            nc, nt = evaluate_hellaswag(orig_model, hella_df, device, rank=ddp_rank, world_size=ddp_world_size, 
                                        limit=cfg.hellaswag_limit or None)
            nc = torch.tensor(nc, dtype=torch.long, device=device)
            nt = torch.tensor(nt, dtype=torch.long, device=device)
            if ddp:
                dist.all_reduce(nc, op=dist.ReduceOp.SUM)
                dist.all_reduce(nt, op=dist.ReduceOp.SUM)
            last_hella_acc = nc.item() / nt.item()
            if master_process:
                msg = f"hellaswag: {nc.item()}/{nt.item()} = {last_hella_acc*100:.2f}%"
                print(msg)
                log_line(log_path, msg)
                    
        # generate samples, apparently throws a scary error when used with torch.compile()
        if cfg.sample_every and ((step > 0 and step % cfg.sample_every == 0) or step == max_steps - 1): # and False: TODO uncomment when using torch.compile() and change 10 to 100
            tokens = torch.tensor(enc.encode("In 1945, "), dtype=torch.long)
            xgen = tokens.unsqueeze(0).repeat(4, 1).to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            out = orig_model.generate(xgen, max_new_tokens=32 - xgen.size(1), top_k=50, generator=sample_rng, 
                                      autocast_dtype=torch.bfloat16 if use_amp else None)

            for i in range(4):
                print(f"rank {ddp_rank} sample {i}: {enc.decode(out[i].tolist())}")

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
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        # get learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups: # sets the learning rate for all parameter groups within the optimiser
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        elif device_type == "mps":
            torch.mps.synchronize()
        else:
            torch.cpu.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            msg = (f"step {step}, loss: {loss_accum.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, " 
                   f"dt: {dt*1000:.2f}ms, tokens_sec: {tokens_per_sec:.2f}hz")
            print(msg)
            log_line(log_path, msg)
        if cfg.checkpoint_every and master_process and ((step + 1) % cfg.checkpoint_every == 0 or step == max_steps - 1):
            save_checkpoint(ckpt_path, orig_model=orig_model, optimizer=optimizer, step=step, cfg=cfg,
                            model_cfg=model_cfg, train_loader=train_loader, world_size=ddp_world_size, 
                            val_loss=last_val_loss, hellaswag_acc=last_hella_acc)
            print(f"saved checkpoint at step {step} -> {ckpt_path}")

    if ddp:
        destroy_process_group()
