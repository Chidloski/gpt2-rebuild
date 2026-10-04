import torch
import time
import math
from train import enc, log_line, checkpoint_paths, GPT, GPTConfig
from sft_data import TOKEN_LOOKUP, END, USER, ASSISTANT, decode_with_special
import numpy as np
import os
from dataclasses import dataclass, fields, replace, asdict
import argparse
from hellaswag import load_examples as load_hellaswag, evaluate as evaluate_hellaswag

@torch.no_grad()
def init_special_tokens(model):
    trained = min(TOKEN_LOOKUP.keys())
    specials = max(TOKEN_LOOKUP.keys()) - trained + 1

    # set each of the special tokens in lm_head to be the average of the trained rows
    head = model.lm_head.weight
    head[trained:trained+specials] = head[:trained].mean(dim=0)

    # Sampled from the trained rows, each special's embedding is distinct and normal-sized
    wte = model.transformer.wte.weight
    wte[trained:trained+specials] = wte[:trained].mean(dim=0) + wte[:trained].std(dim=0) * torch.randn(specials, model.config.n_embd)

class SFTDataLoader:
    def __init__(self, split, B, process_rank, num_processes, seed, data_root="sft_data"):
        DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_root)
        self.tokens = np.load(os.path.join(DATA_DIR, f"{split}_tokens.npy"), mmap_mode="r")
        self.mask = np.load(os.path.join(DATA_DIR, f"{split}_mask.npy"), mmap_mode="r")
        self.offsets = np.load(os.path.join(DATA_DIR, f"{split}_offsets.npy"))
        self.lengths = np.diff(self.offsets)
        self.B = B
        self.process_rank = process_rank
        self.world_size = num_processes
        self.epoch = 0
        self.seed = seed

        self._build_epoch()

    def _row(self, i):
        s, e = self.offsets[i], self.offsets[i+1]
        toks = self.tokens[s:e].astype(np.int64)
        msk = self.mask[s:e]
        x = toks[:-1]
        y = toks[1:].copy()
        # F.cross_entropy has a default value of -100 to skip the position
        y[msk[1:] == 0] = -100

        return x, y

    def _build_epoch(self):
        num_conversations = len(self.lengths)
        rng = np.random.default_rng(self.seed + self.epoch)
        shuffled_indices = rng.permutation(num_conversations)
        shuffled_indices = shuffled_indices[num_conversations % self.B:]
        chunk_size = 100 * self.B
        for idx in range(0, len(shuffled_indices), chunk_size):
            chunk = shuffled_indices[idx : idx + chunk_size]
            chunk[:] = chunk[np.argsort(self.lengths[chunk])] # writes into the view chunk has on shuffled_indices

        batched_indices = shuffled_indices.reshape(-1, self.B)
        rng.shuffle(batched_indices)
        self.batches = batched_indices[len(batched_indices) % self.world_size:]
        self.batches = self.batches[self.process_rank::self.world_size]
        self.pos = 0

    def next_batch(self):
        if self.pos == len(self.batches):
            self.epoch += 1
            self._build_epoch()

        idxs = self.batches[self.pos]
        self.pos += 1
        T = self.lengths[idxs].max() - 1

        x = np.full((self.B, T), END, dtype=np.int64)
        y = np.full((self.B, T), -100, dtype=np.int64)

        for r, idx in enumerate(idxs):
            xr, yr = self._row(idx)
            x[r, :len(xr)] = xr
            y[r, :len(yr)] = yr

        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return len(self.batches)

    def reset(self):
        self.epoch = 0
        self._build_epoch()

@dataclass
class SFTConfig:
    init_from: str = "weights.pt"
    out_dir: str = "SFT_log"
    run_name: str = "SFT_run"
    B: int = 64
    epochs: int = 2
    max_lr: float = 1e-4
    min_lr_ratio: float = 0.1
    max_steps: int = 0
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_batches: int = 50
    seed: int = 1337
    compile: bool = True
    hellaswag: bool = True
    hellaswag_limit: int = 1000

def evaluate_val(model, loader, n_batches, device, use_amp):
    model.eval()
    loader.reset()
    loss_sum = 0.0
    tok_sum = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = loader.next_batch()
            x, y = x.to(device), y.to(device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)

            n = (y != -100).sum()
            loss_sum += loss * n
            tok_sum += n

    return (loss_sum / tok_sum).item()

PROMPTS = ["What is the capital of France?",
               "Write a haiku about planes",
               "Explain what a neural network is in simple terms",
               "Give me three tips for lifting weights"]

def sample_chats(model, device, use_amp, log_path):
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)
    replies = []
    for prompt in PROMPTS:
        tokenised = torch.tensor([USER] + enc.encode_ordinary(prompt) + [END, ASSISTANT], dtype=torch.long).unsqueeze(0).to(device)
        out = model.generate(tokenised, max_new_tokens=128, temperature=0.0, eos_token=END,
                             generator=sample_rng, autocast_dtype=torch.bfloat16 if use_amp else None)
        reply = out[0, tokenised.size(1):].tolist()
        stopped = reply[-1] == END
        reply = "".join(decode_with_special(reply)) + ("(no END: hit 128 tokens)" if not stopped else "")
        replies.append(reply)

    msg = ""
    for prompt, reply in zip(PROMPTS, replies):
        msg += f"Q. {prompt}\nA. {reply}\n"

    print(msg)
    log_line(log_path, msg)

def run_hellaswag(orig_model, df, device, limit, label, log_path):
    orig_model.eval()
    correct, total = evaluate_hellaswag(orig_model, df, device, limit=limit or None)
    msg = f"hellaswag {label}: {correct}/{total} = {correct/total:.2%}"
    print(msg)
    log_line(log_path, msg)
    return correct / total

def save_checkpoint(path, orig_model, step, val_loss, cfg):
    payload = {
        "step": step,
        "model": orig_model.state_dict(),
        "model_config": asdict(orig_model.config),
        "val_loss": val_loss,
        "sft_config": asdict(cfg),
    }

    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)

def build_config_from_cli(argv=None):
    parser = argparse.ArgumentParser()
    for f in fields(SFTConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        elif f.type in (int, float, str):
            parser.add_argument(flag, type=f.type, default=None)
    args = parser.parse_args(argv)

    # overrides any fields which have been changed in CLI
    # knows which are changed as flags are all defaulted to None
    overrides = {f.name: getattr(args, f.name) for f in fields(SFTConfig) if getattr(args, f.name) is not None}
    return replace(SFTConfig(), **overrides), overrides

if __name__ == "__main__":
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"using device: {device}")
    use_amp = device == "cuda"
 
    cfg, overrides = build_config_from_cli()
    run_dir, ckpt_path, log_path = checkpoint_paths(cfg)
    os.makedirs(run_dir, exist_ok=True)
    print(f"overrides: {overrides}")
    log_line(log_path, f"overrides: {overrides}")
    torch.manual_seed(cfg.seed)
    if device == "cuda":
        torch.cuda.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision('high')

    initial_weights = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.init_from), map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**initial_weights["model_config"]))
    model.load_state_dict(initial_weights["model"])
    del initial_weights
    init_special_tokens(model)
    model.to(device)
    orig_model = model
    if cfg.compile and device == "cuda":
        model = torch.compile(model)

    train_loader = SFTDataLoader("train", cfg.B, 0, 1, cfg.seed)
    val_loader = SFTDataLoader("val", cfg.B, 0, 1, cfg.seed)
    assert cfg.eval_batches <= len(val_loader), f"Eval batches {cfg.eval_batches} > {len(val_loader)} val batches"

    hella_df = load_hellaswag() if cfg.hellaswag else None

    # learning rate scheduler
    max_lr = cfg.max_lr
    min_lr = max_lr * cfg.min_lr_ratio
    warmup_steps = cfg.warmup_steps
    if cfg.max_steps == 0:
        max_steps = cfg.epochs * len(train_loader)
    else:
        max_steps = cfg.max_steps
    assert cfg.warmup_steps < max_steps
    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr

        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # lr follows the slope of a cosine graph from 0 to pi
        return min_lr + coeff * (max_lr - min_lr)

    optimizer = orig_model.configure_optimizers(weight_decay=cfg.weight_decay, learning_rate=cfg.max_lr, device=device)

    if cfg.hellaswag:
        pre_training_accuracy = run_hellaswag(orig_model, hella_df, device, cfg.hellaswag_limit, "pre-SFT", log_path)

    for step in range(max_steps):
        last_step = (step == max_steps - 1)

        if step % cfg.eval_every == 0 and not last_step:
            model.eval()
            val_loss = evaluate_val(model, val_loader, cfg.eval_batches, device, use_amp)
            msg = f"Step {step} val loss: {val_loss:.4f}"
            print(msg)
            log_line(log_path, msg)
            sample_chats(orig_model, device, use_amp, log_path)

        t0 = time.time()

        model.train()
        optimizer.zero_grad()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if use_amp:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)

        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

        dt = time.time() - t0
        tokens_per_sec = (y != -100).sum().item() / dt
        msg = f"step {step}, loss = {loss.item():.4f}, lr = {lr:.4e}, norm = {norm:.4f}, T: {x.shape[1]}, tokens/sec: {tokens_per_sec:.2f}"
        log_line(log_path, msg)
        if step % 50 == 0 or last_step:
            print(msg)

        end_of_epoch = (step + 1) % len(train_loader) == 0
        if end_of_epoch or last_step:
            val_loss = evaluate_val(model, val_loader, cfg.eval_batches, device, use_amp)
            name = f"ckpt_epoch{(step + 1) // len(train_loader)}.pt" if end_of_epoch else "ckpt_final.pt"
            save_checkpoint(os.path.join(run_dir, name), orig_model, step, val_loss, cfg)
            msg = f"saved {name}: step {step}, val {val_loss:.4f}"
            print(msg)
            log_line(log_path, msg)
            sample_chats(orig_model, device, use_amp, log_path)

    if cfg.hellaswag:
        post_training_accuracy = run_hellaswag(orig_model, hella_df, device, cfg.hellaswag_limit, "post-SFT", log_path)
        msg = f"Hellaswag goes from {pre_training_accuracy:.2%} pre training to {post_training_accuracy:.2%} post training"
        print(msg)
        log_line(log_path, msg)






