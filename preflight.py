"""
Pre-flight checks. Run this on a freshly rented box BEFORE anything expensive.

    python preflight.py

Every check here exists because getting it wrong costs money: a stale torch that
lacks enable_gqa, an image without bf16, a half-generated dataset, or a model that
silently builds at the wrong size. Ten seconds here beats finding out at step 40.

Exits 0 if everything critical passes, 1 otherwise.
"""

import os
import shutil
import sys
import traceback

# what the real run should produce - if these drift, the config plumbing broke
EXPECT_PARAMS_M = 152.8
EXPECT_FFN = 2048
MIN_FREE_GB = 30          # ~20GB data + checkpoints + headroom

results = []              # (severity, name, ok, detail)


def check(name, critical=True):
    """Decorator: run a check, capture its (ok, detail), never let it kill the script."""
    def wrap(fn):
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append(("CRITICAL" if critical else "WARN", name, ok, detail))
        return fn
    return wrap


# ---------------------------------------------------------------- environment

@check("python >= 3.10")
def _():
    v = sys.version_info
    # train.py uses `float | None` annotations, a syntax error before 3.10
    return v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}"


@check("torch imports")
def _():
    import torch
    return True, torch.__version__


@check("cuda available")
def _():
    import torch
    n = torch.cuda.device_count()
    return torch.cuda.is_available() and n > 0, f"{n} device(s)"


@check("gpus uniform", critical=False)
def _():
    import torch
    if not torch.cuda.is_available():
        return False, "no cuda"
    names = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        names.append(f"{p.name} {p.total_memory/1e9:.0f}GB")
    uniq = sorted(set(names))
    # mixed GPU types in one node give wildly uneven step times under DDP
    return len(uniq) == 1, " | ".join(uniq)


@check("nccl available")
def _():
    import torch.distributed as dist
    ok = dist.is_nccl_available()
    return ok, "available" if ok else "NOT available - required for multi-gpu DDP"


@check("dependencies importable")
def _():
    missing = []
    for m in ("numpy", "pandas", "pyarrow", "tiktoken", "datasets", "tqdm"):
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    return not missing, f"missing: {missing}" if missing else "all present"


@check("tmux installed", critical=False)
def _():
    p = shutil.which("tmux")
    return p is not None, p or "not found - a dropped ssh session would kill your run"


# ---------------------------------------------------------------- torch features

@check("sdpa enable_gqa")
def _():
    import torch
    import torch.nn.functional as F
    # CausalSelfAttention.forward depends on this; older torch raises TypeError
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    F.scaled_dot_product_attention(q, k, k, is_causal=True, enable_gqa=True)
    return True, "supported"


@check("bf16 autocast on cuda")
def _():
    import torch
    if not torch.cuda.is_available():
        return False, "no cuda"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")
    return y.dtype == torch.bfloat16, f"matmul -> {y.dtype}"


@check("torch.compile works", critical=False)
def _():
    import torch
    f = torch.compile(lambda x: x * 2 + 1)
    return f(torch.randn(8)).shape == (8,), "compiled a trivial fn"


# ---------------------------------------------------------------- disk & data

@check("disk space")
def _():
    free = shutil.disk_usage(".").free / 1e9
    return free >= MIN_FREE_GB, f"{free:.0f} GB free (need >= {MIN_FREE_GB})"


@check("training shards")
def _():
    import numpy as np
    from train import build_config_from_cli
    cfg, _ = build_config_from_cli(["--preset", "llama-124m"])
    root = cfg.data_root
    if not os.path.isdir(root):
        return False, f"{root}/ missing - run `python fineweb.py`"
    train = sorted(f for f in os.listdir(root) if "train" in f and f.endswith(".npy"))
    val = sorted(f for f in os.listdir(root) if "val" in f and f.endswith(".npy"))
    if not train or not val:
        return False, f"train={len(train)} val={len(val)} - run `python fineweb.py`"
    a = np.load(os.path.join(root, train[0]), mmap_mode="r")
    total = len(a) * len(train)
    ok = a.dtype == np.uint16 and len(train) >= 90
    return ok, f"{len(train)} train + {len(val)} val shards, {a.dtype}, ~{total/1e9:.2f}B train tokens"


@check("hellaswag data")
def _():
    from hellaswag import load_examples
    df = load_examples()
    return len(df) == 10042, f"{len(df)} examples"


# ---------------------------------------------------------------- the model

@check("model builds at expected shape")
def _():
    from dataclasses import asdict, fields
    import torch
    from train import GPT, GPTConfig, build_config_from_cli
    cfg, _ = build_config_from_cli(["--preset", "llama-124m"])
    names = {f.name for f in fields(GPTConfig)}
    mc = GPTConfig(**{k: v for k, v in asdict(cfg).items() if k in names})
    with torch.device("meta"):
        m = GPT(mc)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    ffn = m.transformer.h[0].mlp.w1.out_features
    ok = abs(n - EXPECT_PARAMS_M) < 0.5 and ffn == EXPECT_FFN
    return ok, f"{n:.1f}M params (expect {EXPECT_PARAMS_M}), ffn={ffn} (expect {EXPECT_FFN})"


@check("kv cache parity")
def _():
    # the cached decode path must reproduce the full-recompute forward exactly.
    # weight-free on purpose: this runs on a fresh box before weights.pt exists.
    # compares logits rather than sampled tokens - a random-init model's argmax is
    # degenerate enough to hide a broken cache.
    from dataclasses import asdict, fields
    import torch
    from train import GPT, GPTConfig, build_config_from_cli
    torch.manual_seed(1337)
    cfg, _ = build_config_from_cli(["--preset", "llama-124m"])
    names = {f.name for f in fields(GPTConfig)}
    mc = GPTConfig(**{k: v for k, v in asdict(cfg).items() if k in names})
    m = GPT(mc).eval()

    B, P, T = 2, 8, 16                      # prefill P tokens, then decode to T
    x = torch.randint(0, mc.vocab_size, (B, T))
    with torch.no_grad():
        ref, _ = m(x)                       # ground truth: one full forward

    def cached_logits(n):                   # prefill + one-token decode, batch size n
        m.setup_caches(n, T, torch.float32, "cpu")
        with torch.no_grad():
            outs = [m(x[:n, :P], start_pos=0)[0]]
            for i in range(P, T):
                outs.append(m(x[:n, i:i+1], start_pos=i)[0])
        m.reset_caches()
        return torch.cat(outs, dim=1)

    d_cache = (ref - cached_logits(B)).abs().max().item()
    d_batch = (ref[:1] - cached_logits(1)).abs().max().item()   # B=1 must match row 0 of B=2

    # generate() must leave no cache behind, or the next training step reads stale keys
    cleared = all(b.attn.cache_k is None and b.attn.cache_v is None for b in m.transformer.h)
    with torch.no_grad():
        after, _ = m(x)
    untouched = torch.equal(ref, after)

    ok = d_cache < 1e-4 and d_batch < 1e-4 and cleared and untouched
    return ok, (f"logits d={d_cache:.1e}, batch d={d_batch:.1e}, "
                f"cleared={cleared}, train path untouched={untouched}")

@check("forward + backward on gpu")
def _():
    from dataclasses import asdict, fields
    import torch
    from train import GPT, GPTConfig, build_config_from_cli
    if not torch.cuda.is_available():
        return False, "no cuda"
    cfg, _ = build_config_from_cli(["--preset", "llama-124m"])
    names = {f.name for f in fields(GPTConfig)}
    mc = GPTConfig(**{k: v for k, v in asdict(cfg).items() if k in names})
    m = GPT(mc).cuda()
    x = torch.randint(0, mc.vocab_size, (2, 256), device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = m(x, x)
    loss.backward()
    g = m.transformer.wte.weight.grad
    mem = torch.cuda.max_memory_allocated() / 1e9
    del m
    torch.cuda.empty_cache()
    return g is not None, f"loss={loss.item():.3f}, peak {mem:.2f} GB"


# ---------------------------------------------------------------- report

def main():
    width = max(len(n) for _, n, _, _ in results)
    print()
    failed = 0
    for sev, name, ok, detail in results:
        if ok:
            mark = "\033[32mPASS\033[0m"
        elif sev == "WARN":
            mark = "\033[33mWARN\033[0m"
        else:
            mark = "\033[31mFAIL\033[0m"
            failed += 1
        print(f"[{mark}] {name:<{width}}  {detail}")

    print()
    if failed:
        print(f"\033[31m{failed} critical check(s) failed - fix before spending GPU time.\033[0m")
        return 1
    warns = sum(1 for s, _, ok, _ in results if not ok and s == "WARN")
    print("\033[32mAll critical checks passed.\033[0m" + (f" ({warns} warning(s))" if warns else ""))
    print("\nNext: python train.py --preset smoke")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
