# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch GPT-2 reimplementation that has been converted to the Llama 3
architecture, being prepared for a real pretraining run on rented GPUs. It is a
learning project: the owner writes the code and prefers to be walked through
changes rather than have them made for them. Default to explaining what to change
and why; only edit files directly when asked.

`llama3Changes.md` records the GPT-2 -> Llama 1 -> Llama 2 -> Llama 3 architecture
deltas and is the reference for *why* the model looks the way it does.

## Commands

```bash
python train.py --preset cpu                  # laptop-sized run (48.1M params, CPU/MPS)
python train.py --preset smoke                # single-GPU dry run, 200 steps
torchrun --standalone --nproc_per_node=8 train.py --preset llama-124m   # the real run

python preflight.py                           # 16 checks; run on a rented box BEFORE anything expensive
python fineweb.py                             # generate training shards (100 x 100M tokens, ~20GB)
python fineweb.py --target-shards 2 --shard-size 1000000   # tiny set for testing
python hellaswag.py                           # download eval set + self-test the render/score path
python shapes.py                              # print param counts for reference configs
python chat.py                                # REPL against weights.pt; --ckpt, --temperature, --top-k

python sft_data.py                            # tokenise smol-smoltalk -> sft_data/*.npy (~36s, ~860MB)
python sft_data.py --limit 100                # 100 rows per split, for testing
python sft.py                                 # SFT from weights.pt, single GPU (no DDP)
python sft.py --max-steps 1000 --max-lr 3e-4 --run-name sft-lr3e-4   # one sweep arm
python chat.py --ckpt SFT_log/SFT_run/ckpt_epoch2.pt                 # chat mode (auto-detected)
```

There is no test suite. `preflight.py` and the `__main__` blocks of `hellaswag.py`
and `shapes.py` are the closest thing; `preflight.py` exits non-zero on failure and
asserts the model still builds at **152.8M params / ffn=2048**.

Any `TrainConfig` field is overridable from the CLI, e.g.
`--max-lr 1.8e-3 --no-compile --hellaswag-limit 1000`.

## Architecture

**`train.py`** holds everything for training: model, data loading, config, and the
`__main__` training loop. **`hellaswag.py`** is a standalone eval module.
**`fineweb.py`** is a one-shot data generator. **`chat.py`** is a REPL wrapper
around `GPT.generate()` and holds no model code of its own.

Post-training lives in two separate files, deliberately not inside `train.py`
(`TrainConfig`'s presets and geometry guard assume fixed-shape pretraining):
**`sft_data.py`** is a one-shot generator like `fineweb.py`, and **`sft.py`** is the
SFT loop with its own `SFTConfig`. See [SFT](#sft-post-training).

### Config system

`TrainConfig` (a dataclass) is the single source of truth. `PRESETS` holds named
diffs against its defaults (`llama-124m`, `smoke`, `cpu`), and
`build_config_from_cli()` generates argparse flags **from the dataclass fields**,
using `default=None` as a sentinel so CLI values can layer over preset values.

Two rules follow from this:

- **Never add a `TrainConfig` field without wiring it.** Fourteen fields were once
  declared but never read, including `ffn_dim_multiplier`, which silently built a
  174M model instead of 152.8M. `preflight.py`'s param-count assertion exists to
  catch exactly this.
- **Do not add `from __future__ import annotations`.** `build_config_from_cli`
  branches on `f.type is bool`; string annotations would break every bool flag
  silently.

`GPTConfig` is built by field-name intersection with `TrainConfig`
(`{k: v for k, v in asdict(cfg).items() if k in gpt_field_names}`) so the two can
never drift. Every `GPTConfig` field name also exists in `TrainConfig`.

### The `orig_model` convention

The model gets wrapped twice: `DDP(torch.compile(GPT))`. A reference to the bare
`GPT` is captured **before** wrapping:

```python
model.to(device)
orig_model = model          # the only thing you checkpoint, sample, or eval with
```

All three names point at the same parameter tensors.

- **Training must go through `model`** (the DDP handle) or gradients never sync
  across ranks — and nothing errors, each GPU just trains a divergent model.
- **Everything else uses `orig_model`**: checkpointing (wrapped state dicts get
  `module._orig_mod.` key prefixes that will not load into a fresh `GPT`), HellaSwag
  (its input shapes vary per call, so the compiled path would recompile continuously),
  and sampling (`generate()` calls `setup_caches`/`reset_caches`, which the DDP handle
  does not forward — `torch.compile`'s wrapper does, but its bound method would run the
  uncompiled `forward` anyway).

### KV cache and sampling

`GPT.generate()` owns all sampling; the training loop and `chat.py` both call it.
`CausalSelfAttention` holds `cache_k`/`cache_v` of shape
`(max_B, max_T, n_kv_head, head_dim)` — `n_kv_head`, not `n_head`, so GQA shrinks the
cache 3x and `enable_gqa=True` expands inside SDPA. `setup_caches()` allocates
per-generation, `reset_caches()` sets them back to `None`.

- The cache stores **post-RoPE** keys. RoPE is a rotation by absolute position and the
  rotation for a given position never changes, so keys are rotated once at write time.
  Caching pre-RoPE keys would mean re-rotating the whole history every step.
- `start_pos` must be threaded `GPT.forward` -> `Block.forward` ->
  `CausalSelfAttention.forward`. It does two jobs that have to agree: slicing
  `freqs_cis[start_pos : start_pos + T]`, and indexing the cache. Dropping it on either
  hop is silent — every decode step writes to row 0 and reads back length 1, so the
  model attends only to the token it just emitted and generates fluent nonsense.
- **`is_causal=(T > 1)`, never `True`.** SDPA's causal mask is top-left aligned, so a
  single query against `P+1` cached keys would mask everything except position 0. One
  query at the end of the sequence needs no mask at all. This holds only for
  full-prefill-then-single-token-decode; chunked prefill would need an explicit mask.
- The cache dtype must equal the compute dtype or SDPA gets an fp32 `k` against a bf16
  `q`. `generate()` owns its own autocast context so a single `autocast_dtype` argument
  drives both.
- Training is unaffected because the entire cache path sits behind
  `if self.cache_k is not None`, and caches are never allocated on the training path.

`preflight.py`'s `kv cache parity` check compares **logits**, not sampled tokens — a
random-init model's argmax is degenerate enough to hide a broken cache. It runs
prefill+decode against one full forward and tolerates `1e-4` against an observed
`~4e-6` (exact equality is unavailable: SDPA picks different kernels for `T=8` vs
`T=1`). It also asserts caches are cleared and the plain forward is bit-identical
afterwards.

Measured on CPU fp32: **2.85x at 32 tokens, 3.50x at 128, 5.10x at 256**. Uncached
cost is quadratic in length and cached is linear, so the gap widens with longer
generations.

### Checkpointing

`save_checkpoint()` writes to `<out_dir>/<run_name>/ckpt_last.pt` via a `.tmp` file
plus `os.replace`, so an interrupted write can never corrupt the checkpoint being
resumed from. The payload is all plain dicts and tensors so `torch.load`'s default
`weights_only=True` can read it — **keep it that way** (store `asdict(cfg)`, never
the dataclass object).

`step` means *the step just completed*; resume starts at `step + 1`.

`model_config` is stored and is authoritative on resume — the model is rebuilt from
it, not from the CLI, which guarantees the shape matches the weights. This is also
what lets a downloaded checkpoint be reloaded anywhere via
`GPT(GPTConfig(**ck['model_config']))`.

`freqs_cis` is registered with `persistent=False`, so it is absent from the state
dict and recomputed by `__init__`. Nothing needs to store or load it.

### DataLoaderLite

Shards stay as **uint16 numpy arrays**; only the `B*T+1` tokens of each batch are
widened to int64 (`cross_entropy` requires Long targets). Widening in `load_tokens`
instead would cost 0.8GB per rank rather than 0.2GB.

`load_state_dict` re-applies the per-rank stride:

```python
self.current_position = sd["current_position"] + self.B * self.T * self.process_rank
```

Only rank 0 writes checkpoints, and rank 0's offset is zero, so the stored value is
the *shared* progress term `n*B*T*W`. Each rank adds its own constant back to rebuild
its slot in the tiling. This is only valid if `B`, `T` and world size are unchanged,
hence the geometry guard.

### HellaSwag

`render_example` turns one question into `tokens (4, T)` and `mask (4, T)`, one row
per candidate ending, mask=1 on ending tokens only. Endings **must** be tokenised as
`" " + ending` — GPT-2 BPE encodes the leading space into the token, and omitting it
degrades all four candidates toward random without any error.

`score_example` shifts logits/targets by one and shifts the **mask with the targets**
(`mask[:, 1:]`, matching `shift_tokens`, not `shift_logits`). It averages loss over
each row's real ending length; summing instead biases toward short endings.

`evaluate()` returns this rank's `(correct, total)` and knows nothing about DDP.
`train.py` combines with `dist.ReduceOp.SUM` — not `AVG`, because 10042 examples do
not divide evenly across ranks and averaging rates would misweight the shards.

An untrained model scores **~26%** (chance is 25%). Use that as the baseline; the
target is GPT-2 124M's ~29.6%.

## Batching

`total_batch_size` (524288 tokens/step) is a hyperparameter coupled to `max_lr` and
`max_steps` — changing it invalidates the LR schedule and changes how many optimizer
steps the run gets. `B` is purely a performance knob:
`grad_accum = total_batch_size / (B * T * world_size)`. Raising `B` changes nothing
but speed, and `B=64` on 8 GPUs is the ceiling (grad_accum reaches 1). Memory is
dominated by the `B*T*50304` logits tensor, not the weights.

## Run 1 baseline (2026-09-17)

The first real pretraining run is done. Any future change should be measured against
these numbers.

| | result |
|---|---|
| val loss | **3.0403** |
| HellaSwag (n=1000) | **34.1%** (peak 34.8%) |
| step | 19072 (full 10B-token pass) |
| reference: GPT-2 124M | 29.6% HellaSwag, ~3.29 val |
| reference: GPT-3 124M | 33.7% HellaSwag |

Config that produced it: `--preset llama-124m --max-lr 1.2e-3 --B 64`, on 2x H100 SXM,
3.24 hours, ~$23. Artifacts are on the RunPod Global Volume at `/workspace/run1/`
(`weights.pt`, `ckpt_last.pt`, `log.txt`); `weights.pt` and `run1_log.txt` are also on
Sam's machine.

Swept before the run, so don't re-derive these:

- **Batch size**: B=16/32/64 gave 794K/864K/886K tok/s. B=64 wins but it is only +2.5%
  over B=32 — past the saturation knee. B=128 needs ~88GB and OOMs an 80GB card.
- **Learning rate**: 6e-4 (the GPT-3 paper value the project started with) was clearly
  worst at 4.03 val; 1.2e-3 and 1.8e-3 tied at ~3.94; 3e-3 was worse *and* unstable
  (max grad norm 15.96 vs ~3.6). Took 1.2e-3 as the lower of the two tied arms.
- An untrained model scores **27.4%** on HellaSwag here, so that is the floor, not 25%.

Known-good throughput on 2x H100: **~880K tok/s**, 0.61 s/step including eval,
HellaSwag, sampling and checkpointing (which together cost only 0.6% of wall-clock).

## SFT (post-training)

Plain SFT (no DPO) on `HuggingFaceTB/smol-smoltalk`, which is filtered for ~135M-class
models. Expect "follows the shape of a conversation", not factual QA.

### Special tokens: no vocab resize

`vocab_size=50304` is tensor-core padding over GPT-2 BPE's 50257, so ids 50257-50303
already exist in both `wte` and `lm_head`. Four of them are the chat template, defined
once in `sft_data.py` and imported everywhere else:

```
USER=50257  ASSISTANT=50258  SYSTEM=50259  END=50260
[SYSTEM] sys [END] [USER] q [END] [ASSISTANT] a [END] [USER] ...
```

`END` is new rather than reusing `<|endoftext|>` (50256), which already means
"unrelated document follows". tiktoken knows none of these ids: they are spliced in as
raw ints, `enc.decode` raises on them (use `decode_with_special`), and text is encoded
with **`encode_ordinary`** because one train conversation contains a literal
`<|endoftext|>` that makes `enc.encode` raise.

These rows never received gradient in pretraining but were weight-decayed: `wte` rows
sat at norm ~0.16 (trained rows ~1.43) and `lm_head` rows produced logits ~-13.9 against
a vocab mean of ~-6.3. `init_special_tokens()` (once, from `weights.pt`, never on a
resumed SFT checkpoint) sets:

- `lm_head` rows to the **mean trained row**, so the END logit starts exactly at the
  vocab-average logit. All four identical is fine: only END is ever a target.
- `wte` rows to **mean + per-dim std x randn**. The plain mean has norm 0.123 (rows cancel)
  and would make all four specials identical inputs, so the model could not tell USER from
  ASSISTANT.

Verified: every ordinary logit is **bit-identical** before and after, and the other 43
padding slots are untouched. It must run after `load_state_dict` and after seeding.

### Data (`sft_data.py`)

`render(messages)` -> `(tokens, mask, truncated)` or `None`. mask=1 on assistant content
**and the assistant's END** (otherwise the model never learns to stop); 0 on role
headers, user and system text. Conversations over `MAX_LEN=1025` (block_size + 1 for the
shift) are **truncated at exchange boundaries**, never inside a turn, so every sample ends
on an assistant END. Dropping long ones instead would lose ~46%. A trailing unanswered
user turn (258 train convs) is dropped by the pairing.

Output: `sft_data/{train,val}_{tokens,mask,offsets}.npy` (uint16 / uint8 / int64,
offsets has N+1 entries; val is HF's `test` split). Measured:

| | train | val |
|---|---|---|
| conversations | 451,417 | 23,764 |
| tokens / trained | 270.7M / 80.1% | 14.2M / 80.2% |
| dropped / truncated | 1.9% / 39.4% | 1.9% / 39.6% |
| length p50 / p90 | 680 / 921 | 677 / 920 |

### `SFTDataLoader`

One conversation per row, padded to the batch max (`x` pad = END, `y` pad = -100).
`y[msk[1:] == 0] = -100`: the mask shifts **with the targets**, same rule as HellaSwag's
`mask[:, 1:]`. `F.cross_entropy`'s default `ignore_index=-100` does the masking, so
**`GPT.forward` is unchanged**. Two traps in `_row`: widen to int64 *before* writing
-100 (uint16 cannot hold it), and `y` must be a `.copy()` — `toks[:-1]` and `toks[1:]`
are overlapping views, so writing -100 into `y` corrupts `x`.

No packing: without document masking, packed conversations would attend to each other.
Padding waste is removed by **chunked length bucketing** in `_build_epoch`: shuffle,
sort within chunks of `100*B` by length, cut into batches, shuffle the batches. Real
tokens per padded batch: **59.1% -> 99.2%**. Do not sort globally (that gives 99.8% but
trains short-to-long every epoch with near-identical groupings). Batches are trimmed to
a multiple of world size before the `[rank::W]` slice so ranks never desync.

### Loss accounting

Batches carry very different numbers of graded tokens, so per-batch means are not
comparable. `evaluate_val` weights by token count (`loss * n` summed, divided by total
`n`). There is **no gradient accumulation** in `sft.py`; adding it would need
sum-reduction and normalisation by graded tokens across micro-steps, not `/ grad_accum`.

### Checkpoints and chat mode

`sft.py` saves `ckpt_epoch{N}.pt` at each epoch end (or `ckpt_final.pt` for a
`--max-steps` run), each with a fresh val loss and samples. Payload is `model`,
`model_config`, `step`, `val_loss`, `sft_config` — **no optimizer state** (~600MB, not
1.8GB) because there is no resume. `chat.py` detects an SFT checkpoint by the
`"sft_config"` key and switches to chat mode: history is a token list in the template,
a cut-off reply gets an END appended, the system prompt is kept separately and
`trim_history` drops whole exchanges from the front (the no-sliding-window gap below).
`max_new_tokens` is reserved out of the 1024 context, so large values shrink memory.

### Numbers so far

- Step-0 val loss (val, B=64-style eval) ~2.2-2.4; the untrained model never emits END
  and continues the prompt (`es.`, `.`) because the template means nothing to it yet.
- Pre-SFT HellaSwag (n=1000, fp32) is **34.4% on MPS vs run 1's 34.1% on H100**: TF32
  (`matmul_precision('high')`) vs full fp32 flips close calls. Compare pre vs post SFT on
  the **same machine** only; `sft.py` does both in one process.
- MPS cannot train this: OOM at B=4, ~50 s/step at B=2. Smoke-test on the GPU.

Plan, not yet run: 20-step check (also checks whether varying `T` recompiles under
`torch.compile`), LR sweep {3e-5, 1e-4, 3e-4} x 1000 steps, then the full run
(B=64, 2 epochs = 14,106 steps, warmup 100). Judge arms on val loss **and** post-SFT
HellaSwag; a drop of several points means forgetting.

## Known gaps

- **Document separation is not implemented.** Deliberately deferred so run 1 stays
  comparable to the nanoGPT baseline; it is intended as run 2's single variable.
  Doing it needs a block-diagonal mask (FlexAttention) *and* per-document RoPE
  position resets — masking alone is half the fix.
- **`generate()` has no sliding window.** It asserts
  `prompt + max_new_tokens <= block_size` rather than cropping, because cropping with a
  cache means sliding the cache and re-deriving positions. Chunked prefill is
  unsupported for the same reason.
- **Generation is not compiled.** With a cache every decode step is a fixed `(B, 1)`
  shape, so the recompilation argument no longer applies and a compiled decode step is
  now possible — untried.
- TorchInductor cannot codegen complex operators, so the complex-valued RoPE falls back
  to eager. Harmless (compile still gives ~55% overall), but a real-valued cos/sin RoPE
  would close it.
- **`sft.py` is single-GPU with no resume.** `SFTDataLoader` already takes rank/world
  size, but the loop has no DDP init, all-reduce or `require_backward_grad_sync`, and an
  interrupted run restarts from `weights.pt`. Both were skipped because a full SFT run is
  well under an hour on one H100.
- DDP *is* verified as of run 1: 1-GPU and 2-GPU runs produced bit-identical loss
  trajectories, which simultaneously proves the per-rank dataloader stride, the gradient
  all-reduce and the grad_accum normalisation. Note it cannot be tested locally — macOS
  gloo cannot complete rendezvous.
