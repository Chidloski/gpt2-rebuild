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

python preflight.py                           # 15 checks; run on a rented box BEFORE anything expensive
python fineweb.py                             # generate training shards (100 x 100M tokens, ~20GB)
python fineweb.py --target-shards 2 --shard-size 1000000   # tiny set for testing
python hellaswag.py                           # download eval set + self-test the render/score path
python shapes.py                              # print param counts for reference configs
```

There is no test suite. `preflight.py` and the `__main__` blocks of `hellaswag.py`
and `shapes.py` are the closest thing; `preflight.py` exits non-zero on failure and
asserts the model still builds at **152.8M params / ffn=2048**.

Any `TrainConfig` field is overridable from the CLI, e.g.
`--max-lr 1.8e-3 --no-compile --hellaswag-limit 1000`.

## Architecture

**`train.py`** holds everything for training: model, data loading, config, and the
`__main__` training loop. **`hellaswag.py`** is a standalone eval module.
**`fineweb.py`** is a one-shot data generator.

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
  `module._orig_mod.` key prefixes that will not load into a fresh `GPT`), and
  sampling / HellaSwag (their input shapes vary per call, so the compiled path
  would recompile continuously).

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

## Known gaps

- **Document separation is not implemented.** Deliberately deferred so run 1 stays
  comparable to the nanoGPT baseline; it is intended as run 2's single variable.
  Doing it needs a block-diagonal mask (FlexAttention) *and* per-document RoPE
  position resets — masking alone is half the fix.
- **Sampling recomputes the whole prefix every token** — the generation loop calls the
  model on the full growing sequence, which is ~256x wasted work over 500 tokens. A KV
  cache is the obvious next piece of work.
- TorchInductor cannot codegen complex operators, so the complex-valued RoPE falls back
  to eager. Harmless (compile still gives ~55% overall), but a real-valued cos/sin RoPE
  would close it.
- DDP *is* verified as of run 1: 1-GPU and 2-GPU runs produced bit-identical loss
  trajectories, which simultaneously proves the per-rank dataloader stride, the gradient
  all-reduce and the grad_accum normalisation. Note it cannot be tested locally — macOS
  gloo cannot complete rendezvous.
