"""
HellaSwag eval: 4-way multiple-choice benchmark to compare my model against GPT2

Run as:
$ python hellaswag.py
"""

import os
import urllib.request
import pandas as pd
import torch
import tiktoken
import torch.nn.functional as F

# github repo taken down
HS_URL = "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/validation-00000-of-00001.parquet"
DEFAULT_LOCAL_DIR = "hellaswag"

enc = tiktoken.get_encoding("gpt2")

def download(local_dir=DEFAULT_LOCAL_DIR):
    # download the set if not already downloaded, then return filepath
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), local_dir)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "validation.parquet")

    if os.path.exists(path):
        print(f"Already downloaded at -> {path}")
        return path

    print(f"downloading hellaswag validation set -> {path}")
    tmp = path + ".tmp"
    urllib.request.urlretrieve(HS_URL, tmp)
    os.replace(tmp, path)
    return path

def load_examples(local_dir=DEFAULT_LOCAL_DIR):
    # returns a dataframe. columns: ctx, endings, label
    return pd.read_parquet(download(local_dir))

def render_example(ctx, endings):
    # turns multi-choice into a tensor the model can score
    # returns: tokens (4, T) - candidates, right-padded with 0s, mask (4, T) - 1 on ending tokens, 0 on context and padding
    ctx_tokens = enc.encode(ctx)

    tok_rows, mask_rows = [], []
    for end in endings:
        end_tokens = enc.encode(" " + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    T = max(len(row) for row in tok_rows)
    tokens = torch.zeros(4, T, dtype=torch.long)
    mask = torch.zeros(4, T, dtype=torch.long)
    for i, (t, m) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(t)] = torch.tensor(t)
        mask[i, :len(m)] = torch.tensor(m)

    return tokens, mask

@torch.no_grad()
def score_example(model, tokens, mask, device):
    # returns the index of the most probable ending
    tokens, mask = tokens.to(device), mask.to(device)
    logits, _ = model(tokens)

    # position i's logits predicts the token at i+1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_tokens = tokens[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()

    # one loss per token rather than single averaged number
    flat = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_tokens.view(-1),
                           reduction="none")
    losses = flat.view(tokens.size(0), -1)

    losses = losses * shift_mask
    avg_loss = losses.sum(dim=1) / shift_mask.sum(dim=1)
    return avg_loss.argmin().item()

def evaluate(model, df, device, rank=0, world_size=1, limit=None):
    # score this rank's share of the examples, returns num_correct, num_total for this rank
    # -> caller must combine across the ranks
    num_correct = num_total = 0
    for i, row in enumerate(df.itertuples()):
        if limit is not None and i >= limit:
            break
        if i % world_size != rank:
            continue
        tokens, mask = render_example(row.ctx, list(row.endings))
        pred = score_example(model, tokens, mask, device)
        num_correct += (pred == int(row.label))
        num_total += 1
    return num_correct, num_total

if __name__ == "__main__":
    df = load_examples()
    print(f"loaded {len(df)} examples")
    print(f"columns: {list(df.columns)}")

    r = df.iloc[0]
    print(f"\nctx: {r['ctx']}")
    for i, ending in enumerate(r['endings']):
        mark = " <- correct" if i == int(r['label']) else ""
        print(f"    [{i}] {ending}{mark}")

    tokens, mask = render_example(r['ctx'], list(r['endings']))
    print(f"\ntokens {tuple(tokens.shape)}  mask {tuple(mask.shape)}")
    for i in range(4):
        scored = enc.decode(tokens[i][mask[i] == 1].tolist())
        print(f"    row {i} scored region: {scored!r}")

