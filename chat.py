"""
Interactive REPL for a trained checkpoint

    python chat.py  # loads weights.pt
    python chat.py --ckpt log/run/ckpt_last.pt --temperature 0.9

Will use a BASE model with no instruction tuning. It completes text rather than asnwering questions
"""
import argparse
import torch
from train import GPT, GPTConfig, enc # reusing train.py's tokenizer

EOT = enc.eot_token # 50256, <|endoftext|>

def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    # shape comes from the checkpoint, thus is guaranteed to match the weights
    model = GPT(GPTConfig(**ck["model_config"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    val = ck.get("val_loss")
    print(f"{path}: step {ck.get('step', '?')}"
          + (f", val {val:.4f}" if val is not None else "")
          + f", {n:.1f}M params on {device}")
    return model

HELP = """  /temp <float>    sampling temperature (0 = greedy)
    /tokens <int>    max new tokens
    /topk <int>      top-k cutoff (0 = off)
    /more            continue from the last completion
    /help            this
    ctrl-d           quit"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="weights.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    model = load_model(args.ckpt, device)

    # bfloat only pays off on cuda
    autocast_dtype = torch.bfloat16 if device == "cuda" else None

    rng = None
    if args.seed is not None:
        rng = torch.Generator(device=device)
        rng.manual_seed(args.seed)

    print("prompt to complete, /help for commands, ctrl-d to quit\n")
    last = None

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            if cmd == "/help":
                print(HELP)
            elif cmd == "/temp":
                args.temperature = float(arg)
            elif cmd == "/tokens":
                args.max_new_tokens = int(arg)
            elif cmd == "/topk":
                args.top_k = int(arg) or None
            elif cmd == "/more" and last is not None:
                line = last
            else:
                print(f"unknown command {cmd!r}")
            if line.startswith("/"):
                continue

        tokens = enc.encode(line)
        room = model.config.block_size - len(tokens)
        if room <= 0:
            print(f"prompt is {len(tokens)} tokens, block_size is {model.config.block_size}")
            continue

        out = model.generate(
            torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0),
            min(args.max_new_tokens, room),
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token=EOT,
            generator=rng,
            autocast_dtype=autocast_dtype,
        )
        text = enc.decode(out[0].tolist())
        print(text[len(line):])
        last = text


if __name__ == "__main__":
    main()