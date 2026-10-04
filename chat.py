"""
Interactive REPL for a trained checkpoint

    python chat.py  # loads weights.pt
    python chat.py --ckpt log/run/ckpt_last.pt --temperature 0.9

Will use a BASE model with no instruction tuning which completes text rather than asnwering questions unless an sft model is loaded
"""
import argparse
import torch
from train import GPT, GPTConfig, enc # reusing train.py's tokenizer
from sft_data import SYSTEM, END, USER, ASSISTANT, decode_with_special

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
    is_sft = "sft_config" in ck

    # shape comes from the checkpoint, thus is guaranteed to match the weights
    model = GPT(GPTConfig(**ck["model_config"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    val = ck.get("val_loss")
    print(f"{path}: step {ck.get('step', '?')}"
          + (f", val {val:.4f}" if val is not None else "")
          + f", {n:.1f}M params on {device}")
    return model, is_sft

def trim_history(history, max_new_tokens, block_size):
    while len(history) + max_new_tokens > block_size:
        assert history[0] == USER, f"history[0] is not USER"
        index = 1
        while  index < len(history) and history[index] != USER:
            index += 1
        if index == len(history):
            return history
        else:
            history = history[index:]

    return history

HELP = """  /temp <float>    sampling temperature (0 = greedy)
    /tokens <int>    max new tokens
    /topk <int>      top-k cutoff (0 = off)
    /system <string> alter the system prompt and reset history
    /reset           reset history
    /history         print history
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
    model, is_sft = load_model(args.ckpt, device)

    # bfloat only pays off on cuda
    autocast_dtype = torch.bfloat16 if device == "cuda" else None

    rng = None
    if args.seed is not None:
        rng = torch.Generator(device=device)
        rng.manual_seed(args.seed)

    print("prompt, /help for commands, ctrl-d to quit\n")
    last = None

    history = []
    system_prefix = []

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
            elif cmd == "/reset":
                if is_sft:
                    history = []
                    print("History cleared")
                else:
                    print("/reset, /system and /history are chat-mode only")
            elif cmd == "/system":
                if is_sft:
                    if arg:
                        new_prefix = [SYSTEM] + enc.encode_ordinary(arg) + [END]
                        if len(new_prefix) + args.max_new_tokens >= model.config.block_size:
                            print("System prompt too long")
                        else:
                            system_prefix = new_prefix
                            history = []
                            print("System prompt changed, history cleared")
                    else:
                        system_prefix = []
                        history = []
                        print("System prompt changed, history cleared")
                else:
                    print("/reset, /system and /history are chat-mode only")
            elif cmd == "/history":
                if is_sft:
                    print("".join(decode_with_special(system_prefix + history)))
                    print(f"[{len(system_prefix) + len(history)} / {model.config.block_size} tokens]")
                else:
                    print("/reset, /system and /history are chat-mode only")
            else:
                print(f"unknown command {cmd!r}")
            if line.startswith("/"):
                continue

        if not is_sft:
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
        else:
            history += [USER] + enc.encode_ordinary(line) +[END, ASSISTANT]
            history = trim_history(history, args.max_new_tokens, model.config.block_size - len(system_prefix))
            room = model.config.block_size - len(history) - len(system_prefix)

            if room <= 0:
                print("message too long")
                history = []
                continue

            out = model.generate(
                torch.tensor(system_prefix + history, dtype=torch.long, device=device).unsqueeze(0),
                min(room, args.max_new_tokens),
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token=END,
                generator=rng,
                autocast_dtype=autocast_dtype,
            )
            reply = out[0, len(system_prefix) + len(history):].tolist()
            stopped = reply[-1] == END
            if stopped:
                history += reply
                reply = reply[:-1]
            else:
                history += reply + [END]
            text = "".join(decode_with_special(reply)) + (f" [cut off at {len(reply)} tokens]" if not stopped else "")
            print(text)

            





if __name__ == "__main__":
    main()