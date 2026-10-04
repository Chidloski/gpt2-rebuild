import numpy as np
import multiprocessing as mp
import os
from tqdm import tqdm
from tiktoken import get_encoding
enc = get_encoding("gpt2")

USER, ASSISTANT, SYSTEM, END = 50257, 50258, 50259, 50260
ROLE_TOKEN = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}
TOKEN_LOOKUP = {50257: "<USER>", 50258: "<ASSISTANT>", 50259: "<SYSTEM>", 50260: "<END>"}
MAX_LEN = 1025 # 1025 rather than 1024 as [:-1] are used to predict [1:] tokens

def decode_with_special(tokens):
    message = []

    for token in tokens:
        if token > 50256:
            message.append(TOKEN_LOOKUP[token])
        else:
            message.append(enc.decode([token]))

    return message

def segment(role, content):
    message = [ROLE_TOKEN[role]] + enc.encode_ordinary(content) + [END]

    if role == "assistant":
        mask = [0] + [1]*(len(message) - 1)
    else:
        mask = [0]*(len(message))

    return message, mask

def render(messages, max_len=MAX_LEN):
    valid = False
    all_tokens = []
    all_masks = []
    length = 0

    if messages[0]["role"] == "system":
        tokens, mask = segment(messages[0]["role"], messages[0]["content"])
        length += len(tokens)
        all_tokens += tokens
        all_masks += mask
        messages = messages[1:]

    for i in range(len(messages) // 2):
        assert messages[2*i]["role"] == "user", "user messages did not come first"
        assert messages[2*i + 1]["role"] == "assistant", "no assistant message in pair"
        user_tokens, user_mask = segment(messages[2*i]["role"], messages[2*i]["content"])
        assistant_tokens, assistant_mask = segment(messages[2*i + 1]["role"], messages[2*i + 1]["content"])

        length += len(user_tokens) + len(assistant_tokens)

        if length > max_len:
            if not valid:
                return None
            else:
                return all_tokens, all_masks, True

        valid = True
        all_tokens += user_tokens + assistant_tokens
        all_masks += user_mask + assistant_mask

    if not valid:
        return None
    else:
        return all_tokens, all_masks, False

def show(tokens, mask):
    message = decode_with_special(tokens)
    debug_message = []
    for index, value in enumerate(mask):
        if value == 0 and index != 0 and mask[index - 1] == 1:
            debug_message.append("»")
        elif value == 1 and index != 0 and mask[index - 1] == 0:
            debug_message.append("«")
        debug_message.append(message[index])

    if mask[-1] == 1:
        debug_message.append("»")

    print("".join(debug_message))

def process_row(row):
    result = render(row["messages"])
    if result:
        tokens, mask, truncated = result
        return np.array(tokens, dtype=np.uint16), np.array(mask, dtype=np.uint8), truncated
    else:
        return None

def build_split(ds_split, name):
    all_tokens = []
    all_mask = []
    all_lengths = [0]
    kept = 0
    truncated = 0
    dropped = 0

    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        for result in tqdm(pool.imap(process_row, ds_split, chunksize=256), total=len(ds_split)):
            if result:
                tokens, mask, was_truncated = result
                all_tokens.append(tokens)
                all_mask.append(mask)
                all_lengths.append(len(tokens))

                kept += 1
                if was_truncated:
                    truncated += 1

            else:
                dropped += 1 

    tokens = np.concatenate(all_tokens)
    mask = np.concatenate(all_mask)
    offsets = np.cumsum(all_lengths)
    assert offsets[-1] == len(tokens) == len(mask), f"offsets[-1] does not match len(tokens)"

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sft_data")
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, f"{name}_tokens"), tokens)
    np.save(os.path.join(DATA_DIR, f"{name}_mask"), mask)
    np.save(os.path.join(DATA_DIR, f"{name}_offsets"), offsets)

    print(f"{name}:\n"
          f"{kept} datapoints were kept ({((kept / (kept + dropped))*100):.1f}%)\n"
          f"    of which {truncated} were truncated ({((truncated / (kept))*100):.1f}%)\n"
          f"{(mask.mean()*100):.1f}% of this data is trained upon")

if __name__ == "__main__":
    import argparse
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N rows of each split (for testing)")
    args = parser.parse_args()

    ds = load_dataset("HuggingFaceTB/smol-smoltalk")   # served from the local HF cache

    for hf_split, name in [("test", "val"), ("train", "train")]:
        split = ds[hf_split]
        if args.limit is not None:
            split = split.select(range(min(args.limit, len(split))))
        build_split(split, name)