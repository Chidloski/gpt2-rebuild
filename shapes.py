import torch
from train import GPT, GPTConfig

CONFIGS = {
    "llama3-8b": GPTConfig(block_size=8192, vocab_size=128526, n_layer=32, n_head=32, n_kv_head=8, n_embd=4096),
    "small": GPTConfig(block_size=1024, vocab_size=128256, n_layer=12, n_head=12, n_kv_head=4, n_embd=768)
}

for name, cfg in CONFIGS.items():
    with torch.device('meta'):
        model = GPT(cfg)
    n = sum(p.numel() for p in model.parameters())
    mlp = model.transformer.h[0].mlp
    attn = model.transformer.h[0].attn 
    print(f"{name:12} {n/1e9:6.3f}B ffn={mlp.w1.out_features:<6}"
          f"head_dim={attn.head_dim:<4} wk={tuple(attn.wk.weight.shape)}")
