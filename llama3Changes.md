# Changes in architecture to get from GPT2 to Llama 3

## GPT2 -> Llama 1
### Architecture Changes
- Sentence Piece tokenizer (50257 -> 32000)
- Pre-Normalisation via RMSNorm
- SwiGLU activation function, dimension of 2/3 * 4d
- Rotary Embeddings
- wte and lm_head are no longer the same matrix
- No bias terms anywhere
- Context window increases from 1024 to 2048

### Hyperparam changes (for 6.7B model)
- 3e-4 max lr, 2000 warmup steps, cosine decay to 0.1 of max lr
- 12 heads (gpt3) -> 32 heads
- 12 layers (gpt3) -> 32 layers
- 768 dims -> 4096 dims (embedding matrix)
- 300B Training Tokens -> 1.0T
*note this compares my current implementation of gpt3-125M to llama-6.7B, gpt3-6.7B and llama-6.7B have the same shape*

## Llama 1 -> Llama 2
- Increased context length, 2048 -> 4096
- Grouped-Query Attention (GQA) in 34B and 70B models

## Llama 2 -> LLama 3
- Vocab increase to 128K tokens
- GQA on all model sizes
- RoPE base frequency to 500,000
- Attention mask stops self-attention between different documents
- Context size 4096 -> 8192

## Llama 3 8B shape
- 32 Layers
- 4096 dimensions (embedding)
- 14336 dimennsions (FFN)
- 32 Attention heads, 8 KV Heads
- SwiGLU
- 128k vocab
- RoPE
- 3e-4 peak lr

