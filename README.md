# Llama-3.2-1B Attention Explorer

Using the low-rankedness of the Q-K space (1 -- paper low rankness, 2 -- paper on qk space is shared while v is not,3 -- using this fact for kv compression) to visualize the mechanisms behind attention. We see known phenomnoms like RoPE rotation litearlly instantianting itself physically as a rotation in L0H2 (reset view), attention sink being close to all the queries in L10H1, and <q, k> = ||q|| ||k|| cos(theta), the cos(theta) term most explains the attention score for Llama.

https://timothygao8710.github.io/KV-Visualizer/

## Reproduce the data

```bash
pip install -r requirements.txt
# meta-llama/Llama-3.2-1B-Instruct is gated — `huggingface-cli login` first,
# or set LLAMA_PATH to a local copy of the model.
LLAMA_PATH=meta-llama/Llama-3.2-1B-Instruct python generate_data.py
```

This regenerates `index.html` and all of `attn_data/`. Edit the constants at the top of
`generate_data.py` to change the number of tokens (`N_VIS`), the top-K (`TOPK`), the default
head, or the input text (`SOLILOQUY`). Works on CPU.
