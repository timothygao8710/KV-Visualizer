# Llama-3.2-1B Attention Explorer

https://timothygao8710.github.io/KV-Visualizer/

Using the low-rankness of the query-key space ([1][1], [2][2], [3][3]) to visualize mechanisms behind attention.

The visual lets us directly observe well-known emperical LLM findings. In L10H1, the attention sink appears close to nearly every query [4]. In L0H2, RoPE literally appears as a rotation (reset view) [5]. Across layers and heads, the cone-shaped geometry becomes visible in the QK point cloud [6]. Scaled dot-product attention is driven by query-key dot products,

$$
\langle q, k \rangle = \|q\| \|k\| \cos(\theta),
$$

for Llama, it is known the angular component, $\cos(\theta)$, explains most of the score variation; we see this is true in the visual as well, the top-K keys for each query are approximately the closest in angle [7].

[1]: https://arxiv.org/abs/2602.04752
[2]: https://arxiv.org/abs/2001.04451
[3]: https://arxiv.org/abs/2408.05646
[4]: https://arxiv.org/abs/2309.17453
[5]: https://arxiv.org/abs/2104.09864
[6]: https://arxiv.org/abs/2601.08297
[7]: https://arxiv.org/abs/2010.04245

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
