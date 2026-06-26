# Llama-3.2-1B Attention Explorer

An interactive, browser-based visualization of how a single attention head in
**Llama-3.2-1B-Instruct** reads a passage of text — Hamlet's *"To be, or not to be"*
soliloquy.

- **Left panel** — a 3D plot of the head's **keys (red)** and **queries (blue)** as vectors,
  projected into that head's **top-3 PCA basis**. The cloud **grows** as you step through the
  text. The current query's **top-K attended keys are drawn as bold red arrows**.
- **Right panel** — the text itself, with every preceding token **shaded orange by how much
  attention** the current query pays it (darker = more), plus layer/head dropdowns and the
  variance explained by the 3 plotted components.

Pick any of the 16 layers × 32 heads, step through tokens with the arrows / slider / arrow
keys, click a token to jump, and hit **Reset view** to look straight down the −PC1 axis.

## View it

**Live (GitHub Pages):** `https://<user>.github.io/<repo>/`

**Locally** (the dropdowns fetch per-head JSON, so it must be served over http, not `file://`):

```bash
python -m http.server          # from this folder
# open http://localhost:8000/
```

## Deploy to GitHub Pages

1. Put the contents of this folder at the root of a repo (or a subfolder you'll serve):
   `index.html`, `attn_data/`, `.nojekyll`.
2. Push, then enable **Settings → Pages** (branch = your branch, folder = `/` or `/docs`).
3. Open `https://<user>.github.io/<repo>/`.

`.nojekyll` disables Jekyll so all files are served verbatim. Keep `attn_data/` next to
`index.html` — the page loads `attn_data/L{layer}_h{head}.json` by relative path. Total size
is ~32 MB across 512 small files, well within Pages' limits.

## Reproduce the data

```bash
pip install -r requirements.txt
# meta-llama/Llama-3.2-1B-Instruct is gated — `huggingface-cli login` first,
# or set LLAMA_PATH to a local copy of the model.
LLAMA_PATH=meta-llama/Llama-3.2-1B-Instruct python generate_data.py
```

This regenerates `index.html` and all of `attn_data/`. Edit the constants at the top of
`generate_data.py` to change the number of tokens (`N_VIS`), the top-K (`TOPK`), the default
head, or the input text (`SOLILOQUY`). CPU is fine.

## How it works

- Queries/keys are captured **post-RoPE** (Llama has no QK-norm) by wrapping
  `apply_rotary_pos_emb`, for the first `N_VIS = 120` tokens.
- For each head, a **top-3 PCA basis** is built from the combined `[K, unitQ]` cloud and the raw
  q/k are projected into it for display.
- Attention shading and the top-K highlight use the **full `head_dim`-D causal attention**
  (`softmax(QKᵀ / √head_dim)`), *not* the 3-D projection.

### Caveat worth knowing
The top-3 components explain only ~**50–65%** of the variance, so the 3-D plot is a lossy view.
Because attention top-K is an arg-max of dot products (sensitive to the discarded dimensions),
the highlighted keys are computed in full dimensions and **may not be the ones nearest the query
in the 3-D plot** — only ~⅓ of the full-attention top-5 fall in the 3-D top-5 for a typical head.
The variance readout in the right panel tells you how trustworthy the projection is for the
current head.

## Repository layout

```
index.html            # the viewer (self-contained except Plotly via CDN)
attn_data/            # 512 files: L{layer}_h{head}.json (Qr, Kr, attn, topk, ev)
generate_data.py      # standalone script that produces the two above
requirements.txt
.nojekyll
```

## Credits & license

- **Code** (`generate_data.py`, the viewer) — MIT.
- **Data** in `attn_data/` is derived from **Llama-3.2-1B-Instruct** and is therefore subject to
  the [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE).
  "Built with Llama."
- The soliloquy text is from Shakespeare's *Hamlet* (public domain).
- Visualization uses [Plotly.js](https://plotly.com/javascript/) (MIT).
