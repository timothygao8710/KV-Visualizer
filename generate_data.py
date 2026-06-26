#!/usr/bin/env python
"""
Generate the Llama-3.2-1B attention-explorer website (index.html + attn_data/*.json).

Self-contained: captures post-RoPE queries/keys from every layer/head of
Llama-3.2-1B-Instruct on the first 120 tokens of Hamlet's "To be, or not to be"
soliloquy, projects them into each head's top-3 PCA basis, and precomputes the FULL
head_dim-D causal attention + top-K + variance for the interactive viewer.

Outputs are written next to this script. Run it, then serve the folder (GitHub Pages or
`python -m http.server`).

Usage:
    pip install -r requirements.txt
    # meta-llama/Llama-3.2-1B-Instruct is gated: `huggingface-cli login` first,
    # or point LLAMA_PATH at a local copy.
    LLAMA_PATH=meta-llama/Llama-3.2-1B-Instruct python generate_data.py
"""

import json
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.llama.modeling_llama as llama_mod

N_VIS = 120          # tokens visualized
TOPK = 5             # highlighted attended keys per query
DEFAULT_L, DEFAULT_H = 0, 2

HERE = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(HERE, "attn_data")
OUT = os.path.join(HERE, "index.html")
MODEL = os.environ.get("LLAMA_PATH", "meta-llama/Llama-3.2-1B-Instruct")

SOLILOQUY = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die-to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream-ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause-there's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
Th'oppressor's wrong, the proud man's contumely,
The pangs of dispriz'd love, the law's delay,
The insolence of office, and the spurns
That patient merit of th'unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country, from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry
And lose the name of action."""


def causal_softmax(logits):
    T = logits.shape[0]
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    x = np.where(mask, -np.inf, logits)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def capture():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    cap = []
    orig = llama_mod.apply_rotary_pos_emb

    def hook(q, k, cos, sin, unsqueeze_dim=1):
        qe, ke = orig(q, k, cos, sin, unsqueeze_dim)
        cap.append((qe.detach().float(), ke.detach().float()))
        return qe, ke

    llama_mod.apply_rotary_pos_emb = hook
    enc = tok(SOLILOQUY, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :N_VIS]
    with torch.no_grad():
        model(enc)
    llama_mod.apply_rotary_pos_emb = orig

    nQ = cap[0][0].shape[1]; nKV = cap[0][1].shape[1]; hd = cap[0][0].shape[-1]
    q_all = [c[0][0].numpy().astype(np.float64) for c in cap]
    k_all = [c[1][0].numpy().astype(np.float64) for c in cap]
    tokens = [tok.decode([int(t)]) for t in enc[0]]
    return q_all, k_all, nQ, nKV, nQ // nKV, hd, tokens


def head_data(Q, K, scale):
    unitQ = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    X = np.concatenate([K, unitQ], 0)
    Xc = X - X.mean(0, keepdims=True)
    evals, evecs = np.linalg.eigh((Xc.T @ Xc) / (X.shape[0] - 1))
    order = np.argsort(evals)[::-1]
    V3 = evecs[:, order[:3]]
    Qr, Kr = Q @ V3, K @ V3
    P = causal_softmax((Q @ K.T) * scale)                          # full head_dim-D attention
    ev = evals[order] / evals[order].sum()
    attn = [np.round(P[j, :j + 1], 4).tolist() for j in range(Q.shape[0])]   # ragged lower-tri
    topk = [[int(i) for i in np.argsort(P[j, :j + 1])[::-1][:TOPK]] for j in range(Q.shape[0])]
    return dict(Qr=np.round(Qr, 4).tolist(), Kr=np.round(Kr, 4).tolist(),
                attn=attn, topk=topk, ev=np.round(ev[:8], 4).tolist())


def main():
    print(f"loading {MODEL} ...")
    q_all, k_all, nQ, nKV, grp, hd, tokens = capture()
    scale = 1.0 / np.sqrt(hd)
    nL = len(q_all)
    os.makedirs(DATADIR, exist_ok=True)

    default = None
    for L in range(nL):
        for h in range(nQ):
            d = head_data(q_all[L][h][:N_VIS], k_all[L][h // grp][:N_VIS], scale)
            d["kvh"] = h // grp
            with open(os.path.join(DATADIR, f"L{L}_h{h}.json"), "w") as f:
                json.dump(d, f)
            if L == DEFAULT_L and h == DEFAULT_H:
                default = d
    print(f"wrote {nL * nQ} head files -> {DATADIR}")

    meta = dict(tokens=tokens, nL=nL, nQ=nQ, hd=int(hd), TOPK=TOPK,
                defL=DEFAULT_L, defH=DEFAULT_H, default=default)
    with open(OUT, "w") as f:
        f.write(TEMPLATE.replace("__JSON__", json.dumps(meta)))
    print(f"saved -> {OUT}")


TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Llama-3.2-1B Attention Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  html,body{margin:0;height:100%;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
  #wrap{display:flex;height:100vh}
  #left{flex:1;min-width:0}
  #plot{width:100%;height:100%}
  #right{width:470px;display:flex;flex-direction:column;border-left:1px solid #ddd}
  #sel{padding:8px 12px;background:#fafafa;border-bottom:1px solid #eee;font-size:13px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
  #sel select{font-size:13px;padding:2px}
  #sel button{font-size:12px;padding:3px 8px;cursor:pointer}
  #var{padding:8px 12px;font-size:12px;border-bottom:1px solid #eee}
  .bar{display:inline-block;height:10px;background:#888;vertical-align:middle;border-radius:2px}
  #ctrl{padding:8px 12px;display:flex;align-items:center;gap:8px;border-bottom:1px solid #eee}
  #ctrl button{font-size:18px;padding:2px 12px;cursor:pointer}
  #pos{font-size:13px;min-width:150px}
  #slider{flex:1}
  #text{padding:12px;overflow-y:auto;line-height:2.0;white-space:pre-wrap;font-size:15px;min-height:0}
  .tok{border-radius:3px;padding:1px 0;cursor:pointer}
  .tok.cur{outline:2px solid #1565c0}
  .tok.future{opacity:.3}
  .tok.topk{text-decoration:underline;text-decoration-color:crimson;text-decoration-thickness:2px}
  .legend{padding:6px 12px;font-size:12px;color:#555;border-top:1px solid #eee}
  .sw{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:middle;margin:0 3px}
  #impl{border-top:1px solid #eee;font-size:12px}
  #impl>summary{padding:8px 12px;cursor:pointer;background:#eaf1ff;color:#1565c0;font-weight:600;list-style:none;user-select:none}
  #impl>summary::-webkit-details-marker{display:none}
  #impl>summary::before{content:'\25B8  '}
  #impl[open]>summary::before{content:'\25BE  '}
  #impl .body{padding:8px 12px 12px;max-height:45vh;overflow:auto;color:#333}
  #impl p{margin:.5em 0}
  #impl pre{background:#f6f8fa;padding:8px;border-radius:5px;overflow:auto}
  #impl code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;line-height:1.45}
</style></head>
<body><div id="wrap">
  <div id="left"><div id="plot"></div></div>
  <div id="right">
    <div id="sel">
      <span>Layer <select id="selL"></select></span>
      <span>Head <select id="selH"></select></span>
      <span id="kvlbl"></span>
      <button id="reset" title="look down the -PC1 axis">Reset view</button>
    </div>
    <div id="var"></div>
    <div id="ctrl">
      <button id="prev">&#9664;</button>
      <button id="next">&#9654;</button>
      <span id="pos"></span>
      <input id="slider" type="range" min="0" value="0">
    </div>
    <div id="text"></div>
    <div class="legend">
      <span class="sw" style="background:crimson"></span>key
      &nbsp;<b style="color:crimson">&#10142;</b> top-K attended key (bold arrow)
      &nbsp;<span class="sw" style="background:royalblue"></span>query &nbsp;|&nbsp;
      text shaded by attention (darker = more)
    </div>
    <details id="impl"><summary>Implementation Details</summary>
      <div class="body">
        <p>We do everything <b>post-RoPE</b>, since these are the vectors that actually enter
        attention (the pre-RoPE vectors are more low-rank). RoPE shows up here as a rotation of the
        Q/K's around the PC1 axis &mdash; applying RoPE rotates the <i>i</i>-th pair by
        <code>freq_i &middot; pos_idx</code>. Everything below is post-RoPE.</p>
        <p>For each layer, for each query head independently (keys are shared within its GQA group),
        we find PC1, PC2, PC3 with:</p>
        <pre><code>def get_comps(Q, K):                      # Q, K: (seq_len, head_dim)
    unitQ = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    X  = np.concatenate([K, unitQ], axis=0)
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)
    return evecs[:, ::-1][:, :3].T        # (3, head_dim)</code></pre>
        <p>For each query <i>i</i> we run ordinary causal attention, then keep the
        <b>top-5</b> attended keys:</p>
        <pre><code>score_j   = &lt;q_i, k_j&gt; / sqrt(head_dim)     for all j &lt;= i
attention = softmax(score)                    # over the sequence
top5      = np.argsort(attention)[::-1][:5]    # indices of the 5 most-attended keys</code></pre>
        <p>Finally we project all vectors onto the PCA-reduced basis (raw / uncentered) and draw
        them with Plotly. The projection is lossy (top-3 &asymp; 50&ndash;65% of variance) and the
        top-5 come from the <i>full</i>-dimensional attention, so a highlighted key may not be the
        nearest one in the 3-D view.</p>
      </div>
    </details>
  </div>
</div>
<script>
const M = __JSON__;
const N = M.tokens.length, tokens = M.tokens, TOPK = M.TOPK, hd = M.hd;
let H = M.default, cur = 0, coneSize = 1, inited = false;
const cache = {}; cache[M.defL + '_' + M.defH] = M.default;

const selL = document.getElementById('selL'), selH = document.getElementById('selH');
for(let l=0;l<M.nL;l++) selL.add(new Option('L'+l, l));
for(let h=0;h<M.nQ;h++) selH.add(new Option('h'+h, h));
selL.value = M.defL; selH.value = M.defH;
selL.onchange = selH.onchange = ()=>loadHead(+selL.value, +selH.value);

const slider = document.getElementById('slider'); slider.max = N - 1;
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

let h0='';
for(let i=0;i<N;i++){ h0 += '<span class="tok" id="tok'+i+'" data-i="'+i+'">'+esc(tokens[i])+'</span>'; }
const textDiv = document.getElementById('text');
textDiv.innerHTML = h0;
textDiv.addEventListener('click', e=>{ if(e.target.dataset.i!==undefined) setCur(+e.target.dataset.i); });

function median(a){const s=[...a].sort((x,y)=>x-y);return s[Math.floor(s.length/2)]||1;}

function lineMarker(indices, coords, color, width, mk, opacity, nm){
  let xs=[],ys=[],zs=[],mx=[],my=[],mz=[],tx=[];
  for(const i of indices){
    xs.push(0,coords[i][0],null); ys.push(0,coords[i][1],null); zs.push(0,coords[i][2],null);
    mx.push(coords[i][0]); my.push(coords[i][1]); mz.push(coords[i][2]);
    tx.push(nm+' '+i+': '+tokens[i]);
  }
  return [
    {type:'scatter3d',mode:'lines',x:xs,y:ys,z:zs,line:{color:color,width:width},
     opacity:opacity,name:nm,hoverinfo:'skip'},
    {type:'scatter3d',mode:'markers',x:mx,y:my,z:mz,marker:{color:color,size:mk},
     opacity:opacity,name:nm,showlegend:false,text:tx,hovertemplate:'%{text}<extra></extra>'}
  ];
}
function cones(indices, coords, color){
  let x=[],y=[],z=[],u=[],v=[],w=[];
  for(const i of indices){
    const p=coords[i], n=Math.hypot(p[0],p[1],p[2])||1;
    x.push(p[0]);y.push(p[1]);z.push(p[2]); u.push(p[0]/n);v.push(p[1]/n);w.push(p[2]/n);
  }
  return {type:'cone',x,y,z,u,v,w,anchor:'tip',sizemode:'absolute',sizeref:coneSize,
    colorscale:[[0,color],[1,color]],showscale:false,name:'top-K key',hoverinfo:'skip'};
}

// uirevision keeps the camera fixed across token steps / head switches; reset button overrides it.
const layout = {margin:{l:0,r:0,t:0,b:0},showlegend:true,uirevision:'keep',
  legend:{x:0,y:1,bgcolor:'rgba(255,255,255,.6)'},
  scene:{xaxis:{title:'PC1'},yaxis:{title:'PC2'},zaxis:{title:'PC3'},aspectmode:'data'}};

function resetView(){   // look INTO the -PC1 direction (eye on +PC1, gaze toward -PC1)
  Plotly.relayout('plot', {'scene.camera':
    {eye:{x:1.9,y:0,z:0}, center:{x:0,y:0,z:0}, up:{x:0,y:0,z:1}}});
}

function draw(){
  const Qr=H.Qr, Kr=H.Kr, tk=H.topk[cur];
  const keys=[...Array(cur+1).keys()];
  const nonTk=keys.filter(i=>!tk.includes(i));
  const queries=keys.filter(i=>i!==cur);
  let traces=[];
  traces=traces.concat(lineMarker(nonTk,Kr,'crimson',2,3,0.45,'key'));
  traces=traces.concat(lineMarker(tk,Kr,'crimson',7,6,1.0,'top-K key'));
  traces.push(cones(tk,Kr,'crimson'));
  traces=traces.concat(lineMarker(queries,Qr,'royalblue',1.5,3,0.30,'query'));
  traces=traces.concat(lineMarker([cur],Qr,'#0d47a1',7,8,1.0,'current query'));
  if(!inited){ Plotly.newPlot('plot',traces,layout,{responsive:true}); inited=true; resetView(); }
  else { Plotly.react('plot',traces,layout); }
}

function shadeText(){
  const row=H.attn[cur]; let mx=0; for(let i=0;i<=cur;i++) mx=Math.max(mx,row[i]); if(mx<=0) mx=1;
  const tk=H.topk[cur];
  for(let i=0;i<N;i++){
    const s=document.getElementById('tok'+i); s.className='tok';
    if(i>cur){ s.classList.add('future'); s.style.backgroundColor='transparent'; s.title=''; }
    else{
      s.style.backgroundColor='rgba(255,140,0,'+(row[i]/mx).toFixed(3)+')';
      s.title='attn '+row[i].toFixed(4);
      if(tk.includes(i)) s.classList.add('topk');
    }
    if(i===cur) s.classList.add('cur');
  }
  document.getElementById('pos').textContent='token '+cur+' / '+(N-1)+': '+JSON.stringify(tokens[cur]);
  slider.value=cur;
  const c=document.getElementById('tok'+cur); if(c) c.scrollIntoView({block:'nearest'});
}

function showVar(){
  const ev=H.ev, c3=(ev[0]+ev[1]+ev[2]);
  let s='<b>3-D plot variance</b> (of '+hd+' dims): '
    + 'PC1 '+(ev[0]*100).toFixed(1)+'% &middot; PC2 '+(ev[1]*100).toFixed(1)
    + '% &middot; PC3 '+(ev[2]*100).toFixed(1)+'% &rarr; <b>top-3 = '+(c3*100).toFixed(1)+'%</b><br>';
  for(let i=0;i<ev.length;i++){
    const col=i<3?'crimson':'#aaa';
    s+='<span class="bar" style="width:'+(ev[i]*260).toFixed(0)+'px;background:'+col+'"></span> '
       +'PC'+(i+1)+' '+(ev[i]*100).toFixed(1)+'%<br>';
  }
  document.getElementById('var').innerHTML=s;
  document.getElementById('kvlbl').textContent='(kv'+H.kvh+')';
}

function render(){ coneSize=0.28*median(H.Kr.map(p=>Math.hypot(p[0],p[1],p[2]))); draw(); shadeText(); showVar(); }

function loadHead(l,h){
  const key=l+'_'+h;
  if(cache[key]){ H=cache[key]; render(); return; }
  document.getElementById('var').textContent='loading L'+l+' h'+h+' ...';
  fetch('attn_data/L'+l+'_h'+h+'.json').then(r=>r.json()).then(d=>{cache[key]=d;H=d;render();})
    .catch(()=>{document.getElementById('var').textContent=
      'Could not fetch head file - serve the folder over http (python -m http.server).';});
}

function setCur(i){ cur=Math.max(0,Math.min(N-1,i)); render(); }
document.getElementById('prev').onclick=()=>setCur(cur-1);
document.getElementById('next').onclick=()=>setCur(cur+1);
document.getElementById('reset').onclick=resetView;
slider.oninput=()=>setCur(+slider.value);
document.addEventListener('keydown',e=>{
  if(e.key==='ArrowRight'||e.key==='ArrowDown'){setCur(cur+1);e.preventDefault();}
  if(e.key==='ArrowLeft'||e.key==='ArrowUp'){setCur(cur-1);e.preventDefault();}
});
render();
</script></body></html>
"""


if __name__ == "__main__":
    main()
