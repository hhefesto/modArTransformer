import { useState, useRef, useEffect } from "react";

/* ───────────────────────── colour palette ───────────────────────── */
const C = {
  bg:       "#06090f",
  panel:    "#0d1117",
  panelBdr: "#1b2535",
  card:     "#111927",
  bdr:      "#1e2d4a",
  txt:      "#bfcee0",
  dim:      "#4e6280",
  bright:   "#e4edf8",
  accent:   "#58b4f6",
  green:    "#5ec26a",
  greenBg:  "#0f1f14",
  orange:   "#f0a346",
  orangeBg: "#1f170d",
  pink:     "#e8628a",
  pinkBg:   "#1f0d15",
  purple:   "#a98be0",
  purpleBg: "#150d1f",
  yellow:   "#e6d34e",
  yellowBg: "#1f1d0d",
  cyan:     "#4fd1c5",
  cyanBg:   "#0d1f1d",
  res:      "#2e3a46",
};

/* ───────────────────────── block data ───────────────────────────── */
const BLOCKS = [
  {
    id: "embed", label: "Token + Position Embedding", sub: "53×64 + 2×64 → [x₀, x₁]",
    color: C.accent, bg: "#0d1a2a",
    what: `Each integer a, b ∈ [0..52] is looked up in a learned embedding table (53 rows × 64 cols) to produce a dense 64-dim vector. A separate positional embedding (2 rows × 64 cols) is added so the model knows which operand is which.

Both tables are learned end-to-end via backpropagation — there is no hand-designed encoding.`,
    why: `Raw integers carry no geometric structure. The embedding gives each number a rich learned representation where modular-arithmetic patterns (like periodicity mod 53) can emerge as directions in ℝ⁶⁴.

Position embedding breaks the permutation symmetry that attention would otherwise have — without it the model couldn't distinguish (a, b) from (b, a).`,
    params: `tokEmbed  53 × 64  =  3,392
posEmbed   2 × 64  =    128
─────────────────────────────
Subtotal              3,520`,
    hs: `embA = tokEmbed LA.! tokA        -- row lookup
embB = tokEmbed LA.! tokB
x0   = embA + (posEmbed LA.! 0)  -- add position
x1   = embB + (posEmbed LA.! 1)
-- embedded = [x0, x1]`,
    shapes: `a, b  ∈  ℤ  (scalars in [0..52])
         ↓  lookup + elementwise add
[x₀, x₁]  ∈  ℝ²ˣ⁶⁴  (2 tokens, 64-dim each)`,
  },
  {
    id: "attn", label: "Single-Head Self-Attention", sub: "Wq, Wk, Wv, Wo each 64×64 — scores 2×2",
    color: C.purple, bg: C.purpleBg,
    what: `Every token is linearly projected into a Query, Key, and Value:
  Q = Wq·x + bq,   K = Wk·x + bk,   V = Wv·x + bv

Dot-product scores QKᵀ/√64 form a 2×2 attention matrix. After row-wise softmax, each token's output is a weighted sum of Values, projected through Wₒ:
  attended = softmax(QKᵀ/√dₖ) · V
  out = Wₒ · attended + bₒ

No causal mask is used — both tokens see each other.`,
    why: `This is the only place the two operands exchange information. The 2×2 attention matrix controls how much of operand b's representation gets mixed into operand a's (and vice versa).

With no causal mask both tokens see each other freely — this is appropriate because addition is commutative and both operands are needed equally.`,
    params: `Wq  64×64 + bq 64  =  4,160
Wk  64×64 + bk 64  =  4,160
Wv  64×64 + bv 64  =  4,160
Wₒ  64×64 + bₒ 64  =  4,160
────────────────────────────
Subtotal             16,640`,
    hs: `qs = map (linearForward attnWq) xs
ks = map (linearForward attnWk) xs
vs = map (linearForward attnWv) xs

-- scale factor
sf = 1 / sqrt (fromIntegral dK)

-- 2×2 attention weights
weights = [ softmax $ scale sf $
              fromList [qi <.> kj | kj <- ks]
          | qi <- qs ]

-- weighted sum of values
attended = [ Σⱼ wᵢⱼ · vⱼ | wᵢ <- weights ]

-- output projection
outs = map (linearForward attnWo) attended`,
    shapes: `Q, K, V      ∈  ℝ²ˣ⁶⁴
raw scores   ∈  ℝ²ˣ²   (before softmax)
weights      ∈  ℝ²ˣ²   (after softmax, rows sum to 1)
attended     ∈  ℝ²ˣ⁶⁴
attnOut      ∈  ℝ²ˣ⁶⁴  (after Wₒ projection)`,
  },
  {
    id: "res1", label: "Residual #1 — Add", sub: "embedded + attnOut",
    color: C.cyan, bg: C.cyanBg, small: true,
    what: `Element-wise addition:
  residual₁ᵢ = xᵢ + attnOutᵢ   for each token i`,
    why: `The skip connection lets gradients flow directly back to the embeddings, preventing vanishing gradients. It also lets attention learn a "correction" (delta) to the input rather than re-encoding the full signal.`,
    params: `0 learnable parameters — just element-wise addition.`,
    hs: `residual1 = zipWith (+) embedded attnOut`,
    shapes: `ℝ²ˣ⁶⁴ + ℝ²ˣ⁶⁴  →  ℝ²ˣ⁶⁴`,
  },
  {
    id: "ln1", label: "LayerNorm #1", sub: "γ, β ∈ ℝ⁶⁴  (shared across tokens)",
    color: C.green, bg: C.greenBg,
    what: `Per token: subtract mean, divide by std-dev, then apply a learned affine transform:
  μ = mean(x)
  x̂ = (x − μ) / √(σ² + ε)
  out = γ · x̂ + β

This is post-norm placement (normalisation happens after the residual, not before attention). ε = 1e-5 for numerical stability.`,
    why: `Keeps activations on a stable scale so the downstream FFN receives well-conditioned inputs. Without it, magnitudes can drift during training and make optimisation fragile.

The learned γ and β let the model recover any scale or shift it needs — LN doesn't permanently constrain the representation.`,
    params: `γ (gamma)   64 params
β (beta)    64 params
──────────────────────
Subtotal       128     (shared across both tokens)`,
    hs: `layerNormForward LayerNormParams{..} x =
  let mu   = sumElements x / d
      diff = x - konst mu n
      var  = sumElements (diff * diff) / d
      invS = 1 / sqrt (var + 1e-5)
      xHat = scale invS diff
      out  = lnGamma * xHat + lnBeta
  in  (out, cache)`,
    shapes: `Per token:  ℝ⁶⁴ → ℝ⁶⁴  (normalised, same shape)`,
  },
  {
    id: "ffn", label: "Feed-Forward Network (MLP)", sub: "64 → 256 → ReLU → 64   per token",
    color: C.orange, bg: C.orangeBg,
    what: `Two linear layers with ReLU between them, applied independently to each token:
  hidden = ReLU(W₁ · x + b₁)      64 → 256  (expand)
  out    = W₂ · hidden + b₂       256 → 64   (compress)

The 4× expansion ratio (64 → 256) is standard. ReLU zeroes negative activations, providing the essential nonlinearity.`,
    why: `Attention mixes information across tokens; the FFN transforms each token's blended representation through a nonlinear bottleneck. The expansion to 256 dims gives the network enough parameters and capacity to learn the complex modular-arithmetic mapping.

This is where most of the model's parameters live (33K of ~57K total). The "grokking" phenomenon likely involves these weights slowly finding Fourier-like representations of mod-53 arithmetic.`,
    params: `W₁  256×64 + b₁ 256  =  16,640
W₂   64×256 + b₂  64  =  16,448
──────────────────────────────
Subtotal                33,088`,
    hs: `ffnForward FFNParams{..} x =
  let h   = linearForward ffnLinear1 x   -- 64 → 256
      act = relu h                        -- max(0, ·)
      out = linearForward ffnLinear2 act  -- 256 → 64
  in  (out, FFNCache h act)`,
    shapes: `Per token:  ℝ⁶⁴ → ℝ²⁵⁶ → ℝ⁶⁴
(the 256-dim hidden is cached for backprop)`,
  },
  {
    id: "res2", label: "Residual #2 — Add", sub: "ln1Out + ffnOut",
    color: C.cyan, bg: C.cyanBg, small: true,
    what: `Same as Residual #1 but wrapping the FFN:
  residual₂ᵢ = ln1Outᵢ + ffnOutᵢ`,
    why: `Provides a gradient highway around the FFN and lets it learn incremental refinements rather than full representations from scratch.`,
    params: `0 learnable parameters.`,
    hs: `residual2 = zipWith (+) ln1Out ffnOut`,
    shapes: `ℝ²ˣ⁶⁴ + ℝ²ˣ⁶⁴  →  ℝ²ˣ⁶⁴`,
  },
  {
    id: "ln2", label: "LayerNorm #2", sub: "γ, β ∈ ℝ⁶⁴",
    color: C.green, bg: C.greenBg,
    what: `Same operation as LN1 — normalize then affine transform — but with its own separate learned γ and β.`,
    why: `Normalises the signal before the classifier head so the unembed projection receives inputs with stable mean and variance regardless of how large or small the FFN outputs become during training.`,
    params: `γ (gamma)   64 params
β (beta)    64 params
──────────────────────
Subtotal       128`,
    hs: `ln2Pairs = map (layerNormForward ln2) residual2
ln2Out   = map fst ln2Pairs`,
    shapes: `Per token:  ℝ⁶⁴ → ℝ⁶⁴  (normalised)`,
  },
  {
    id: "readout", label: "First-Token Readout", sub: "head ln2Out  →  ℝ⁶⁴",
    color: C.yellow, bg: C.yellowBg, small: true,
    what: `Discard token 1 entirely. Only token 0's final representation is passed to the classifier.`,
    why: `Token 0 is designated as the "answer" position — similar to BERT's [CLS] token. Through attention, token 0 has already gathered all the information it needs from token 1 (the second operand).

Using just one token also halves the unembed parameters (53×64 instead of 53×128 if we concatenated both).`,
    params: `0 learnable parameters — just list indexing.`,
    hs: `readout = head ln2Out
-- Haskell 'head' = first element of list`,
    shapes: `[token₀, token₁]  ∈  ℝ²ˣ⁶⁴
              ↓  take position 0
         token₀  ∈  ℝ⁶⁴`,
  },
  {
    id: "unembed", label: "Unembed / Classifier + Softmax", sub: "Linear 64 → 53  then softmax → Δ⁵²",
    color: C.pink, bg: C.pinkBg,
    what: `A single linear layer projects from ℝ⁶⁴ to ℝ⁵³ (one logit per possible answer class). Softmax converts raw logits to a valid probability distribution over [0..52].

  logits = W · readout + b          (64 → 53)
  probs  = softmax(logits)

The predicted answer = argmax(probs).`,
    why: `This "decodes" the model's internal 64-dim representation back into the answer space. Each of the 53 outputs corresponds to one residue class mod 53. The softmax ensures probabilities sum to 1, making the loss well-defined.`,
    params: `W  53×64 + b 53  =  3,445
────────────────────────────
Subtotal            3,445`,
    hs: `logits = linearForward unembed readout
probs  = softmax logits`,
    shapes: `ℝ⁶⁴  →  logits ∈ ℝ⁵³  →  probs ∈ Δ⁵²
(Δ⁵² = probability simplex, sums to 1)`,
  },
  {
    id: "loss", label: "Cross-Entropy Loss", sub: "−log p_target  →  scalar ∈ ℝ",
    color: "#8899aa", bg: "#111520",
    what: `loss = −log(p_target)

where p_target is the softmax probability the model assigns to the correct answer (a+b) mod 53. The value is clamped: max(1e-12, p) for numerical safety.

This is the scalar that gets minimised by AdamW via backpropagation through the entire graph above.`,
    why: `Cross-entropy is the standard classification objective. It heavily penalises confident wrong predictions (when p_target ≈ 0 the loss → ∞) while rewarding confident correct ones (when p_target → 1 the loss → 0).

Gradient w.r.t. logits = (probs − one_hot), which is elegant and numerically stable. This gradient flows backward through every layer above.`,
    params: `0 learnable parameters — it is the objective function, not a layer.`,
    hs: `crossEntropyLoss probs target =
  let p = max 1e-12 (probs ! target)
  in  negate (log p)

-- gradient w.r.t. logits (before softmax):
crossEntropyGradLogits probs target =
  probs - assoc n 0 [(target, 1)]`,
    shapes: `probs ∈ ℝ⁵³  ×  target ∈ [0..52]
              ↓
         loss ∈ ℝ  (scalar)`,
  },
];

const TOTAL_PARAMS = "56,949";

/* ───────────────── info tabs ────────────────────────────────────── */
const TABS = ["what", "why", "params", "shapes", "hs"];
const TAB_LABELS = { what: "What", why: "Why", params: "Params", shapes: "Shapes", hs: "Haskell" };

function InfoPanel({ block }) {
  const [tab, setTab] = useState("what");

  useEffect(() => { setTab("what"); }, [block?.id]);

  if (!block) return (
    <div style={{
      padding: "40px 28px", color: C.dim,
      fontFamily: "'Geist', 'IBM Plex Sans', system-ui, sans-serif",
      textAlign: "center", lineHeight: 1.8,
    }}>
      <div style={{ fontSize: 40, marginBottom: 16, opacity: 0.18 }}>◆</div>
      <div style={{ fontSize: 14, fontWeight: 500 }}>Click any block to explore it</div>
      <div style={{ fontSize: 12, marginTop: 10, opacity: 0.55, lineHeight: 1.7 }}>
        Each panel shows <em>what</em> the layer does, <em>why</em> it's there,
        parameter counts, tensor shapes, and the actual Haskell code.
      </div>
      <div style={{
        marginTop: 28, padding: "14px 18px",
        background: "#0a0e18", borderRadius: 6,
        border: `1px solid ${C.panelBdr}`,
        fontSize: 11.5, textAlign: "left", lineHeight: 1.8,
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        <div style={{ color: C.dim, marginBottom: 4 }}>Total learnable parameters</div>
        <div style={{ color: C.accent, fontSize: 16, fontWeight: 700 }}>{TOTAL_PARAMS}</div>
        <div style={{ color: C.dim, marginTop: 10, fontSize: 10.5, lineHeight: 1.7 }}>
          Embed 3,520 · Attn 16,640 · LN×2 256
          <br/>FFN 33,088 · Unembed 3,445
        </div>
      </div>
    </div>
  );

  const content = block[tab] || "";
  return (
    <div style={{ fontFamily: "'Geist', 'IBM Plex Sans', system-ui, sans-serif" }}>
      <div style={{
        padding: "16px 22px 12px", borderBottom: `1px solid ${C.panelBdr}`,
        display: "flex", alignItems: "center", gap: 10,
      }}>
        <div style={{
          width: 10, height: 10, borderRadius: "50%",
          background: block.color, boxShadow: `0 0 10px ${block.color}44`,
          flexShrink: 0,
        }} />
        <div style={{ fontSize: 14, fontWeight: 700, color: C.bright, letterSpacing: "-0.01em" }}>
          {block.label}
        </div>
      </div>
      <div style={{
        display: "flex", borderBottom: `1px solid ${C.panelBdr}`, padding: "0 14px",
      }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: "none", border: "none", cursor: "pointer",
            padding: "10px 13px", fontSize: 11.5, fontWeight: 600,
            fontFamily: "'Geist', 'IBM Plex Sans', system-ui, sans-serif",
            color: tab === t ? block.color : C.dim,
            borderBottom: tab === t ? `2px solid ${block.color}` : "2px solid transparent",
            marginBottom: -1, transition: "color 0.15s",
          }}>
            {TAB_LABELS[t]}
          </button>
        ))}
      </div>
      <div style={{
        padding: (tab === "hs") ? 0 : "18px 22px",
        fontSize: 13, lineHeight: 1.8,
        color: C.txt,
        whiteSpace: "pre-wrap",
        fontFamily: (tab === "hs" || tab === "params" || tab === "shapes")
          ? "'JetBrains Mono', 'Fira Code', monospace"
          : "'Geist', 'IBM Plex Sans', system-ui, sans-serif",
        ...((tab === "hs") ? {
          fontSize: 12, lineHeight: 1.65,
          background: "#060a12",
          padding: "18px 22px",
        } : {}),
        ...((tab === "params" || tab === "shapes") ? { fontSize: 12.5, lineHeight: 1.7 } : {}),
      }}>
        {content}
      </div>
    </div>
  );
}

/* ───────────────────── SVG diagram ─────────────────────────────── */
function Diagram({ selected, onSelect }) {
  const W = 560, cx = W / 2;
  const bW = 330, sW = 250;
  const bH = 44, sH = 34;
  const gap = 6;

  const Y = { embed:30, attn:118, res1:206, ln1:260, ffn:328, res2:416, ln2:470, readout:538, unembed:596, loss:684 };

  function BRect({ id, y, w, sm }) {
    const b = BLOCKS.find(b => b.id === id);
    const sel = selected === id;
    const h = sm ? sH : bH;
    return (
      <g onClick={() => onSelect(id)} style={{ cursor: "pointer" }}>
        <rect x={cx - w/2} y={y} width={w} height={h} rx={5}
          fill={sel ? b.bg : C.card}
          stroke={sel ? b.color : C.bdr}
          strokeWidth={sel ? 2 : 1}
          style={{ transition: "all 0.15s" }} />
        <text x={cx} y={y + (sm ? 14 : 18)} textAnchor="middle"
          fill={b.color} fontSize={sm ? 11 : 12.5} fontWeight="600"
          fontFamily="'JetBrains Mono','Fira Code',monospace">
          {b.label}
        </text>
        {!sm && <text x={cx} y={y + 34} textAnchor="middle"
          fill={C.dim} fontSize={9.5}
          fontFamily="'JetBrains Mono',monospace">
          {b.sub}
        </text>}
      </g>
    );
  }

  function Arr({ y1, y2 }) {
    return <line x1={cx} y1={y1} x2={cx} y2={y2}
      stroke={C.dim} strokeWidth={1.3} markerEnd="url(#ah)" />;
  }

  function ResArc({ y1, y2, label, side = 1 }) {
    const off = side * 190;
    const my = (y1 + y2) / 2;
    return (
      <g>
        <path d={`M ${cx} ${y1} C ${cx+off} ${y1}, ${cx+off} ${y2}, ${cx} ${y2}`}
          fill="none" stroke={C.res} strokeWidth={1.4} strokeDasharray="5,4" opacity={0.5} />
        <circle cx={cx + off*0.52} cy={my} r={8} fill={C.bg} stroke={C.res} strokeWidth={1} />
        <text x={cx + off*0.52} y={my + 3.5} textAnchor="middle"
          fill={C.dim} fontSize={11} fontWeight="bold" fontFamily="monospace">+</text>
        <text x={cx + off*0.73} y={my + 3} textAnchor={side > 0 ? "start" : "end"}
          fill={C.dim} fontSize={8} fontFamily="monospace" opacity={0.5}>{label}</text>
      </g>
    );
  }

  function Dim({ x, y, text }) {
    return <text x={x} y={y} fill={C.dim} fontSize={9}
      fontFamily="'JetBrains Mono',monospace" opacity={0.5}>{text}</text>;
  }

  const svgH = Y.loss + bH + 32;
  const dX = cx + bW/2 + 12;
  const dXs = cx + sW/2 + 12;

  return (
    <svg viewBox={`0 0 ${W} ${svgH}`} width="100%" style={{ display: "block" }}>
      <defs>
        <marker id="ah" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
          <polygon points="0 0, 7 2.5, 0 5" fill={C.dim} />
        </marker>
      </defs>

      <text x={cx} y={20} textAnchor="middle" fill={C.bright} fontSize={12}
        fontFamily="'JetBrains Mono',monospace" fontWeight="700">
        Inputs:  a, b  ∈ [0..52]
      </text>

      <BRect id="embed"   y={Y.embed}   w={bW} />
      <Arr y1={Y.embed+bH+gap} y2={Y.attn-gap} />
      <BRect id="attn"    y={Y.attn}    w={bW} />
      <Arr y1={Y.attn+bH+gap}  y2={Y.res1-gap} />
      <BRect id="res1"    y={Y.res1}    w={sW} sm />
      <Arr y1={Y.res1+sH+gap}  y2={Y.ln1-gap} />
      <BRect id="ln1"     y={Y.ln1}     w={bW} />
      <Arr y1={Y.ln1+bH+gap}   y2={Y.ffn-gap} />
      <BRect id="ffn"     y={Y.ffn}     w={bW} />
      <Arr y1={Y.ffn+bH+gap}   y2={Y.res2-gap} />
      <BRect id="res2"    y={Y.res2}    w={sW} sm />
      <Arr y1={Y.res2+sH+gap}  y2={Y.ln2-gap} />
      <BRect id="ln2"     y={Y.ln2}     w={bW} />
      <Arr y1={Y.ln2+bH+gap}   y2={Y.readout-gap} />
      <BRect id="readout" y={Y.readout} w={sW} sm />
      <Arr y1={Y.readout+sH+gap} y2={Y.unembed-gap} />
      <BRect id="unembed" y={Y.unembed} w={bW} />
      <Arr y1={Y.unembed+bH+gap} y2={Y.loss-gap} />
      <BRect id="loss"    y={Y.loss}    w={bW} />

      <ResArc y1={Y.embed+bH/2} y2={Y.res1+sH/2} label="skip" side={1} />
      <ResArc y1={Y.ln1+bH/2}   y2={Y.res2+sH/2} label="skip" side={-1} />

      <Dim x={dX}  y={Y.embed+28}  text="2 × ℝ⁶⁴" />
      <Dim x={dX}  y={Y.attn+28}   text="2 × ℝ⁶⁴" />
      <Dim x={dXs} y={Y.res1+22}   text="2 × ℝ⁶⁴" />
      <Dim x={dX}  y={Y.ln1+28}    text="2 × ℝ⁶⁴" />
      <Dim x={dX}  y={Y.ffn+28}    text="2 × ℝ⁶⁴" />
      <Dim x={dXs} y={Y.res2+22}   text="2 × ℝ⁶⁴" />
      <Dim x={dX}  y={Y.ln2+28}    text="2 × ℝ⁶⁴" />
      <Dim x={dXs} y={Y.readout+22} text="ℝ⁶⁴" />
      <Dim x={dX}  y={Y.unembed+28} text="ℝ⁵³ → Δ⁵²" />
      <Dim x={dX}  y={Y.loss+28}    text="scalar" />

      <text x={cx} y={svgH - 5} textAnchor="middle" fill={C.bright} fontSize={11}
        fontFamily="'JetBrains Mono',monospace" opacity={0.5}>
        prediction:  (a + b) mod 53
      </text>
    </svg>
  );
}

/* ───────────────────── main ─────────────────────────────────────── */
export default function App() {
  const [selected, setSelected] = useState(null);
  const panelRef = useRef(null);

  const block = BLOCKS.find(b => b.id === selected) || null;

  useEffect(() => {
    if (panelRef.current && selected) panelRef.current.scrollTo({ top: 0 });
  }, [selected]);

  return (
    <div style={{
      display: "flex", flexDirection: "column",
      height: "100vh", width: "100vw",
      background: C.bg, color: C.txt, overflow: "hidden",
      fontFamily: "'Geist', 'IBM Plex Sans', system-ui, sans-serif",
    }}>
      <div style={{
        padding: "13px 22px", borderBottom: `1px solid ${C.panelBdr}`,
        display: "flex", alignItems: "baseline", gap: 14, flexShrink: 0,
      }}>
        <span style={{ fontSize: 15, fontWeight: 800, color: C.bright, letterSpacing: "-0.02em" }}>
          Modular Arithmetic Transformer
        </span>
        <span style={{ fontSize: 11, color: C.dim }}>
          1 layer · 1 head · d=64 · (a+b) mod 53
        </span>
        <span style={{
          marginLeft: "auto", fontSize: 10.5, color: C.dim,
          fontFamily: "'JetBrains Mono', monospace",
        }}>
          {TOTAL_PARAMS} params
        </span>
      </div>

      <div style={{ display: "flex", flex: 1, minHeight: 0, overflow: "hidden" }}>
        <div style={{
          flex: "0 0 52%", overflowY: "auto", padding: "14px 6px",
          borderRight: `1px solid ${C.panelBdr}`,
        }}>
          <Diagram selected={selected} onSelect={setSelected} />
        </div>
        <div ref={panelRef} style={{ flex: 1, overflowY: "auto", background: C.panel }}>
          <InfoPanel block={block} />
        </div>
      </div>

      <div style={{
        padding: "7px 22px", borderTop: `1px solid ${C.panelBdr}`,
        fontSize: 10, color: C.dim, display: "flex", gap: 20, flexShrink: 0,
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        <span>post-norm · no causal mask · AdamW + cosine LR</span>
        <span style={{ marginLeft: "auto" }}>p=53 · dModel=64 · dFF=256 · dK=64 · batch=32</span>
      </div>
    </div>
  );
}
