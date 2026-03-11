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
    id: "embed", label: "Embedding", sub: "Token + Position → ℝ²ˣ⁶⁴",
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
    id: "attn", label: "Attention", sub: "Single-Head Self-Attention",
    color: C.purple, bg: C.purpleBg,
    what: `Every token is linearly projected into a Query, Key, and Value:
  Q = Wq·x + bq,   K = Wk·x + bk,   V = Wv·x + bv

Dot-product scores QKᵀ/√64 form a 2×2 attention matrix. After row-wise softmax, each token's output is a weighted sum of Values, projected through Wₒ:
  attended = softmax(QKᵀ/√dₖ) · V
  out = Wₒ · attended + bₒ

Includes residual connection and LayerNorm (post-norm). No causal mask — both tokens see each other.`,
    why: `This is the only place the two operands exchange information. The 2×2 attention matrix controls how much of operand b's representation gets mixed into operand a's (and vice versa).

With no causal mask both tokens see each other freely — this is appropriate because addition is commutative and both operands are needed equally.`,
    params: `Wq  64×64 + bq 64  =  4,160
Wk  64×64 + bk 64  =  4,160
Wv  64×64 + bv 64  =  4,160
Wₒ  64×64 + bₒ 64  =  4,160
LayerNorm γ,β      =    128
────────────────────────────
Subtotal             16,768`,
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

-- output projection + residual + layernorm
outs = map (linearForward attnWo) attended
residual = zipWith (+) embedded outs
ln1Out = map (layerNormForward ln1) residual`,
    shapes: `Q, K, V      ∈  ℝ²ˣ⁶⁴
raw scores   ∈  ℝ²ˣ²   (before softmax)
weights      ∈  ℝ²ˣ²   (after softmax, rows sum to 1)
attended     ∈  ℝ²ˣ⁶⁴
attnOut      ∈  ℝ²ˣ⁶⁴  (after Wₒ, residual, LayerNorm)`,
  },
  {
    id: "mlp", label: "MLP", sub: "64 → 256 → ReLU → 64",
    color: C.orange, bg: C.orangeBg,
    what: `Two linear layers with ReLU between them, applied independently to each token:
  hidden = ReLU(W₁ · x + b₁)      64 → 256  (expand)
  out    = W₂ · hidden + b₂       256 → 64   (compress)

The 4× expansion ratio (64 → 256) is standard. ReLU zeroes negative activations, providing the essential nonlinearity.

Includes residual connection and LayerNorm (post-norm).`,
    why: `Attention mixes information across tokens; the MLP transforms each token's blended representation through a nonlinear bottleneck. The expansion to 256 dims gives the network enough parameters and capacity to learn the complex modular-arithmetic mapping.

This is where most of the model's parameters live (33K of ~57K total). The "grokking" phenomenon likely involves these weights slowly finding Fourier-like representations of mod-53 arithmetic.`,
    params: `W₁  256×64 + b₁ 256  =  16,640
W₂   64×256 + b₂  64  =  16,448
LayerNorm γ,β        =     128
──────────────────────────────
Subtotal                33,216`,
    hs: `ffnForward FFNParams{..} x =
  let h   = linearForward ffnLinear1 x   -- 64 → 256
      act = relu h                        -- max(0, ·)
      out = linearForward ffnLinear2 act  -- 256 → 64
  in  (out, FFNCache h act)

-- with residual + layernorm
residual2 = zipWith (+) ln1Out ffnOut
ln2Out = map (layerNormForward ln2) residual2`,
    shapes: `Per token:  ℝ⁶⁴ → ℝ²⁵⁶ → ℝ⁶⁴
(the 256-dim hidden is cached for backprop)`,
  },
  {
    id: "unembed", label: "Unembed", sub: "Linear 64 → 53 + Softmax",
    color: C.pink, bg: C.pinkBg,
    what: `First, take only token 0's representation (the "answer" position).

Then a single linear layer projects from ℝ⁶⁴ to ℝ⁵³ (one logit per possible answer class). Softmax converts raw logits to a valid probability distribution over [0..52].

  readout = head tokens           (take token 0)
  logits = W · readout + b        (64 → 53)
  probs  = softmax(logits)`,
    why: `This "decodes" the model's internal 64-dim representation back into the answer space. Each of the 53 outputs corresponds to one residue class mod 53. The softmax ensures probabilities sum to 1, making the loss well-defined.

Token 0 is designated as the "answer" position — through attention, it has already gathered all the information it needs from token 1.`,
    params: `W  53×64 + b 53  =  3,445
────────────────────────────
Subtotal            3,445`,
    hs: `readout = head ln2Out  -- first token only
logits = linearForward unembed readout
probs  = softmax logits`,
    shapes: `ℝ²ˣ⁶⁴  →  ℝ⁶⁴ (token 0)  →  logits ∈ ℝ⁵³  →  probs ∈ Δ⁵²
(Δ⁵² = probability simplex, sums to 1)`,
  },
  {
    id: "result", label: "Result", sub: "argmax(probs) → (a + b) mod 53",
    color: C.green, bg: C.greenBg,
    what: `The predicted answer is simply the argmax of the probability distribution:

  prediction = argmax(probs)

During training, cross-entropy loss is computed:
  loss = −log(p_target)

where p_target is the probability assigned to the correct answer.

AdamW optimizer computes gradients via backpropagation and updates all learnable parameters in Embed, Attention, MLP, and Unembed blocks.`,
    why: `Cross-entropy penalises wrong predictions (p_target → 0 means loss → ∞).

Backpropagation: compute ∂loss/∂θ for every parameter θ by applying the chain rule backwards through the network — from loss → unembed → MLP → attention → embedding.

AdamW then uses these gradients to update parameters:
• Adaptive lr + momentum → faster, stable convergence
• Weight decay → critical for grokking; without it (wd ≈ 1e-3) the model memorizes but never generalizes`,
    params: `0 learnable parameters (objective function)

AdamW: β₁=0.9, β₂=0.999, wd=1e-3`,
    hs: `-- inference
prediction = maxIndex probs

-- training loss
crossEntropyLoss probs target =
  let p = max 1e-12 (probs ! target)
  in  negate (log p)

-- AdamW update (per parameter)
adamW m v g t θ =
  let m' = β1 * m + (1 - β1) * g
      v' = β2 * v + (1 - β2) * g * g
      mHat = m' / (1 - β1^t)
      vHat = v' / (1 - β2^t)
      θ'  = θ - lr * (mHat / (sqrt vHat + ε) + wd * θ)
  in  (θ', m', v')`,
    shapes: `probs ∈ Δ⁵²  →  prediction ∈ [0..52]
              ↓  (training)
         loss ∈ ℝ  (scalar)
              ↓  backprop
       ∇θ loss ∈ ℝ⁵⁶ᐧ⁹⁴⁹  (gradients for all params)
              ↓  AdamW
          Δθ  ∈ ℝ⁵⁶ᐧ⁹⁴⁹  (parameter updates)`,
  },
];

const TOTAL_PARAMS = "56,949";

/* param breakdown for info panel */
const PARAM_BREAKDOWN = "Embed 3,520 · Attn 16,768 · MLP 33,216 · Unembed 3,445";

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
          {PARAM_BREAKDOWN}
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
  const bW = 330;
  const bH = 54;
  const gap = 8;

  const Y = { embed: 40, attn: 130, mlp: 220, unembed: 310, result: 400 };

  function BRect({ id, y, w }) {
    const b = BLOCKS.find(b => b.id === id);
    const sel = selected === id;
    return (
      <g onClick={() => onSelect(id)} style={{ cursor: "pointer" }}>
        <rect x={cx - w/2} y={y} width={w} height={bH} rx={6}
          fill={sel ? b.bg : C.card}
          stroke={sel ? b.color : C.bdr}
          strokeWidth={sel ? 2.5 : 1.5}
          style={{ transition: "all 0.15s" }} />
        <text x={cx} y={y + 22} textAnchor="middle"
          fill={b.color} fontSize={14} fontWeight="700"
          fontFamily="'JetBrains Mono','Fira Code',monospace">
          {b.label}
        </text>
        <text x={cx} y={y + 40} textAnchor="middle"
          fill={C.dim} fontSize={10}
          fontFamily="'JetBrains Mono',monospace">
          {b.sub}
        </text>
      </g>
    );
  }

  function Arr({ y1, y2 }) {
    return <line x1={cx} y1={y1} x2={cx} y2={y2}
      stroke={C.dim} strokeWidth={1.5} markerEnd="url(#ah)" />;
  }

  function Dim({ x, y, text }) {
    return <text x={x} y={y} fill={C.dim} fontSize={10}
      fontFamily="'JetBrains Mono',monospace" opacity={0.6}>{text}</text>;
  }

  const svgH = Y.result + bH + 40;
  const dX = cx + bW/2 + 14;

  return (
    <svg viewBox={`0 0 ${W} ${svgH}`} width="100%" style={{ display: "block" }}>
      <defs>
        <marker id="ah" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill={C.dim} />
        </marker>
      </defs>

      <text x={cx} y={24} textAnchor="middle" fill={C.bright} fontSize={13}
        fontFamily="'JetBrains Mono',monospace" fontWeight="700">
        Inputs:  a, b  ∈ [0..52]
      </text>

      <BRect id="embed"   y={Y.embed}   w={bW} />
      <Arr y1={Y.embed+bH+gap} y2={Y.attn-gap} />
      <BRect id="attn"    y={Y.attn}    w={bW} />
      <Arr y1={Y.attn+bH+gap}  y2={Y.mlp-gap} />
      <BRect id="mlp"     y={Y.mlp}     w={bW} />
      <Arr y1={Y.mlp+bH+gap}   y2={Y.unembed-gap} />
      <BRect id="unembed" y={Y.unembed} w={bW} />
      <Arr y1={Y.unembed+bH+gap} y2={Y.result-gap} />
      <BRect id="result"  y={Y.result}  w={bW} />

      <Dim x={dX} y={Y.embed+32}   text="ℝ²ˣ⁶⁴" />
      <Dim x={dX} y={Y.attn+32}    text="ℝ²ˣ⁶⁴" />
      <Dim x={dX} y={Y.mlp+32}     text="ℝ²ˣ⁶⁴" />
      <Dim x={dX} y={Y.unembed+32} text="ℝ⁵³" />
      <Dim x={dX} y={Y.result+32}  text="[0..52]" />

      {/* AdamW backpropagation arrow */}
      <defs>
        <marker id="ah-back" markerWidth="8" markerHeight="6" refX="1" refY="3" orient="auto">
          <polygon points="8 0, 0 3, 8 6" fill={C.orange} />
        </marker>
      </defs>
      <path
        d={`M ${cx - bW/2 - 20} ${Y.result + bH/2}
            L ${cx - bW/2 - 20} ${Y.embed + bH/2}`}
        fill="none"
        stroke={C.orange}
        strokeWidth={2}
        strokeDasharray="6,4"
        markerEnd="url(#ah-back)"
        opacity={0.7}
      />
      <text
        transform={`translate(${cx - bW/2 - 32}, ${(Y.embed + Y.result + bH) / 2}) rotate(-90)`}
        textAnchor="middle" fill={C.orange} fontSize={10}
        fontFamily="'JetBrains Mono',monospace"
        opacity={0.8}>
        AdamW backprop
      </text>

      <text x={cx} y={svgH - 8} textAnchor="middle" fill={C.bright} fontSize={12}
        fontFamily="'JetBrains Mono',monospace" opacity={0.6}>
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
