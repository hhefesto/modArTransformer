{-# OPTIONS --guardedness #-}
module ModArTransformer.Layers.Transformer where

open import Agda.Builtin.Float using (Float; primNatToFloat)
open import Data.Nat     using (ℕ; suc)
open import Data.Fin     using (Fin) renaming (zero to fzero'; suc to fsuc')
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Vec.Base using (Vec; _∷_; []; lookup; map; zipWith)
open import ModArTransformer.Tensor
open import ModArTransformer.Additive
open import ModArTransformer.AD.AddFun
open import ModArTransformer.AD.Dual
open import ModArTransformer.AD.Core
open import ModArTransformer.Layers.Linear
open import ModArTransformer.Layers.LayerNorm
open import ModArTransformer.Layers.FFN
open import ModArTransformer.Layers.Attention
open import ModArTransformer.Layers.Embedding

-- ─── Full transformer parameter bundle ────────────────────────────────────────

record TransformerParams (p dModel dFF dK : ℕ) : Set where
  constructor mkTransformer
  field
    tokEmbed : ℝMat p dModel
    posEmbed : ℝMat 2 dModel
    attnP    : AttentionParams dModel dK
    ln1P     : LayerNormParams dModel
    ffnP     : FFNParams dModel dFF
    ln2P     : LayerNormParams dModel
    unembed  : LinearParams p dModel

open TransformerParams public

zeroTransformer : {p dModel dFF dK : ℕ} → TransformerParams p dModel dFF dK
zeroTransformer = mkTransformer mzero mzero zeroAttn zeroLN zeroFFN zeroLN zeroLinear

addTransformer : {p dModel dFF dK : ℕ}
               → TransformerParams p dModel dFF dK
               → TransformerParams p dModel dFF dK
               → TransformerParams p dModel dFF dK
addTransformer a b = mkTransformer
  (tokEmbed a m+ tokEmbed b) (posEmbed a m+ posEmbed b)
  (addAttn  (attnP a) (attnP b))
  (addLN    (ln1P  a) (ln1P  b))
  (addFFN   (ffnP  a) (ffnP  b))
  (addLN    (ln2P  a) (ln2P  b))
  (addLinear (unembed a) (unembed b))

scaleTransformer : {p dModel dFF dK : ℕ} → Float
                 → TransformerParams p dModel dFF dK
                 → TransformerParams p dModel dFF dK
scaleTransformer s a = mkTransformer
  (mscale s (tokEmbed a)) (mscale s (posEmbed a))
  (scaleAttn s (attnP a)) (scaleLN s (ln1P a))
  (scaleFFN  s (ffnP  a)) (scaleLN s (ln2P a))
  (scaleLinear s (unembed a))

instance
  additiveTransformer : {p dModel dFF dK : ℕ}
                      → Additive (TransformerParams p dModel dFF dK)
  additiveTransformer = record { zero = zeroTransformer ; _⊕_ = addTransformer }

-- ─── D (Dual AddFun) over TransformerParams ────────────────────────────────────
-- For fixed example (tokA, tokB, target), produce the forward loss and a
-- pullback: Float → TransformerParams (the combined gradient record).
-- This is the Conal chain rule: the pullback closes over all forward values
-- and runs the backward pass without an explicit tape or mutable state.
-- Replaces forwardTrain (Main.hs:367-406) + backward (Main.hs:492-570).

private
  k = Dual AddFun

  applyPB : {A B : Set} → Dual AddFun A B → B → A
  applyPB (mkDual (mkAddFun f)) = f

transformerD : {p dModel dFF dK : ℕ}
             → Fin (suc p) → Fin (suc p) → Fin (suc p)
             → D k (TransformerParams (suc p) dModel dFF dK) Float
transformerD {p} {dModel} {dFF} {dK} tokA tokB target = mkD (λ params →
  let
    -- ── 1. Embedding (Main.hs:347-353) ──────────────────────────────────────
    e0 = mrow (tokEmbed params) tokA v+ mrow (posEmbed params) fzero'
    e1 = mrow (tokEmbed params) tokB v+ mrow (posEmbed params) (fsuc' fzero')

    -- ── 2. Attention + residual (Main.hs:354-360) ────────────────────────────
    ((a0 , a1) , attnPB) = run attnD (attnP params , (e0 , e1))
    r10 = e0 v+ a0
    r11 = e1 v+ a1

    -- ── 3. LayerNorm 1 (Main.hs:361-362) ─────────────────────────────────────
    (n10 , ln1PB0) = run layerNormD (ln1P params , r10)
    (n11 , ln1PB1) = run layerNormD (ln1P params , r11)

    -- ── 4. FFN + residual (Main.hs:363-367) ──────────────────────────────────
    (f0  , ffnPB0) = run ffnD (ffnP params , n10)
    (f1  , ffnPB1) = run ffnD (ffnP params , n11)
    r20 = n10 v+ f0
    r21 = n11 v+ f1

    -- ── 5. LayerNorm 2 (Main.hs:368-369) ─────────────────────────────────────
    (o0  , ln2PB0) = run layerNormD (ln2P params , r20)

    -- ── 6. Readout, unembed, loss (Main.hs:370-406) ──────────────────────────
    logits = linW (unembed params) #> o0 v+ linB (unembed params)
    m      = vmaxElement logits
    es     = map (λ x → fexp (x f- m)) logits
    s      = vsum es
    probs  = map (_f/ s) es
    oh     = oneHot target
    loss   = fneg (flog (lookup probs target f+ 1.0e-12))

    -- ── 7. Pullback (chain rule, replaces Main.hs:492-570) ───────────────────
    pullback : Float → TransformerParams (suc p) dModel dFF dK
    pullback dL =
      -- CE gradient: dL/d(logits) = dL · (probs - oneHot(target))
      let dLogits = vscale dL (zipWith _f-_ probs oh)

          -- backward unembed (Main.hs:505)
          dUnembedW = outer dLogits o0
          dUnembedB = dLogits
          dO0       = mtr (linW (unembed params)) #> dLogits

          -- backward LN2 for pos 0 (Main.hs:507-517)
          (dLn2P , dR20)  = applyPB ln2PB0 dO0

          -- backward FFN pos 0 and pos 1 (pos 1 has zero upstream from loss)
          (dFfnP0 , dN10f) = applyPB ffnPB0 dR20
          (dFfnP1 , dN11f) = applyPB ffnPB1 vzero
          dFfn = addFFN dFfnP0 dFfnP1

          -- residual adds: d flows through both FFN and identity path
          dN10  = dN10f v+ dR20   -- LN1-pos0 receives grad from both FFN and residual
          dN11  = dN11f            -- pos1: no upstream from loss readout

          -- backward LN1 (Main.hs:534-542)
          (dLn1P0 , dR10) = applyPB ln1PB0 dN10
          (dLn1P1 , dR11) = applyPB ln1PB1 dN11
          dLn1 = addLN dLn1P0 dLn1P1

          -- backward attention + residual (Main.hs:548-553)
          (dAttnP , (dE0a , dE1a)) = applyPB attnPB (dR10 , dR11)
          -- residual: embeddings get gradient from both the attn output and identity skip
          dE0 = dR10 v+ dE0a
          dE1 = dR11 v+ dE1a

          -- backward embedding scatter (replaces ST mutation Main.hs:812-819)
          dTokEmb = mAddRow (mAddRow mzero tokA dE0) tokB dE1
          dPosEmb = mAddRow (mAddRow mzero fzero' dE0) (fsuc' fzero') dE1

      in  mkTransformer dTokEmb dPosEmb dAttnP dLn1 dFfn dLn2P
                        (mkLinear dUnembedW dUnembedB)

  in  (loss , mkDual (mkAddFun pullback)))
