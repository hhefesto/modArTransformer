-- Single-head self-attention for seqLen = 2, built ENTIRELY by composing D
-- primitives — no hand-written backward pass anywhere.  Q/K/V/O are
-- `linearLayerD` projections; scores use `vdotD`+`scaleByD`; the row softmaxes
-- use `softmax2D`; the value mixes use `scaleVecD`+`vaddD`.  Gradients w.r.t.
-- all four weight matrices, both biases, and both inputs are *derived* by the
-- chain rule in D (Dual AddFun).  Forward reference: old Attention.agda:69-96
-- (≡ Main.hs:193-319 for n=2).
{-# OPTIONS --guardedness #-}
module ModArTransformer.Layers.Attention where

open import Data.Nat using (ℕ)
open import Agda.Builtin.Float using (primNatToFloat)

open import ModArTransformer.Tensor using (Float; ℝVec; ℝMat; fone; fsqrt; _f/_)
open import ModArTransformer.Cat.Objects
open import ModArTransformer.Cat.Additive
open import ModArTransformer.Cat.D
open import ModArTransformer.Cat.NumCat using (scaleByD)
open import ModArTransformer.Cat.VecPrim using (vdotD; scaleVecD; softmax2D; vaddD)
open import ModArTransformer.Cat.AdditiveTensor
open import ModArTransformer.Layers.Linear

private variable dModel dK : ℕ

-- Wq, Wk, Wv (project dModel → dK) and Wo (project dK → dModel).
AttnParams : ℕ → ℕ → Set
AttnParams dModel dK =
  LinParams dK dModel × LinParams dK dModel × LinParams dK dModel × LinParams dModel dK

attnD : D (AttnParams dModel dK × (ℝVec dModel × ℝVec dModel))
            (ℝVec dModel × ℝVec dModel)
attnD {dModel} {dK} = y0 ▵D y1
  where
    Dom : Set
    Dom = AttnParams dModel dK × (ℝVec dModel × ℝVec dModel)

    scaleDK : Float
    scaleDK = fone f/ fsqrt (primNatToFloat dK)

    -- parameter / input projections out of the domain
    ps  : D Dom (AttnParams dModel dK)
    ps  = exlD
    Wq  : D Dom (LinParams dK dModel)
    Wq  = exlD ∘D ps
    Wk  : D Dom (LinParams dK dModel)
    Wk  = (exlD ∘D exrD) ∘D ps
    Wv  : D Dom (LinParams dK dModel)
    Wv  = (exlD ∘D (exrD ∘D exrD)) ∘D ps
    Wo  : D Dom (LinParams dModel dK)
    Wo  = (exrD ∘D (exrD ∘D exrD)) ∘D ps
    x0  : D Dom (ℝVec dModel)
    x0  = exlD ∘D exrD
    x1  : D Dom (ℝVec dModel)
    x1  = exrD ∘D exrD

    -- Q, K, V projections at each position
    proj : D Dom (LinParams dK dModel) → D Dom (ℝVec dModel) → D Dom (ℝVec dK)
    proj W x = linearLayerD ∘D (W ▵D x)
    q0 = proj Wq x0 ; q1 = proj Wq x1
    k0 = proj Wk x0 ; k1 = proj Wk x1
    v0 = proj Wv x0 ; v1 = proj Wv x1

    -- scaled dot-product scores
    score : D Dom (ℝVec dK) → D Dom (ℝVec dK) → D Dom Float
    score q k = scaleByD scaleDK ∘D (vdotD ∘D (q ▵D k))

    -- row softmaxes
    w0 : D Dom (Float × Float)
    w0 = softmax2D ∘D (score q0 k0 ▵D score q0 k1)
    w1 : D Dom (Float × Float)
    w1 = softmax2D ∘D (score q1 k0 ▵D score q1 k1)

    -- value mixing  a_i = Σ_j w_ij · v_j   (values live in ℝVec dK)
    mix : D Dom Float → D Dom (ℝVec dK) → D Dom (ℝVec dK)
    mix w v = scaleVecD ∘D (w ▵D v)
    a0 : D Dom (ℝVec dK)
    a0 = vaddD ∘D (mix (exlD ∘D w0) v0 ▵D mix (exrD ∘D w0) v1)
    a1 : D Dom (ℝVec dK)
    a1 = vaddD ∘D (mix (exlD ∘D w1) v0 ▵D mix (exrD ∘D w1) v1)

    -- output projection
    y0 : D Dom (ℝVec dModel)
    y0 = linearLayerD ∘D (Wo ▵D a0)
    y1 : D Dom (ℝVec dModel)
    y1 = linearLayerD ∘D (Wo ▵D a1)
