{-# OPTIONS --guardedness #-}
module ModArTransformer.AD.NN where

open import Agda.Builtin.Float using (Float; primNatToFloat)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Nat     using (ℕ; suc)
open import Data.Vec.Base using (Vec; map; zipWith; replicate; tabulate)
open import Data.Fin     using (Fin)
open import Data.Bool    using (Bool; true; false; if_then_else_)
open import ModArTransformer.Tensor
open import ModArTransformer.Additive
open import ModArTransformer.AD.AddFun
open import ModArTransformer.AD.Dual
open import ModArTransformer.AD.Core

-- ─── NN-specific differentiable primitives ────────────────────────────────────
-- All typed as D (Dual AddFun) A B.
-- The pullback (unDual field) is an AddFun that maps upstream → param gradient.

private
  k : Set → Set → Set
  k = Dual AddFun

-- ── relu: y = max(0, x), dy/dx = 1 if x > 0 else 0 ─────────────────────────
reluP : D k Float Float
reluP = mkD λ x →
  let y = if x f< fzero then fzero else x
      dy = if x f< fzero then fzero else fone
  in  (y , mkDual (mkAddFun (dy f*_)))

-- ── relu applied elementwise ──────────────────────────────────────────────────
reluVecP : {n : ℕ} → D k (ℝVec n) (ℝVec n)
reluVecP = mkD λ xs →
  let indicators = map (λ x → if x f< fzero then fzero else fone) xs
      ys         = zipWith _f*_ indicators xs
  in  (ys , mkDual (mkAddFun (zipWith _f*_ indicators)))

-- ── exp, recip, sqrt (scalar, for LayerNorm internals) ───────────────────────
expP : D k Float Float
expP = mkD λ x →
  let y = fexp x
  in  (y , mkDual (mkAddFun (y f*_)))

recipP : D k Float Float
recipP = mkD λ x →
  let y   = fone f/ x
      dy  = fneg (y f* y)  -- -1/x²
  in  (y , mkDual (mkAddFun (dy f*_)))

sqrtP : D k Float Float
sqrtP = mkD λ x →
  let y  = fsqrt x
      dy = fone f/ (2.0 f* y)  -- 1/(2√x)
  in  (y , mkDual (mkAddFun (dy f*_)))

-- ── sum of a vector (linear) ─────────────────────────────────────────────────
sumVecP : {n : ℕ} → D k (ℝVec n) Float
sumVecP = mkD λ v →
  (vsum v , mkDual (mkAddFun (λ s → vkonst s)))  -- pullback: broadcast s to all positions

-- ── matrix-vector multiply: M → (M #> x) where x is fixed ───────────────────
-- dL/dM_ij = dL_i · x_j  ⟹  pullback is outer product
matvecP : {m n : ℕ} → ℝVec n → D k (ℝMat m n) (ℝVec m)
matvecP x = mkD λ M →
  (M #> x , mkDual (mkAddFun (λ dY → outer dY x)))

-- ── vector-matrix multiply (transposed): v → (Mᵀ #> v) where M is fixed ──────
-- Needed for propagating gradient back through x in a linear layer.
-- dL/dx = Mᵀ · dL/dy
tmatvecP : {m n : ℕ} → ℝMat m n → D k (ℝVec m) (ℝVec n)
tmatvecP M = mkD λ v →
  (mtr M #> v , mkDual (mkAddFun (λ dX → M #> dX)))

-- ── add bias: x → x + b where b is fixed ─────────────────────────────────────
addBiasFixedP : {n : ℕ} → ℝVec n → D k (ℝVec n) (ℝVec n)
addBiasFixedP b = mkD λ x →
  (x v+ b , mkDual (mkAddFun (λ dY → dY)))  -- identity pullback (bias is fixed here)

-- ── add bias: b → (x + b) where x is fixed ───────────────────────────────────
addBiasParamP : {n : ℕ} → ℝVec n → D k (ℝVec n) (ℝVec n)
addBiasParamP x = mkD λ b →
  (x v+ b , mkDual (mkAddFun (λ dY → dY)))  -- identity pullback (b gradient = dY)

-- ── elementwise vector multiply: v → (u ⊙ v) where u is fixed ───────────────
-- Used in LayerNorm (scale by gamma)
emulFixedP : {n : ℕ} → ℝVec n → D k (ℝVec n) (ℝVec n)
emulFixedP u = mkD λ v →
  (zipWith _f*_ u v , mkDual (mkAddFun (zipWith _f*_ u)))

-- ── dot product: fixed u, differentiable v ────────────────────────────────────
dotWithP : {n : ℕ} → ℝVec n → D k (ℝVec n) Float
dotWithP u = mkD λ v →
  (vdot u v , mkDual (mkAddFun (λ s → vscale s u)))

-- ── softmax: numerically stable, pullback = y ⊙ (dy − (y·dy)) ───────────────
-- This is the Jacobian of softmax applied to upstream dy.
-- Main.hs:72-76: softmaxBackward y dy = y * (dy - replicate(y `dot` dy))
softmaxVecP : {n : ℕ} → D k (ℝVec (suc n)) (ℝVec (suc n))
softmaxVecP = mkD λ logits →
  let m    = vmaxElement logits
      exps = map (λ x → fexp (x f- m)) logits
      s    = vsum exps
      y    = map (_f/ s) exps
      pullback dy =
        let dot_y_dy = vdot y dy
        in  zipWith _f*_ y (map (λ dyi → dyi f- dot_y_dy) dy)
  in  (y , mkDual (mkAddFun pullback))

-- ── fused softmax + cross-entropy loss ────────────────────────────────────────
-- For classification only: the gradient w.r.t. logits is (probs - oneHot(target)).
-- Avoids computing the full softmax Jacobian. Main.hs:417–421.
-- Input: logits (ℝVec p).  Target label encoded at call site via Fin p.
-- Output: scalar loss.
softmaxCEP : {n : ℕ} → Fin (suc n) → D k (ℝVec (suc n)) Float
softmaxCEP target = mkD λ logits →
  let m     = vmaxElement logits
      exps  = map (λ x → fexp (x f- m)) logits
      s     = vsum exps
      probs = map (_f/ s) exps
      loss  = fneg (flog (Data.Vec.Base.lookup probs target f+ 1.0e-12))
      -- pullback: dL/d(logits) = probs - oneHot(target)  (when upstream = 1.0)
      pullback dL =
        let oh = oneHot target
        in  vscale dL (zipWith _f-_ probs oh)
  in  (loss , mkDual (mkAddFun pullback))

-- ── LayerNorm: closed-form pullback (Main.hs:141-159) ─────────────────────────
-- Forward: xHat = (x - mean) / sqrt(var + eps); out = gamma * xHat + beta
-- We treat (gamma, beta) as fixed here and differentiate w.r.t. x.
-- The pullback matches layerNormBackward in Main.hs exactly.
layerNormXP : {n : ℕ} → ℝVec n → ℝVec n → D k (ℝVec n) (ℝVec n)
layerNormXP {n} gamma beta = mkD λ x →
  let d       = primNatToFloat n
      eps     = 1.0e-5
      mean    = vsum x f/ d
      xc      = map (_f- mean) x
      var     = vsum (map (λ xi → xi f* xi) xc) f/ d
      invStd  = fone f/ fsqrt (var f+ eps)
      xHat    = vscale invStd xc
      out     = zipWith _f+_ (zipWith _f*_ gamma xHat) beta
      -- pullback: Main.hs:141-159
      pullback : ℝVec n → ℝVec n
      pullback dOut =
        let dXHat   = zipWith _f*_ gamma dOut
            t1      = vscale invStd dXHat
            t2      = vkonst (vsum dXHat f/ d)
            t3      = vscale (vdot dXHat xHat f/ d) xHat
        in  zipWith _f+_ t1 (zipWith _f-_ (vscale (fneg fone) t2) t3)
  in  (out , mkDual (mkAddFun pullback))

-- ── LayerNorm pullback w.r.t. gamma and beta ─────────────────────────────────
-- Needed to accumulate LN parameter gradients.
-- Given xHat (from forward), dOut → (dGamma, dBeta).
layerNormGammaBetaGrad : {n : ℕ} → ℝVec n → ℝVec n → (ℝVec n × ℝVec n)
layerNormGammaBetaGrad xHat dOut =
  let dGamma = zipWith _f*_ dOut xHat
  in  dGamma , dOut
