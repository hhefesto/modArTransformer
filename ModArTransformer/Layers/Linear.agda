{-# OPTIONS --guardedness #-}
module ModArTransformer.Layers.Linear where

open import Agda.Builtin.Float using (Float)
open import Data.Nat     using (ℕ)
open import Data.Product using (_×_; _,_)
open import ModArTransformer.Tensor
open import ModArTransformer.Additive
open import ModArTransformer.AD.AddFun
open import ModArTransformer.AD.Dual
open import ModArTransformer.AD.Core

-- ─── Linear layer ─────────────────────────────────────────────────────────────
-- Params: weight W : ℝMat out inp, bias b : ℝVec out.
-- Forward: x → W #> x + b.
-- Differentiable w.r.t. ((W, b), x) jointly.

record LinearParams (out inp : ℕ) : Set where
  constructor mkLinear
  field
    linW : ℝMat out inp
    linB : ℝVec out

open LinearParams public

-- Zero-gradient linear params
zeroLinear : {out inp : ℕ} → LinearParams out inp
zeroLinear = mkLinear mzero vzero

addLinear : {out inp : ℕ} → LinearParams out inp → LinearParams out inp → LinearParams out inp
addLinear p q = mkLinear (linW p m+ linW q) (linB p v+ linB q)

scaleLinear : {out inp : ℕ} → Float → LinearParams out inp → LinearParams out inp
scaleLinear s p = mkLinear (mscale s (linW p)) (vscale s (linB p))

instance
  additiveLinear : {out inp : ℕ} → Additive (LinearParams out inp)
  additiveLinear = record { zero = zeroLinear ; _⊕_ = addLinear }

-- ─── D (Dual AddFun) over (params × input) ────────────────────────────────────
-- The full differentiable linear layer: diff. w.r.t. both params and input.
-- Pullback decomposes: dY → (dW = outer dY x, dB = dY, dX = Wᵀ · dY)
-- Matching Main.hs:101-106.

private
  k = Dual AddFun

linearLayerD : {out inp : ℕ}
        → D k (LinearParams out inp × ℝVec inp) (ℝVec out)
linearLayerD = mkD (λ (p , x) →
  let y  = linW p #> x v+ linB p
      pb dY =
        let dW = outer dY x
            dB = dY
            dX = mtr (linW p) #> dY
        in  (mkLinear dW dB , dX)
  in  (y , mkDual (mkAddFun pb)))

-- Convenience: linear forward with only-params differentiation (input fixed).
linearParamsD : {out inp : ℕ} → ℝVec inp → D k (LinearParams out inp) (ℝVec out)
linearParamsD x = mkD (λ p →
  let y  = linW p #> x v+ linB p
      pb dY = mkLinear (outer dY x) dY
  in  (y , mkDual (mkAddFun pb)))

-- Convenience: linear forward with only-input differentiation (params fixed).
linearInputD : {out inp : ℕ} → LinearParams out inp → D k (ℝVec inp) (ℝVec out)
linearInputD p = mkD (λ x →
  let y      = linW p #> x v+ linB p
      pb dY  = mtr (linW p) #> dY
  in  (y , mkDual (mkAddFun pb)))
