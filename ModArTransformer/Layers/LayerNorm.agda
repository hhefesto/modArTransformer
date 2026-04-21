{-# OPTIONS --guardedness #-}
module ModArTransformer.Layers.LayerNorm where

open import Agda.Builtin.Float using (Float; primNatToFloat)
open import Data.Nat     using (ℕ; suc)
open import Data.Product using (_×_; _,_)
open import Data.Vec.Base using (Vec; map; zipWith)
open import ModArTransformer.Tensor
open import ModArTransformer.Additive
open import ModArTransformer.AD.AddFun
open import ModArTransformer.AD.Dual
open import ModArTransformer.AD.Core

record LayerNormParams (n : ℕ) : Set where
  constructor mkLN
  field
    lnGamma : ℝVec n
    lnBeta  : ℝVec n

open LayerNormParams public

zeroLN : {n : ℕ} → LayerNormParams n
zeroLN = mkLN vzero vzero

addLN : {n : ℕ} → LayerNormParams n → LayerNormParams n → LayerNormParams n
addLN p q = mkLN (lnGamma p v+ lnGamma q) (lnBeta p v+ lnBeta q)

scaleLN : {n : ℕ} → Float → LayerNormParams n → LayerNormParams n
scaleLN s p = mkLN (vscale s (lnGamma p)) (vscale s (lnBeta p))

instance
  additiveLN : {n : ℕ} → Additive (LayerNormParams n)
  additiveLN = record { zero = zeroLN ; _⊕_ = addLN }

-- ─── Differentiable LayerNorm w.r.t. (params × input) ────────────────────────
-- Pullback matches Main.hs:141-159.

private
  k = Dual AddFun

layerNormD : {n : ℕ} → D k (LayerNormParams n × ℝVec n) (ℝVec n)
layerNormD {n} = mkD λ (p , x) →
  let d      = primNatToFloat n
      eps    = 1.0e-5
      mean   = vsum x f/ d
      xc     = map (_f- mean) x
      var    = vsum (map (λ xi → xi f* xi) xc) f/ d
      invStd = fone f/ fsqrt (var f+ eps)
      xHat   = vscale invStd xc
      out    = zipWith _f+_ (zipWith _f*_ (lnGamma p) xHat) (lnBeta p)

      pb : ℝVec n → LayerNormParams n × ℝVec n
      pb dOut =
        -- gradient w.r.t. gamma and beta (Main.hs conceptually):
        let dGamma = zipWith _f*_ dOut xHat
            dBeta  = dOut
            -- gradient w.r.t. x (Main.hs:141-159):
            dXHat  = zipWith _f*_ (lnGamma p) dOut
            t1     = vscale invStd dXHat
            t2     = vkonst (vsum dXHat f/ d)
            t3     = vscale (vdot dXHat xHat f/ d) xHat
            dX     = zipWith _f-_ (zipWith _f-_ t1 t2) t3
        in  (mkLN dGamma dBeta , dX)
  in  (out , mkDual (mkAddFun pb))
