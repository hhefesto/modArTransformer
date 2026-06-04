-- Layer normalization as a single reusable D-primitive over product params.
-- It is one mathematical operation (normalize, then scale/shift), so — like
-- `matvecD` — it carries its own closed-form derivative; layers
-- that *use* it remain pure compositions.  Params are a product (γ , β); the
-- gradient splits into (dγ , dβ) and dx.  Forward + pullback transcribe the old
-- LayerNorm.agda:42-64 (≡ Main.hs:141-159).
{-# OPTIONS --guardedness #-}
module ModArTransformer.Layers.LayerNorm where

open import Data.Nat using (ℕ)
open import Data.Product using (_,_)
open import Data.Vec.Base using (map; zipWith)
open import Agda.Builtin.Float using (primNatToFloat)

open import ModArTransformer.Tensor
open import ModArTransformer.Cat.Objects
open import ModArTransformer.Cat.Additive
open import ModArTransformer.Cat.D
open import ModArTransformer.Cat.AdditiveTensor

private variable n : ℕ

-- (γ , β) parameters.
LNParams : ℕ → Set
LNParams n = ℝVec n × ℝVec n

layerNormD : D (LNParams n × ℝVec n) (ℝVec n)
layerNormD {n} = mkD (λ p →
  let γ      = proj₁ (proj₁ p)
      β      = proj₂ (proj₁ p)
      x      = proj₂ p
      d      = primNatToFloat n
      eps    = 1.0e-5
      mean   = vsum x f/ d
      xc     = map (λ xi → xi f- mean) x
      var    = vsum (map (λ xi → xi f* xi) xc) f/ d
      invStd = fone f/ fsqrt (var f+ eps)
      xHat   = vscale invStd xc
      out    = zipWith _f+_ (zipWith _f*_ γ xHat) β

      pb : ℝVec n → LNParams n × ℝVec n
      pb dOut =
        let dGamma = zipWith _f*_ dOut xHat
            dBeta  = dOut
            dXHat  = zipWith _f*_ γ dOut
            t1     = vkonst (vsum dXHat f/ d)
            t2     = vscale (vdot dXHat xHat f/ d) xHat
            dX     = vscale invStd ((dXHat v- t1) v- t2)
        in  ((dGamma , dBeta) , dX)
  in  (out , mkDual (mkAddFun pb)))
  where open import Data.Product using (proj₁; proj₂)
        open import ModArTransformer.Cat.AddFun using (mkAddFun)
        open import ModArTransformer.Cat.Dual using (mkDual)
