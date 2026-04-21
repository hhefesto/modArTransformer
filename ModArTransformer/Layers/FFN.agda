{-# OPTIONS --guardedness #-}
module ModArTransformer.Layers.FFN where

open import Agda.Builtin.Float using (Float)
open import Data.Nat     using (ℕ)
open import Data.Product using (_×_; _,_)
open import Data.Bool    using (Bool; true; false; if_then_else_)
open import Data.Vec.Base using (map; zipWith)
open import ModArTransformer.Tensor
open import ModArTransformer.Additive
open import ModArTransformer.AD.AddFun
open import ModArTransformer.AD.Dual
open import ModArTransformer.AD.Core
open import ModArTransformer.Layers.Linear

-- ─── Two-layer MLP with ReLU (Main.hs:165-187) ────────────────────────────────
-- Forward: x → linear2(relu(linear1(x)))
-- Pullback: chain of linear backwards separated by relu' indicator.

record FFNParams (dModel dFF : ℕ) : Set where
  constructor mkFFN
  field
    ffnLinear1 : LinearParams dFF   dModel
    ffnLinear2 : LinearParams dModel dFF

open FFNParams public

zeroFFN : {dModel dFF : ℕ} → FFNParams dModel dFF
zeroFFN = mkFFN zeroLinear zeroLinear

addFFN : {dModel dFF : ℕ} → FFNParams dModel dFF → FFNParams dModel dFF → FFNParams dModel dFF
addFFN p q = mkFFN (addLinear (ffnLinear1 p) (ffnLinear1 q))
                   (addLinear (ffnLinear2 p) (ffnLinear2 q))

scaleFFN : {dModel dFF : ℕ} → Float → FFNParams dModel dFF → FFNParams dModel dFF
scaleFFN s p = mkFFN (scaleLinear s (ffnLinear1 p)) (scaleLinear s (ffnLinear2 p))

instance
  additiveFFN : {dModel dFF : ℕ} → Additive (FFNParams dModel dFF)
  additiveFFN = record { zero = zeroFFN ; _⊕_ = addFFN }

-- ─── Differentiable FFN w.r.t. (params × input) ───────────────────────────────
-- Inline forward + pullback, matching Main.hs:182-187.

private
  k = Dual AddFun

ffnD : {dModel dFF : ℕ}
     → D k (FFNParams dModel dFF × ℝVec dModel) (ℝVec dModel)
ffnD = mkD (λ (p , x) →
  -- Forward pass (Main.hs:175-180)
  let h0      = linW (ffnLinear1 p) #> x v+ linB (ffnLinear1 p)
      relu'   = map (λ v → if v f< fzero then fzero else fone) h0
      hidden  = map (λ v → if v f< fzero then fzero else v)  h0
      out     = linW (ffnLinear2 p) #> hidden v+ linB (ffnLinear2 p)
      pb dOut =
        -- backward through linear2
        let dHidden  = mtr (linW (ffnLinear2 p)) #> dOut
            dW2      = outer dOut hidden
            dB2      = dOut
            -- backward through relu
            dH0      = zipWith _f*_ relu' dHidden
            -- backward through linear1
            dX       = mtr (linW (ffnLinear1 p)) #> dH0
            dW1      = outer dH0 x
            dB1      = dH0
        in  (mkFFN (mkLinear dW1 dB1) (mkLinear dW2 dB2) , dX)
  in  (out , mkDual (mkAddFun pb)))
