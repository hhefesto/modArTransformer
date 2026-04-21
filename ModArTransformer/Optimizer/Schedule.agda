{-# OPTIONS --guardedness #-}
module ModArTransformer.Optimizer.Schedule where

open import Agda.Builtin.Float using (Float; primNatToFloat; primFloatCos)
open import Data.Nat using (ℕ; _<ᵇ_)
open import Data.Bool using (Bool; true; false; if_then_else_)
open import ModArTransformer.Tensor

-- ─── Cosine LR schedule with linear warmup (Main.hs:642-660) ──────────────────
-- step     : current global step
-- warmup   : number of warmup steps
-- baseLR   : peak learning rate
-- minLR    : minimum learning rate (end of cosine)
-- totalSteps: total training steps

lrWarmupCosine : ℕ → ℕ → Float → Float → ℕ → Float
lrWarmupCosine step warmup baseLR minLR totalSteps =
  let stepF    = primNatToFloat step
      warmupF  = primNatToFloat warmup
      totalF   = primNatToFloat totalSteps
      pi       = 3.141592653589793
  in
  if step <ᵇ warmup
    then minLR f+ (baseLR f- minLR) f* (stepF f/ warmupF)
    else
      let progress = (stepF f- warmupF) f/ (totalF f- warmupF)
          cosVal   = primFloatCos (pi f* progress)
      in  minLR f+ (baseLR f- minLR) f* (fone f+ cosVal) f/ 2.0
