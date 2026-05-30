-- Additive instances for the tensor objects (vectors and matrices), so that
-- they may serve as objects of the biproduct/derivative categories.
{-# OPTIONS --guardedness #-}
module ModArTransformer.Cat.AdditiveTensor where

open import Data.Nat using (ℕ)
open import ModArTransformer.Tensor
open import ModArTransformer.Cat.Additive

instance
  Additive-ℝVec : {n : ℕ} → Additive (ℝVec n)
  Additive-ℝVec = record { zeroA = vzero ; _⊕_ = _v+_ }

  Additive-ℝMat : {m n : ℕ} → Additive (ℝMat m n)
  Additive-ℝMat = record { zeroA = mzero ; _⊕_ = _m+_ }
