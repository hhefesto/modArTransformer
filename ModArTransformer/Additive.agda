{-# OPTIONS --guardedness #-}
module ModArTransformer.Additive where

open import Agda.Builtin.Float using (Float)
open import Data.Vec.Base      using (Vec; zipWith; replicate)
open import Data.Nat           using (ℕ)
open import Data.Product       using (_×_; _,_)
open import ModArTransformer.Tensor
  using (_f+_; fzero; _v+_; vzero; _m+_; mzero; ℝVec; ℝMat)

-- ─── Additive typeclass ────────────────────────────────────────────────────────
-- Matches Elliott's paper p.10: objects with zero and (+).
-- Used by AddFun instances (inlF, inrF, jamF all need zero).

record Additive (A : Set) : Set where
  field
    zero : A
    _⊕_  : A → A → A

open Additive ⦃ … ⦄ public

instance
  additiveFloat : Additive Float
  additiveFloat = record { zero = fzero ; _⊕_ = _f+_ }

  additiveVec : {n : ℕ} → Additive (ℝVec n)
  additiveVec = record { zero = vzero ; _⊕_ = _v+_ }

  additiveMat : {m n : ℕ} → Additive (ℝMat m n)
  additiveMat = record { zero = mzero ; _⊕_ = _m+_ }

  additivePair : {A B : Set} → ⦃ Additive A ⦄ → ⦃ Additive B ⦄ → Additive (A × B)
  additivePair = record
    { zero = zero , zero
    ; _⊕_  = λ (a₁ , b₁) (a₂ , b₂) → (a₁ ⊕ a₂) , (b₁ ⊕ b₂)
    }
