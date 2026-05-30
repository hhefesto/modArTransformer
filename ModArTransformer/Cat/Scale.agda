-- Generic scalar multiplication over a parameter bundle, by structural
-- recursion on the product of tensor leaves.  Replaces the old per-record
-- `scaleTransformer`/`scaleAttn`/… family.
{-# OPTIONS --guardedness #-}
module ModArTransformer.Cat.Scale where

open import Data.Nat using (ℕ)
open import Data.Product using (_×_; _,_; proj₁; proj₂)

open import ModArTransformer.Tensor

private variable A B : Set

record Scale (A : Set) : Set where
  field scaleA : Float → A → A
open Scale ⦃ … ⦄ public

instance
  Scale-ℝMat : {m n : ℕ} → Scale (ℝMat m n)
  Scale-ℝMat = record { scaleA = mscale }

  Scale-ℝVec : {n : ℕ} → Scale (ℝVec n)
  Scale-ℝVec = record { scaleA = vscale }

  Scale-× : ⦃ Scale A ⦄ → ⦃ Scale B ⦄ → Scale (A × B)
  Scale-× = record { scaleA = λ s p → scaleA s (proj₁ p) , scaleA s (proj₂ p) }
