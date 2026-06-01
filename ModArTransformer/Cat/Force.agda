-- Generic "force to a scalar" over a parameter bundle, used to defeat thunk
-- buildup during training (paired with the `seqBy`/`foldl'` FFI in Train).
-- Replaces the old `forceXxx` tower; one fold over the tensor leaves.
{-# OPTIONS --guardedness #-}
module ModArTransformer.Cat.Force where

open import Data.Nat using (ℕ)
open import Data.Product using (proj₁; proj₂)
open import Data.Vec.Base using (map)

open import ModArTransformer.Tensor
open import ModArTransformer.Cat.Objects using (_×_)

private variable A B : Set

record Forceable (A : Set) : Set where
  field force : A → Float
open Forceable ⦃ … ⦄ public

instance
  Forceable-Float : Forceable Float
  Forceable-Float = record { force = λ x → x }

  Forceable-ℝVec : {n : ℕ} → Forceable (ℝVec n)
  Forceable-ℝVec = record { force = vsum }

  Forceable-ℝMat : {m n : ℕ} → Forceable (ℝMat m n)
  Forceable-ℝMat = record { force = λ M → vsum (map vsum M) }

  Forceable-× : ⦃ Forceable A ⦄ → ⦃ Forceable B ⦄ → Forceable (A × B)
  Forceable-× = record { force = λ p → force (proj₁ p) f+ force (proj₂ p) }
