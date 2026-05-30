-- Additive structure: the commutative monoids over which linear maps form a
-- *biproduct* category (Elliott, "The Simple Essence of Automatic
-- Differentiation", §5).  `zeroA` and `_⊕_` are what make the cocartesian
-- operations (inl, inr, join, jam) — and hence reverse-mode gradients —
-- definable on linear maps.  (Field name `zeroA`, not `zero`, to avoid clashing
-- with `Data.Nat.zero` / `Data.Fin.zero`.)
{-# OPTIONS --without-K #-}
module ModArTransformer.Cat.Additive where

open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Unit using (⊤; tt)
open import Agda.Builtin.Float using (Float; primFloatPlus)

private variable A B : Set

record Additive (A : Set) : Set where
  field
    zeroA : A
    _⊕_   : A → A → A

open Additive ⦃ … ⦄ public

instance
  Additive-Float : Additive Float
  Additive-Float = record { zeroA = 0.0 ; _⊕_ = primFloatPlus }

  Additive-⊤ : Additive ⊤
  Additive-⊤ = record { zeroA = tt ; _⊕_ = λ _ _ → tt }

  Additive-× : ⦃ Additive A ⦄ → ⦃ Additive B ⦄ → Additive (A × B)
  Additive-× = record
    { zeroA = zeroA , zeroA
    ; _⊕_  = λ p q → (proj₁ p ⊕ proj₁ q) , (proj₂ p ⊕ proj₂ q)
    }
