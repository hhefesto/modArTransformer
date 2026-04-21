{-# OPTIONS --guardedness #-}
module ModArTransformer.AD.Core where

open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import ModArTransformer.AD.AddFun
open import ModArTransformer.Additive

-- ─── Generalized differentiable function (Elliott paper p.15, Figure 6) ──────
--
-- D k a b wraps: a → b × (a `k` b)
-- where k is any category of linear maps.
-- Plugging in Dual AddFun gives reverse-mode / backprop.

record D (k : Set → Set → Set) (A B : Set) : Set where
  constructor mkD
  field run : A → B × k A B

open D public

-- Smart constructor for linear functions:
-- a linear map's derivative is itself (Theorem 3 in the paper).
linearD : {k : Set → Set → Set} {A B : Set}
        → (A → B) → k A B → D k A B
linearD f f' = mkD (λ a → (f a , f'))

-- ─── Category instance for D ──────────────────────────────────────────────────
-- This IS the chain rule (paper p.7 / Figure 6):
-- compose the primal results, compose the linear-map derivatives.

idD : {k : Set → Set → Set} {A : Set}
    → (∀ {X} → k X X)  -- id in the underlying category
    → D k A A
idD kid = linearD (λ a → a) kid

_∘D_ : {k : Set → Set → Set} {A B C : Set}
     → (∀ {X Y Z} → k Y Z → k X Y → k X Z)  -- compose in k
     → D k B C → D k A B → D k A C
_∘D_ compK (mkD g) (mkD f) = mkD λ a →
  let (b , f') = f a
      (c , g') = g b
  in  (c , compK g' f')

-- ─── Cartesian instance for D ─────────────────────────────────────────────────

exlD : {k : Set → Set → Set} {A B : Set}
     → k (A × B) A  -- exl in k
     → D k (A × B) A
exlD kexl = linearD proj₁ kexl

exrD : {k : Set → Set → Set} {A B : Set}
     → k (A × B) B
     → D k (A × B) B
exrD kexr = linearD proj₂ kexr

_▵D_ : {k : Set → Set → Set} {A B C : Set}
     → (∀ {X Y Z} → k X Y → k X Z → k X (Y × Z))  -- fork in k
     → D k A B → D k A C → D k A (B × C)
_▵D_ forkK (mkD f) (mkD g) = mkD λ a →
  let (b , f') = f a
      (c , g') = g a
  in  ((b , c) , forkK f' g')

-- ─── Cocartesian instance for D ───────────────────────────────────────────────
-- (needed so the Dual instances work; also for gradient accumulation)

inlD : {k : Set → Set → Set} {A B : Set}
     → ⦃ Additive B ⦄
     → k A (A × B)
     → D k A (A × B)
inlD kinl = linearD (λ a → (a , zero)) kinl

inrD : {k : Set → Set → Set} {A B : Set}
     → ⦃ Additive A ⦄
     → k B (A × B)
     → D k B (A × B)
inrD kinr = linearD (λ b → (zero , b)) kinr

open import ModArTransformer.Additive using (_⊕_)

jamD : {k : Set → Set → Set} {A : Set}
     → ⦃ Additive A ⦄
     → k (A × A) A
     → D k (A × A) A
jamD kjam = mkD (λ (a₁ , a₂) → (a₁ ⊕ a₂ , kjam))
