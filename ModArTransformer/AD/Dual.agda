{-# OPTIONS --guardedness #-}
module ModArTransformer.AD.Dual where

open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import ModArTransformer.AD.AddFun

-- ─── Dual category (Elliott paper p.20, Figure 10) ───────────────────────────
--
-- Dual k a b wraps a morphism k b a (reversed arrow).
-- Backprop = D (Dual AddFun):
--   - composition reverses
--   - Cartesian ops of Dual k = Cocartesian ops of k, and vice-versa
-- This is what makes the gradient flow backwards without any tape.

record Dual (k : Set → Set → Set) (A B : Set) : Set where
  constructor mkDual
  field unDual : k B A

open Dual public

-- Category for Dual k: composition reverses the arrow direction
idDual : {k : Set → Set → Set} {A : Set}
       → (∀ {X} → k X X)
       → Dual k A A
idDual kid = mkDual kid

_∘Dual_ : {k : Set → Set → Set} {A B C : Set}
        → (∀ {X Y Z} → k Y Z → k X Y → k X Z)
        → Dual k B C → Dual k A B → Dual k A C
_∘Dual_ compK (mkDual g) (mkDual f) = mkDual (compK f g)  -- NOTE: f ∘ g, not g ∘ f

-- Cartesian for Dual k = Cocartesian for k (paper Figure 10)
exlDual : {k : Set → Set → Set} {A B : Set}
        → k A (A × B)  -- inl in k
        → Dual k (A × B) A
exlDual kinl = mkDual kinl

exrDual : {k : Set → Set → Set} {A B : Set}
        → k B (A × B)
        → Dual k (A × B) B
exrDual kinr = mkDual kinr

_▵Dual_ : {k : Set → Set → Set} {A B C : Set}
        → (∀ {X Y Z} → k X Z → k Y Z → k (X × Y) Z)  -- jam-like in k
        → Dual k A B → Dual k A C → Dual k A (B × C)
_▵Dual_ jamK (mkDual f) (mkDual g) = mkDual (jamK f g)

-- Cocartesian for Dual k = Cartesian for k
inlDual : {k : Set → Set → Set} {A B : Set}
        → k (A × B) A  -- exl in k
        → Dual k A (A × B)
inlDual kexl = mkDual kexl

inrDual : {k : Set → Set → Set} {A B : Set}
        → k (A × B) B
        → Dual k B (A × B)
inrDual kexr = mkDual kexr

jamDual : {k : Set → Set → Set} {A : Set}
        → k A (A × A)  -- dup in k
        → Dual k (A × A) A
jamDual kdup = mkDual kdup

open import Agda.Builtin.Float using (Float)

-- Scale: for Dual AddFun, scale s is self-dual (scaling is its own transpose)
scaleDual : {A : Set}
          → (Float → A → A)
          → Float → Dual AddFun A A
scaleDual scaleOp s = mkDual (mkAddFun (scaleOp s))
