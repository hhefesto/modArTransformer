{-# OPTIONS --guardedness #-}
module ModArTransformer.AD.AddFun where

open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import ModArTransformer.Additive

-- ─── The additive-function category (Elliott paper p.10, Figure 1) ───────────
--
-- AddFun a b wraps an additive (linear) function a → b.
-- Plugging this into D_k gives forward-mode AD.

record AddFun (A B : Set) : Set where
  constructor mkAddFun
  field apply : A → B

open AddFun public

-- Category
idAF : {A : Set} → AddFun A A
idAF = mkAddFun (λ a → a)

_∘AF_ : {A B C : Set} → AddFun B C → AddFun A B → AddFun A C
(mkAddFun g) ∘AF (mkAddFun f) = mkAddFun (λ a → g (f a))

-- Cartesian
exlAF : {A B : Set} → AddFun (A × B) A
exlAF = mkAddFun proj₁

exrAF : {A B : Set} → AddFun (A × B) B
exrAF = mkAddFun proj₂

_▵AF_ : {A B C : Set} → AddFun A B → AddFun A C → AddFun A (B × C)
(mkAddFun f) ▵AF (mkAddFun g) = mkAddFun (λ a → (f a , g a))

-- Cocartesian (Elliott paper p.10: inlF a = (a,0), inrF b = (0,b), jamF (a,b) = a+b)
inlAF : {A B : Set} → ⦃ Additive B ⦄ → AddFun A (A × B)
inlAF = mkAddFun (λ a → (a , zero))

inrAF : {A B : Set} → ⦃ Additive A ⦄ → AddFun B (A × B)
inrAF = mkAddFun (λ b → (zero , b))

jamAF : {A : Set} → ⦃ Additive A ⦄ → AddFun (A × A) A
jamAF = mkAddFun (λ (a , b) → a ⊕ b)

-- Scale: multiply by a scalar constant (used in NumCat / NN primitives)
open import Agda.Builtin.Float using (Float)

scaleAF : {A : Set} → ⦃ Additive A ⦄ → (Float → A → A) → Float → AddFun A A
scaleAF scaleOp s = mkAddFun (scaleOp s)
