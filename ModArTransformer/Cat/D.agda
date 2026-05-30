-- The derivative category D (Dual AddFun): reverse-mode automatic
-- differentiation à la Conal Elliott ("The Simple Essence of AD").
--
--   D A B = A → B × Dual AddFun A B
--
-- A morphism carries, at each input, both its value and its derivative as a
-- *reversed* linear map (the pullback).  Composition is the one uniform chain
-- rule.  The cartesian operations are `linearD` of the corresponding linear
-- maps in `Dual AddFun`.  No layer ever writes a backward pass: gradients are
-- *derived* by composing these instances.
{-# OPTIONS --without-K #-}
module ModArTransformer.Cat.D where

open import Data.Product using (_,_; proj₁; proj₂)

open import ModArTransformer.Cat.Objects
open import ModArTransformer.Cat.Additive
open import ModArTransformer.Cat.AddFun
open import ModArTransformer.Cat.Dual

private variable A B C E : Set

record D (A B : Set) : Set where
  constructor mkD
  field runD : A → B × Dual A B
open D public

-- A linear map is its own derivative (Elliott, Thm 3): pair the forward
-- function with the (constant) reversed linear map.
linearD : (A → B) → Dual A B → D A B
linearD f f' = mkD (λ a → (f a , f'))

idD : D A A
idD = linearD (λ x → x) idDu

infixr 9 _∘D_
_∘D_ : D B C → D A B → D A C
g ∘D f = mkD (λ a →
  let (b , f') = runD f a
      (c , g') = runD g b
  in  (c , g' ∘Du f'))

-- Cartesian-style combinators (defined directly; felix's `Cartesian` is not
-- usable here because its terminal map `!` lacks the `Additive` evidence the
-- dual zero-map needs — see Cat.Objects / Cat.Dual).
exlD : ⦃ Additive B ⦄ → D (A × B) A
exlD = linearD proj₁ exlDu

exrD : ⦃ Additive A ⦄ → D (A × B) B
exrD = linearD proj₂ exrDu

infixr 7 _▵D_
_▵D_ : ⦃ Additive A ⦄ → D A C → D A E → D A (C × E)
f ▵D g = mkD (λ a →
  let (c , f') = runD f a
      (e , g') = runD g a
  in  ((c , e) , f' ▵Du g'))

dupD : ⦃ Additive A ⦄ → D A (A × A)
dupD = idD ▵D idD

-- Parallel product and the usual derived combinators.
infixr 7 _⊗D_
_⊗D_ : ⦃ Additive A ⦄ → ⦃ Additive B ⦄ → D A C → D B E → D (A × B) (C × E)
f ⊗D g = (f ∘D exlD) ▵D (g ∘D exrD)

firstD : ⦃ Additive A ⦄ → ⦃ Additive B ⦄ → D A C → D (A × B) (C × B)
firstD f = f ⊗D idD

secondD : ⦃ Additive A ⦄ → ⦃ Additive B ⦄ → D B C → D (A × B) (A × C)
secondD g = idD ⊗D g
