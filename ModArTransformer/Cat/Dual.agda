-- The dual category Dual AddFun: linear maps with their arrows reversed.
--
-- Elliott, "The Simple Essence of AD", §4.3 & Fig. 10: representing a linear
-- map `a ⊸ b` by its transpose `b ⊸ a` gives reverse-mode AD / backpropagation.
-- The cartesian operations of `Dual k` are the *cocartesian* operations of `k`
-- (exl ↦ inl, fork ▵ ↦ join, dup ↦ jam), which is why `Dual AddFun` is where
-- gradients flow backwards.  We instantiate `k = AddFun` directly.
{-# OPTIONS --without-K #-}
module ModArTransformer.Cat.Dual where

open import ModArTransformer.Cat.Objects
open import ModArTransformer.Cat.Additive
open import ModArTransformer.Cat.AddFun

private variable A B C E : Set

record Dual (A B : Set) : Set where
  constructor mkDual
  field unDual : AddFun B A         -- the transposed linear map
open Dual public

-- Category: identity and (arrow-reversing) composition.
idDu : Dual A A
idDu = mkDual id

infixr 9 _∘Du_
_∘Du_ : Dual B C → Dual A B → Dual A C
g ∘Du f = mkDual (unDual f ∘ unDual g)

-- Cartesian-over-products, obtained from AddFun's cocartesian operations.
exlDu : ⦃ Additive B ⦄ → Dual (A × B) A
exlDu = mkDual inlL

exrDu : ⦃ Additive A ⦄ → Dual (A × B) B
exrDu = mkDual inrL

infixr 7 _▵Du_
_▵Du_ : ⦃ Additive A ⦄ → Dual A C → Dual A E → Dual A (C × E)
f ▵Du g = mkDual (joinL (unDual f) (unDual g))

dupDu : ⦃ Additive A ⦄ → Dual A (A × A)
dupDu = idDu ▵Du idDu
