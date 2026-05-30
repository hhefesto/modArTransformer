-- The category of (additive) linear maps, AddFun, represented as functions.
--
-- Elliott, "The Simple Essence of AD", §3, §5: linear maps form the base
-- category whose objects are vector spaces and morphisms are linear functions.
-- It is a *biproduct* category — the categorical product × doubles as the
-- coproduct — so it carries both the cartesian operations (felix `Cartesian`)
-- and the cocartesian operations (`inlL`/`inrL`/`joinL`/`jamL`, defined here
-- using `Additive` since felix's `Cocartesian` is over a separate coproduct ⊎).
{-# OPTIONS --without-K #-}
module ModArTransformer.Cat.AddFun where

open import Data.Product using (_,_; proj₁; proj₂)
open import Data.Unit using (tt)
open import Agda.Builtin.Float using (Float; primFloatTimes)

open import ModArTransformer.Cat.Objects
open import ModArTransformer.Cat.Additive

private variable A B C : Set

record AddFun (A B : Set) : Set where
  constructor mkAddFun
  field applyL : A → B
open AddFun public

instance
  AddFun-cat : Category AddFun
  AddFun-cat = record
    { id  = mkAddFun (λ x → x)
    ; _∘_ = λ g f → mkAddFun (λ x → applyL g (applyL f x))
    }

  AddFun-cart : Cartesian AddFun
  AddFun-cart = record
    { !   = mkAddFun (λ _ → tt)
    ; _▵_ = λ f g → mkAddFun (λ x → applyL f x , applyL g x)
    ; exl = mkAddFun proj₁
    ; exr = mkAddFun proj₂
    }

-- ─── Biproduct (cocartesian-over-products) operations ──────────────────────────

zeroL : ⦃ Additive B ⦄ → AddFun A B
zeroL = mkAddFun (λ _ → zeroA)

inlL : ⦃ Additive B ⦄ → AddFun A (A × B)
inlL = mkAddFun (λ x → x , zeroA)

inrL : ⦃ Additive A ⦄ → AddFun B (A × B)
inrL = mkAddFun (λ y → zeroA , y)

joinL : ⦃ Additive C ⦄ → AddFun A C → AddFun B C → AddFun (A × B) C
joinL f g = mkAddFun (λ p → applyL f (proj₁ p) ⊕ applyL g (proj₂ p))

jamL : ⦃ Additive A ⦄ → AddFun (A × A) A
jamL = mkAddFun (λ p → proj₁ p ⊕ proj₂ p)

-- scaling by a scalar is linear
scaleL : Float → AddFun Float Float
scaleL s = mkAddFun (primFloatTimes s)
