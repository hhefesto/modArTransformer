-- Gradient descent à la Conal Elliott: extract the gradient of a scalar-valued
-- morphism in D (Dual AddFun) by running it and feeding the unit cotangent 1
-- into the reversed linear map (the pullback).  `eval` runs the same morphism
-- forward-only (for inference).  One definition each; works for every morphism.
{-# OPTIONS --without-K #-}
module ModArTransformer.Cat.Grad where

open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Agda.Builtin.Float using (Float)

open import ModArTransformer.Cat.Additive
open import ModArTransformer.Cat.AddFun
open import ModArTransformer.Cat.Dual
open import ModArTransformer.Cat.D

private variable A B : Set

-- | Forward-only evaluation: the value of the morphism, ignoring derivatives.
eval : D A B → A → B
eval f a = proj₁ (runD f a)

-- | The gradient of a scalar-valued morphism at a point: the pullback of 1.0.
gradient : D A Float → A → A
gradient f a = applyL (unDual (proj₂ (runD f a))) 1.0

-- | Loss and gradient together from a single forward run (one `runD`): the
--   value, plus the pullback of 1.0.  Used by the training loop.
gradAndLoss : D A Float → A → A × Float
gradAndLoss f a = let r = runD f a in (applyL (unDual (proj₂ r)) 1.0 , proj₁ r)
