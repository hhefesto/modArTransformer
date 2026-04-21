{-# OPTIONS --guardedness #-}
module ModArTransformer.AD.Grad where

open import Agda.Builtin.Float using (Float)
open import Data.Product using (_×_; _,_)
open import ModArTransformer.AD.AddFun
open import ModArTransformer.AD.Dual
open import ModArTransformer.AD.Core

-- ─── Extract the gradient from a D (Dual AddFun) scalar-valued function ──────
--
-- Elliott paper slide 31 / p.19:
-- For f : a → ℝ expressed as D (Dual AddFun) a ℝ,
-- gradient f a = (Dual.unDual f') applied to 1.0
-- where f' : Dual AddFun a ℝ = AddFun ℝ a (the transposed map).
-- Applying the transposed map to 1 yields the gradient vector.

gradient : {A : Set} → D (Dual AddFun) A Float → A → A
gradient (mkD f) a =
  let (_ , mkDual (mkAddFun f')) = f a
  in  f' 1.0
