{-# OPTIONS --guardedness #-}
module ModArTransformer.AD.NumPrim where

open import Agda.Builtin.Float using (Float)
open import Data.Product using (_×_; _,_)
open import ModArTransformer.Tensor
open import ModArTransformer.Additive
open import ModArTransformer.AD.AddFun
open import ModArTransformer.AD.Dual
open import ModArTransformer.AD.Core

-- ─── Numeric primitives for D (Dual AddFun) ──────────────────────────────────
-- Elliott paper p.15, NumCat instance.
-- Instantiated at k = Dual AddFun (= backprop mode).

private
  k : Set → Set → Set
  k = Dual AddFun

-- negateC: linear, derivative is itself
negateP : D k Float Float
negateP = linearD fneg (mkDual (mkAddFun fneg))

-- addC: linear in both args, derivative is id ▵ id essentially
-- As a map (a,b) → (a+b), its transpose is (s ↦ (s,s)) = dup-like
-- Here we represent it as jamAF applied via Dual:
-- Dual exl : A×B → A becomes the transpose of inl : A → A×B
-- For addC : (ℝ×ℝ → ℝ), transpose is dup : ℝ → ℝ×ℝ
addP : D k (Float × Float) Float
addP = mkD λ (a , b) →
  (a f+ b , mkDual (mkAddFun (λ s → (s , s))))

-- mulC: non-linear, product rule
-- d(a·b) = b·da + a·db, so transpose maps s ↦ (s·b, s·a)
mulP : D k (Float × Float) Float
mulP = mkD λ (a , b) →
  (a f* b , mkDual (mkAddFun (λ s → (s f* b , s f* a))))

-- scale s: linear, self-transpose (scaling is its own adjoint)
scaleP : Float → D k Float Float
scaleP s = linearD (s f*_) (mkDual (mkAddFun (s f*_)))
