{-# OPTIONS --guardedness #-}
module ModArTransformer.Layers.Embedding where

open import Agda.Builtin.Float using (Float)
open import Data.Nat     using (ℕ)
open import Data.Fin     using (Fin)
open import Data.Product using (_×_; _,_)
open import ModArTransformer.Tensor
open import ModArTransformer.Additive
open import ModArTransformer.AD.AddFun
open import ModArTransformer.AD.Dual
open import ModArTransformer.AD.Core

-- ─── Embedding lookup as a differentiable primitive ───────────────────────────
-- Input:  embedding matrix E : ℝMat vocab dModel, index i : Fin vocab (fixed)
-- Output: row i of E, i.e. E[i]
-- Pullback: dOut → sparse matrix with dOut in row i, zero elsewhere.
-- This replaces the ST scatter-add of Main.hs:812-819.

private
  k = Dual AddFun

embedLookupD : {vocab dModel : ℕ} → Fin vocab
             → D k (ℝMat vocab dModel) (ℝVec dModel)
embedLookupD i = mkD λ E →
  let y  = mrow E i
      pb : ℝVec _ → ℝMat _ _
      pb dY = mAddRow mzero i dY  -- sparse: dY at row i, zero elsewhere
  in  (y , mkDual (mkAddFun pb))
