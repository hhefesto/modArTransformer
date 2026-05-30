-- Embedding lookup as a differentiable primitive: forward returns row i of the
-- embedding matrix; the pullback scatters the upstream cotangent back into row i
-- (zeros elsewhere) — replacing the ST scatter-add of Main.hs:812-819.  Used for
-- both token and position embeddings.
{-# OPTIONS --guardedness #-}
module ModArTransformer.Layers.Embedding where

open import Data.Nat using (ℕ)
open import Data.Fin using (Fin)
open import Data.Product using (_,_)

open import ModArTransformer.Tensor
open import ModArTransformer.Cat.Objects
open import ModArTransformer.Cat.AddFun using (mkAddFun)
open import ModArTransformer.Cat.Dual using (mkDual)
open import ModArTransformer.Cat.D
open import ModArTransformer.Cat.AdditiveTensor

embedRowD : {r c : ℕ} → Fin r → D (ℝMat r c) (ℝVec c)
embedRowD i = mkD (λ E →
  ( mrow E i
  , mkDual (mkAddFun (λ dY → mAddRow mzero i dY)) ))
