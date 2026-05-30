-- A linear (affine) layer y = W·x + b, built *by composition* of the D
-- primitives in Cat.VecPrim — no backward pass is written here.  Parameters are
-- carried as a product (W , b) so that `Additive` (hence the gradient) is
-- derived structurally; the gradient w.r.t. W, b and x all fall out of the
-- chain rule in D (Dual AddFun).
{-# OPTIONS --guardedness #-}
module ModArTransformer.Layers.Linear where

open import Data.Nat using (ℕ)

open import ModArTransformer.Tensor using (ℝVec; ℝMat)
open import ModArTransformer.Cat.Objects
open import ModArTransformer.Cat.Additive
open import ModArTransformer.Cat.D
open import ModArTransformer.Cat.VecPrim
open import ModArTransformer.Cat.AdditiveTensor

private variable m n : ℕ

-- Parameters of a linear layer: a weight matrix and a bias vector.
LinParams : ℕ → ℕ → Set
LinParams m n = ℝMat m n × ℝVec m

-- The layer as a morphism  (params × input) ⇨ output  in the derivative
-- category.  `eval` runs it forward; `gradient` differentiates it.
linearLayerD : D (LinParams m n × ℝVec n) (ℝVec m)
linearLayerD = vaddD ∘D ((matvecD ∘D (getW ▵D getx)) ▵D getb)
  where
    getW  : D (LinParams m n × ℝVec n) (ℝMat m n)
    getW  = exlD ∘D exlD
    getb  : D (LinParams m n × ℝVec n) (ℝVec m)
    getb  = exrD ∘D exlD
    getx  : D (LinParams m n × ℝVec n) (ℝVec n)
    getx  = exrD
