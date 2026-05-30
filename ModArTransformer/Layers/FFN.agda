-- Position-wise feed-forward network  x ↦ W₂ · relu(W₁·x + b₁) + b₂, built by
-- composing two linear layers and the ReLU primitive.  Like every layer here it
-- is just a composite morphism in D — gradients w.r.t. all parameters and the
-- input are derived by the chain rule, with no backward code.
{-# OPTIONS --guardedness #-}
module ModArTransformer.Layers.FFN where

open import Data.Nat using (ℕ)

open import ModArTransformer.Tensor using (ℝVec)
open import ModArTransformer.Cat.Objects
open import ModArTransformer.Cat.Additive
open import ModArTransformer.Cat.D
open import ModArTransformer.Cat.VecPrim
open import ModArTransformer.Cat.AdditiveTensor
open import ModArTransformer.Layers.Linear

private variable dModel dFF : ℕ

-- Parameters: (W₁,b₁) for the up-projection, (W₂,b₂) for the down-projection.
FFNParams : ℕ → ℕ → Set
FFNParams dModel dFF = LinParams dFF dModel × LinParams dModel dFF

ffnD : D (FFNParams dModel dFF × ℝVec dModel) (ℝVec dModel)
ffnD = linearLayerD ∘D (p2 ▵D (reluVecD ∘D (linearLayerD ∘D (p1 ▵D x))))
  where
    p1 : D (FFNParams dModel dFF × ℝVec dModel) (LinParams dFF dModel)
    p1 = exlD ∘D exlD
    p2 : D (FFNParams dModel dFF × ℝVec dModel) (LinParams dModel dFF)
    p2 = exrD ∘D exlD
    x  : D (FFNParams dModel dFF × ℝVec dModel) (ℝVec dModel)
    x  = exrD
