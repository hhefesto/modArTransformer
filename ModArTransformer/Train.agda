-- Training loop, rewired onto the new stack.  Per-example gradients come from
-- `Cat.Grad.gradAndLoss (transformerLoss …)` (one chain rule, no bespoke
-- pullbacks); batch gradients accumulate with the derived `Additive._⊕_`, scale
-- by 1/n with `Scale`, and update via the generic `Adamable.adamStep`.
-- Inference/accuracy run `eval (transformerLogits …)`.  Strictness uses the
-- generic `Force.force` with the `seqBy`/`foldl'` FFI to bound memory.
{-# OPTIONS --guardedness #-}
module ModArTransformer.Train where

open import Agda.Builtin.Float using (Float; primNatToFloat)
open import Data.Nat     using (ℕ; zero; suc)
open import Data.List    using (List; []; _∷_; length)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Bool    using (Bool; true; false; if_then_else_)
open import Data.Fin     using (Fin; toℕ)

open import ModArTransformer.Tensor using (_f+_; _f/_; vmaxIndex)
-- opened without `using` so their tensor/product instances are in scope:
open import ModArTransformer.Cat.Additive
open import ModArTransformer.Cat.AdditiveTensor
open import ModArTransformer.Cat.Scale
open import ModArTransformer.Cat.Adamable
open import ModArTransformer.Cat.Force
open import ModArTransformer.Cat.Grad      using (eval; gradAndLoss)
open import ModArTransformer.Layers.Transformer
  using (TransformerParams; transformerLoss; transformerLogits)
open import ModArTransformer.Optimizer.Schedule using (lrWarmupCosine)
open import ModArTransformer.Data   using (Example; shuffleList; chunksOf)
open import ModArTransformer.Random using (StdGen)

private variable p dModel dFF dK : ℕ

-- Strict foldl backed by Haskell's Data.List.foldl' to prevent thunk buildup.
{-# FOREIGN GHC import qualified Data.List as DL #-}
postulate
  foldl' : {A B : Set} → (A → B → A) → A → List B → A
  seqBy  : {A B : Set} → A → B → B
{-# COMPILE GHC foldl' = \ _ _ f z xs -> DL.foldl' f z xs #-}
{-# COMPILE GHC seqBy  = \ _ _ x y -> seq x y #-}

private
  natEq : ℕ → ℕ → Bool
  natEq zero    zero    = true
  natEq zero    (suc _) = false
  natEq (suc _) zero    = false
  natEq (suc m) (suc n) = natEq m n

Params : ℕ → ℕ → ℕ → ℕ → Set
Params p dModel dFF dK = TransformerParams p dModel dFF dK

-- ─── Inference / accuracy ─────────────────────────────────────────────────────

inferInfix : Params p dModel dFF dK → Fin (suc p) → Fin (suc p) → Fin (suc p)
inferInfix params a b = vmaxIndex (eval (transformerLogits a b) params)

accuracy : Params p dModel dFF dK → List (Example (suc p)) → Float
accuracy _      [] = 0.0
accuracy params xs =
  let correct = foldl' (λ acc ex →
        let pred = inferInfix params (Example.exA ex) (Example.exB ex)
        in  if natEq (toℕ pred) (toℕ (Example.exTarget ex)) then suc acc else acc)
        0 xs
  in  primNatToFloat correct f/ primNatToFloat (length xs)

-- ─── Per-example gradient + loss (the Conal AD call) ──────────────────────────

exampleGradLoss : Params p dModel dFF dK → Example (suc p)
                → Params p dModel dFF dK × Float
exampleGradLoss params ex =
  gradAndLoss (transformerLoss (Example.exA ex) (Example.exB ex) (Example.exTarget ex)) params

-- ─── Batch step: accumulate grads, average, Adam update ───────────────────────

private
  batchAcc : Params p dModel dFF dK
           → Params p dModel dFF dK × Float → Example (suc p)
           → Params p dModel dFF dK × Float
  batchAcc params acc ex =
    let gl   = exampleGradLoss params ex
        acc' = (proj₁ acc ⊕ proj₁ gl , proj₂ acc f+ proj₂ gl)
    in  seqBy (force (proj₁ acc') f+ proj₂ acc') acc'

trainBatch : Float → AdamConfig
           → Params p dModel dFF dK → AdamState (Params p dModel dFF dK)
           → List (Example (suc p))
           → Params p dModel dFF dK × AdamState (Params p dModel dFF dK) × Float
trainBatch _  _   params adam [] = (params , adam , 0.0)
trainBatch lr cfg params adam batch =
  let (gSum , lSum) = foldl' (batchAcc params) (zeroA , 0.0) batch
      n        = primNatToFloat (length batch)
      gAvg     = scaleA (1.0 f/ n) gSum
      (params' , adam') = adamStep lr cfg params gAvg adam
  in  seqBy (force params') (params' , adam' , lSum f/ n)

-- ─── One epoch: shuffle, chunk, fold batches ──────────────────────────────────

private
  epochAcc : Float → AdamConfig
           → Params p dModel dFF dK × AdamState (Params p dModel dFF dK) × Float
           → List (Example (suc p))
           → Params p dModel dFF dK × AdamState (Params p dModel dFF dK) × Float
  epochAcc lr cfg (params , adam , lacc) batch =
    let (params' , adam' , bl) = trainBatch lr cfg params adam batch
    in  seqBy (force params') (params' , adam' , lacc f+ bl)

trainEpoch : ℕ → ℕ → ℕ → Float → Float → AdamConfig
           → Params p dModel dFF dK → AdamState (Params p dModel dFF dK)
           → List (Example (suc p)) → StdGen
           → Params p dModel dFF dK × AdamState (Params p dModel dFF dK) × Float × StdGen
trainEpoch epoch bsz warmup baseLR minLR cfg params adam trainData g0 =
  let (shuffled , g1) = shuffleList trainData g0
      batches  = chunksOf bsz shuffled
      lr       = lrWarmupCosine epoch warmup baseLR minLR 500000
      (params' , adam' , lossSum) =
        foldl' (epochAcc lr cfg) (params , adam , 0.0) batches
      nB = length batches
  in  (params' , adam' , (if natEq nB 0 then 0.0 else lossSum f/ primNatToFloat nB) , g1)
