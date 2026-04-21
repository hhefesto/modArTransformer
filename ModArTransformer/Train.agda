{-# OPTIONS --guardedness #-}
module ModArTransformer.Train where

open import Agda.Builtin.Float using (Float; primNatToFloat)
open import Data.Nat     using (ℕ; zero; suc; _+_)
open import Data.List    using (List; []; _∷_; foldl; length)
open import Data.Product using (_×_; _,_)
open import Data.Bool    using (Bool; true; false; if_then_else_)
open import ModArTransformer.Tensor
open import ModArTransformer.Additive
open import ModArTransformer.AD.AddFun
open import ModArTransformer.AD.Dual
open import ModArTransformer.AD.Core
open import ModArTransformer.Layers.Linear
open import ModArTransformer.Layers.LayerNorm
open import ModArTransformer.Layers.FFN
open import ModArTransformer.Layers.Attention
open import ModArTransformer.Layers.Transformer
open import ModArTransformer.Optimizer.Adam
open import ModArTransformer.Optimizer.Schedule
open import ModArTransformer.Data
open import ModArTransformer.Random
open import Data.Fin using (Fin; toℕ) renaming (zero to fz; suc to fs)

-- ─── Inference: argmax of logits ─────────────────────────────────────────────

inferInfix : {p dModel dFF dK : ℕ}
           → TransformerParams (suc p) dModel dFF dK
           → Fin (suc p) → Fin (suc p)
           → Fin (suc p)
inferInfix params a b =
  let e0 = mrow (tokEmbed params) a v+ mrow (posEmbed params) fz
      e1 = mrow (tokEmbed params) b v+ mrow (posEmbed params) (fs fz)
      ((a0 , _) , _) = run attnD (attnP params , (e0 , e1))
      r10 = e0 v+ a0
      (n10 , _) = run layerNormD (ln1P params , r10)
      (f0  , _) = run ffnD (ffnP params , n10)
      r20 = n10 v+ f0
      (o0  , _) = run layerNormD (ln2P params , r20)
      logits = linW (unembed params) #> o0 v+ linB (unembed params)
  in  vmaxIndex logits

-- ─── Accuracy ─────────────────────────────────────────────────────────────────

private
  natEq : ℕ → ℕ → Bool
  natEq zero    zero    = true
  natEq zero    (suc _) = false
  natEq (suc _) zero    = false
  natEq (suc m) (suc n) = natEq m n

accuracy : {p dModel dFF dK : ℕ}
         → TransformerParams (suc p) dModel dFF dK
         → List (Example (suc p))
         → Float
accuracy _      [] = 0.0
accuracy params xs =
  let correct = foldl (λ acc ex →
        let pred = inferInfix params (Example.exA ex) (Example.exB ex)
        in  if natEq (toℕ pred) (toℕ (Example.exTarget ex))
              then suc acc else acc)
        0 xs
  in  primNatToFloat correct f/ primNatToFloat (length xs)

-- ─── Single-example gradient (the Conal AD call) ─────────────────────────────
-- gradient (transformerD a b target) params  →  ∂loss/∂params
-- This is the entire backward pass, derived categorically.

exampleGradLoss : {p dModel dFF dK : ℕ}
                → TransformerParams (suc p) dModel dFF dK
                → Example (suc p)
                → TransformerParams (suc p) dModel dFF dK × Float
exampleGradLoss params ex =
  let a      = Example.exA ex
      b      = Example.exB ex
      target = Example.exTarget ex
      (loss , mkDual (mkAddFun pb)) = run (transformerD a b target) params
  in  (pb 1.0 , loss)

-- ─── Batch training step (Main.hs:866-913) ────────────────────────────────────

trainBatch : {p dModel dFF dK : ℕ}
           → Float → AdamConfig → Float → Float → ℕ
           → TransformerParams (suc p) dModel dFF dK
           → AdamState p dModel dFF dK
           → List (Example (suc p))
           → TransformerParams (suc p) dModel dFF dK
           × AdamState p dModel dFF dK
           × Float
trainBatch _  _   _   _   _  params adam [] = (params , adam , 0.0)
trainBatch lr cfg b1t b2t totalSteps params adam batch =
  let n = length batch
      (totalGrad , totalLoss) = foldl
        (λ (accG , accL) ex →
          let (g , l) = exampleGradLoss params ex
          in  (addTransformer accG g , accL f+ l))
        (zeroTransformer , 0.0)
        batch
      s         = fone f/ primNatToFloat n
      meanGrad  = scaleTransformer s totalGrad
      meanLoss  = totalLoss f/ primNatToFloat n
      (params' , adam') = adamStep lr cfg b1t b2t params meanGrad adam
  in  (params' , adam' , meanLoss)

-- ─── One epoch (Main.hs:915-935) ─────────────────────────────────────────────

trainEpoch : {p dModel dFF dK : ℕ}
           → ℕ      -- epoch number
           → ℕ      -- batch size
           → ℕ      -- warmup steps
           → Float  -- base LR
           → Float  -- min LR
           → AdamConfig
           → TransformerParams (suc p) dModel dFF dK
           → AdamState p dModel dFF dK
           → List (Example (suc p))
           → StdGen
           → TransformerParams (suc p) dModel dFF dK
           × AdamState p dModel dFF dK
           × Float × StdGen
trainEpoch epoch bsz warmup baseLR minLR cfg params adam trainData g0 =
  let (shuffled , g1) = shuffleList trainData g0
      batches         = chunksOf bsz shuffled
      totalSteps      = 500000  -- matches Main.hs
      lr              = lrWarmupCosine epoch warmup baseLR minLR totalSteps
      b1t             = AdamState.adamB1t adam
      b2t             = AdamState.adamB2t adam
      (params' , adam' , totalLoss) = foldl
        (λ (p , a , l) batch →
          let (p' , a' , bl) = trainBatch lr cfg b1t b2t totalSteps p a batch
          in  (p' , a' , l f+ bl))
        (params , adam , 0.0)
        batches
      nBatches = length batches
      meanLoss = if natEq nBatches 0 then 0.0
                 else totalLoss f/ primNatToFloat nBatches
  in  (params' , adam' , meanLoss , g1)
