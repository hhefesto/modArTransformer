{-# OPTIONS --guardedness #-}
module ModArTransformer.Train where

open import Agda.Builtin.Float using (Float; primNatToFloat)
open import Data.Nat     using (ℕ; zero; suc; _+_)
open import Data.List    using (List; []; _∷_; length)
open import Data.Product using (_×_; _,_)
open import Data.Bool    using (Bool; true; false; if_then_else_)
open import Data.Vec.Base as Vec using (map)
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

-- Strict foldl backed by Haskell's Data.List.foldl' to prevent thunk buildup.
{-# FOREIGN GHC import qualified Data.List as DL #-}

postulate
  foldl' : {A B : Set} → (A → B → A) → A → List B → A
  seqBy  : {A B : Set} → A → B → B

{-# COMPILE GHC foldl' = \ _ _ f z xs -> DL.foldl' f z xs #-}
{-# COMPILE GHC seqBy  = \ _ _ x y -> seq x y #-}

private
  forceMat : {m n : ℕ} → ℝMat m n → Float
  forceMat M = vsum (Vec.map vsum M)

  forceLinear : {out inp : ℕ} → LinearParams out inp → Float
  forceLinear p = forceMat (linW p) f+ vsum (linB p)

  forceLN : {n : ℕ} → LayerNormParams n → Float
  forceLN p = vsum (lnGamma p) f+ vsum (lnBeta p)

  forceFFN : {dModel dFF : ℕ} → FFNParams dModel dFF → Float
  forceFFN p = forceLinear (ffnLinear1 p) f+ forceLinear (ffnLinear2 p)

  forceAttn : {dModel dK : ℕ} → AttentionParams dModel dK → Float
  forceAttn p = forceLinear (attnWq p) f+
                forceLinear (attnWk p) f+
                forceLinear (attnWv p) f+
                forceLinear (attnWo p)

  forceTransformer : {p dModel dFF dK : ℕ}
                   → TransformerParams (suc p) dModel dFF dK
                   → Float
  forceTransformer p = forceMat (tokEmbed p) f+
                       forceMat (posEmbed p) f+
                       forceAttn (attnP p) f+
                       forceLN (ln1P p) f+
                       forceFFN (ffnP p) f+
                       forceLN (ln2P p) f+
                       forceLinear (unembed p)

  forceMatMoments : {m n : ℕ} → MatMoments m n → Float
  forceMatMoments m = forceMat (MatMoments.mmM m) f+ forceMat (MatMoments.mmV m)

  forceVecMoments : {n : ℕ} → VecMoments n → Float
  forceVecMoments v = vsum (VecMoments.vmM v) f+ vsum (VecMoments.vmV v)

  forceLinearMoments : {out inp : ℕ} → LinearMoments out inp → Float
  forceLinearMoments m = forceMatMoments (LinearMoments.lmW m) f+
                         forceVecMoments (LinearMoments.lmB m)

  forceLNMoments : {n : ℕ} → LNMoments n → Float
  forceLNMoments m = forceVecMoments (LNMoments.lnmGamma m) f+
                     forceVecMoments (LNMoments.lnmBeta m)

  forceAttnMoments : {dModel dK : ℕ} → AttnMoments dModel dK → Float
  forceAttnMoments m = forceLinearMoments (AttnMoments.aqM m) f+
                       forceLinearMoments (AttnMoments.akM m) f+
                       forceLinearMoments (AttnMoments.avM m) f+
                       forceLinearMoments (AttnMoments.aoM m)

  forceAdamState : {p dModel dFF dK : ℕ} → AdamState p dModel dFF dK → Float
  forceAdamState a = primNatToFloat (AdamState.adamT a) f+
                     AdamState.adamB1t a f+
                     AdamState.adamB2t a f+
                     forceMatMoments (AdamState.mTokEmb a) f+
                     forceMatMoments (AdamState.mPosEmb a) f+
                     forceAttnMoments (AdamState.mAttn a) f+
                     forceLNMoments (AdamState.mLn1 a) f+
                     forceLinearMoments (AdamState.mFFN1 a) f+
                     forceLinearMoments (AdamState.mFFN2 a) f+
                     forceLNMoments (AdamState.mLn2 a) f+
                     forceLinearMoments (AdamState.mUnembed a)

  strictBatchAcc : {p dModel dFF dK : ℕ}
                 → TransformerParams (suc p) dModel dFF dK × Float
                 → TransformerParams (suc p) dModel dFF dK × Float
  strictBatchAcc acc@(g , l) = seqBy (forceTransformer g f+ l) acc

  strictTrainAcc : {p dModel dFF dK : ℕ}
                 → TransformerParams (suc p) dModel dFF dK
                 × AdamState p dModel dFF dK
                 × Float
                 → TransformerParams (suc p) dModel dFF dK
                 × AdamState p dModel dFF dK
                 × Float
  strictTrainAcc acc@(p , a , l) = seqBy (forceTransformer p f+ forceAdamState a f+ l) acc

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
  let correct = foldl' (λ acc ex →
        let pred = inferInfix params (Example.exA ex) (Example.exB ex)
        in  if natEq (toℕ pred) (toℕ (Example.exTarget ex))
              then suc acc else acc)
        0 xs
  in  primNatToFloat correct f/ primNatToFloat (length xs)

-- ─── Single-example gradient (the Conal AD call) ─────────────────────────────

private
  unpackGradLoss : {p dModel dFF dK : ℕ}
                 → Float × Dual AddFun (TransformerParams (suc p) dModel dFF dK) Float
                 → TransformerParams (suc p) dModel dFF dK × Float
  unpackGradLoss (loss , mkDual (mkAddFun pb)) = (pb 1.0 , loss)

exampleGradLoss : {p dModel dFF dK : ℕ}
                → TransformerParams (suc p) dModel dFF dK
                → Example (suc p)
                → TransformerParams (suc p) dModel dFF dK × Float
exampleGradLoss params ex =
  unpackGradLoss
    (run (transformerD (Example.exA ex) (Example.exB ex) (Example.exTarget ex)) params)

-- ─── Batch training step (Main.hs:866-913) ────────────────────────────────────

private
  addGradLoss : {p dModel dFF dK : ℕ}
              → TransformerParams (suc p) dModel dFF dK × Float
              → TransformerParams (suc p) dModel dFF dK × Float
              → TransformerParams (suc p) dModel dFF dK × Float
  addGradLoss (accG , accL) (g , l) = (addTransformer accG g , accL f+ l)

  batchStep : {p dModel dFF dK : ℕ}
            → TransformerParams (suc p) dModel dFF dK
            → TransformerParams (suc p) dModel dFF dK × Float
            → Example (suc p)
            → TransformerParams (suc p) dModel dFF dK × Float
  batchStep params acc ex = strictBatchAcc (addGradLoss acc (exampleGradLoss params ex))

  finishBatch : {p dModel dFF dK : ℕ}
              → Float → AdamConfig → Float → Float
              → TransformerParams (suc p) dModel dFF dK
              → AdamState p dModel dFF dK
              → ℕ
              → TransformerParams (suc p) dModel dFF dK × Float
              → TransformerParams (suc p) dModel dFF dK
              × AdamState p dModel dFF dK
              × Float
  finishBatch {p} {dModel} {dFF} {dK} lr cfg b1t b2t params adam n (totalGrad , totalLoss) =
    continue (adamStep lr cfg b1t b2t params
                       (scaleTransformer (fone f/ primNatToFloat n) totalGrad)
                       adam)
    where
      continue : TransformerParams (suc p) dModel dFF dK × AdamState p dModel dFF dK
               → TransformerParams (suc p) dModel dFF dK
               × AdamState p dModel dFF dK
               × Float
      continue (params' , adam') = strictTrainAcc (params' , adam' , totalLoss f/ primNatToFloat n)

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
  let totals = foldl' (batchStep params) (zeroTransformer , 0.0) batch
  in  finishBatch lr cfg b1t b2t params adam (length batch) totals

-- ─── One epoch (Main.hs:915-935) ─────────────────────────────────────────────

private
  epochStep : {p dModel dFF dK : ℕ}
            → Float → AdamConfig → Float → Float → ℕ
            → TransformerParams (suc p) dModel dFF dK
            × AdamState p dModel dFF dK
            × Float
            → List (Example (suc p))
            → TransformerParams (suc p) dModel dFF dK
            × AdamState p dModel dFF dK
            × Float
  epochStep {p} {dModel} {dFF} {dK} lr cfg b1t b2t totalSteps (params0 , adam0 , l) batch =
    continue (trainBatch lr cfg b1t b2t totalSteps params0 adam0 batch)
    where
      continue : TransformerParams (suc p) dModel dFF dK
               × AdamState p dModel dFF dK
               × Float
               → TransformerParams (suc p) dModel dFF dK
               × AdamState p dModel dFF dK
               × Float
      continue (params' , adam' , bl) = strictTrainAcc (params' , adam' , l f+ bl)

  finishEpoch : {p dModel dFF dK : ℕ}
              → StdGen
              → List (List (Example (suc p)))
              → TransformerParams (suc p) dModel dFF dK
              × AdamState p dModel dFF dK
              × Float
              → TransformerParams (suc p) dModel dFF dK
              × AdamState p dModel dFF dK
              × Float × StdGen
  finishEpoch g1 batches (params' , adam' , totalLoss) =
    (params' , adam' , meanLoss , g1)
    where
      nBatches : ℕ
      nBatches = length batches

      meanLoss : Float
      meanLoss = if natEq nBatches 0 then 0.0
                 else totalLoss f/ primNatToFloat nBatches

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
trainEpoch {p} {dModel} {dFF} {dK} epoch bsz warmup baseLR minLR cfg params adam trainData g0 =
  onShuffled (shuffleList trainData g0)
  where
    totalSteps : ℕ
    totalSteps = 500000  -- matches Main.hs

    lr : Float
    lr = lrWarmupCosine epoch warmup baseLR minLR totalSteps

    b1t : Float
    b1t = AdamState.adamB1t adam

    b2t : Float
    b2t = AdamState.adamB2t adam

    onShuffled : List (Example (suc p)) × StdGen
               → TransformerParams (suc p) dModel dFF dK
               × AdamState p dModel dFF dK
               × Float × StdGen
    onShuffled (shuffled , g1) =
      continue g1 (chunksOf bsz shuffled)
      where
        continue : StdGen → List (List (Example (suc p)))
                 → TransformerParams (suc p) dModel dFF dK
                 × AdamState p dModel dFF dK
                 × Float × StdGen
        continue g1 batches =
          let totals = foldl' (epochStep lr cfg b1t b2t totalSteps) (params , adam , 0.0) batches
          in  finishEpoch g1 batches totals
