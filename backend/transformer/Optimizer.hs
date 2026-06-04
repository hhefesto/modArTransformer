{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE RecordWildCards #-}

-- AdamW, warmup+cosine schedule (per global batch step), plus data generation,
-- split, and shuffle.  Transcribed from the recovered
-- grokking-success Main.hs @ 62b0b4d (NOT the Agda Adamable/Schedule, whose
-- eps-placement and warmup curve differ).  Weight decay applies to matrices only.
module Optimizer
  ( AdamConfig(..)
  , AdamState(..)
  , Adam(..)
  , initAdam
  , adamStep
  , lrWarmupCosine
  , chunksOf
  , shuffle
  , splitData
  , generateData
  ) where

import GHC.TypeNats (KnownNat)
import qualified Numeric.LinearAlgebra as LA
import System.Random (StdGen, randomR)
import Tensor

-- ── config / state ────────────────────────────────────────────────────────────

data AdamConfig = AdamConfig
  { adamB1  :: !Double
  , adamB2  :: !Double
  , adamEps :: !Double
  , adamWD  :: !Double
  }

data AdamState p = AdamState
  { asT     :: !Int
  , asB1Pow :: !Double
  , asB2Pow :: !Double
  , asM     :: !p
  , asV     :: !p
  }

initAdam :: Additive p => AdamState p
initAdam = AdamState 0 1 1 zeroA zeroA

-- ── per-leaf AdamW update (matrices: decoupled WD; vectors: none) ──────────────

-- (cfg, lr, b1Pow', b2Pow') -> w -> m -> v -> g -> (w', m', v')
class Adam a where
  adamUpd :: AdamConfig -> Double -> Double -> Double -> a -> a -> a -> a -> (a, a, a)

-- shared elementwise update on a flat hmatrix Vector; `wd` is the decay coefficient
-- (0 for vectors).  Mirrors Main.hs adamUpdateMat / adamUpdateVecNoWD.
adamVecR
  :: AdamConfig -> Double -> Double -> Double -> Double
  -> LA.Vector LA.R -> LA.Vector LA.R -> LA.Vector LA.R -> LA.Vector LA.R
  -> (LA.Vector LA.R, LA.Vector LA.R, LA.Vector LA.R)
adamVecR AdamConfig{..} lr b1p b2p wd w m v g =
  let !m'   = LA.scale adamB1 m + LA.scale (1 - adamB1) g
      !g2   = g * g
      !v'   = LA.scale adamB2 v + LA.scale (1 - adamB2) g2
      !mHat = LA.scale (1 / (1 - b1p)) m'
      !vHat = LA.scale (1 / (1 - b2p)) v'
      !den  = LA.cmap (+ adamEps) (LA.cmap sqrt vHat)
      !step = mHat / den
      !w'   = w - LA.scale lr step - LA.scale (lr * wd) w
  in (w', m', v')

instance (KnownNat m, KnownNat n) => Adam (M m n) where
  adamUpd cfg lr b1p b2p (M w) (M mm) (M vv) (M g) =
    let c = LA.cols w
        (w', m', v') = adamVecR cfg lr b1p b2p (adamWD cfg)
                         (LA.flatten w) (LA.flatten mm) (LA.flatten vv) (LA.flatten g)
    in (M (LA.reshape c w'), M (LA.reshape c m'), M (LA.reshape c v'))

instance KnownNat n => Adam (V n) where
  adamUpd cfg lr b1p b2p (V w) (V mm) (V vv) (V g) =
    let (w', m', v') = adamVecR cfg lr b1p b2p 0 w mm vv g   -- no weight decay
    in (V w', V m', V v')

instance (Adam a, Adam b) => Adam (a, b) where
  adamUpd cfg lr b1p b2p (wa, wb) (ma, mb) (va, vb) (ga, gb) =
    let (wa', ma', va') = adamUpd cfg lr b1p b2p wa ma va ga
        (wb', mb', vb') = adamUpd cfg lr b1p b2p wb mb vb gb
    in ((wa', wb'), (ma', mb'), (va', vb'))

adamStep :: Adam p => AdamConfig -> Double -> p -> AdamState p -> p -> (p, AdamState p)
adamStep cfg lr w st g =
  let !t'   = asT st + 1
      !b1p' = asB1Pow st * adamB1 cfg
      !b2p' = asB2Pow st * adamB2 cfg
      (w', m', v') = adamUpd cfg lr b1p' b2p' w (asM st) (asV st) g
  in (w', AdamState t' b1p' b2p' m' v')

-- ── learning-rate schedule (per global batch step) ─────────────────────────────

lrWarmupCosine :: Int -> Int -> Double -> Double -> Int -> Double
lrWarmupCosine warmup total base minLR step
  | total <= 1     = base
  | step  <= 0     = 0
  | step  <= warmup = base * (fromIntegral step / fromIntegral (max 1 warmup))
  | otherwise =
      let t        = fromIntegral (min step total)
          w        = fromIntegral warmup
          tt       = fromIntegral total
          progress = (t - w) / max 1 (tt - w)
          cosine   = 0.5 * (1 + cos (pi * progress))
      in minLR + (base - minLR) * cosine

-- ── data ───────────────────────────────────────────────────────────────────────

generateData :: Int -> [(Int, Int, Int)]
generateData p = [ (a, b, (a + b) `mod` p) | a <- [0 .. p - 1], b <- [0 .. p - 1] ]

chunksOf :: Int -> [a] -> [[a]]
chunksOf k _ | k <= 0 = error "chunksOf: chunk size must be positive"
chunksOf _ [] = []
chunksOf k xs = let (h, t) = splitAt k xs in h : chunksOf k t

-- Fisher-Yates shuffle threading a StdGen.  (Kept as the O(n²) list version: an
-- O(n) mutable-vector variant produced a different draw sequence per seed, which
-- changed the train/test split and the grokking trajectory — and its wall-clock
-- benefit was negligible vs the batch parallelism, so it isn't worth the change.)
shuffle :: StdGen -> [a] -> ([a], StdGen)
shuffle g0 xs0 = go g0 (length xs0) xs0
  where
    go g _ [] = ([], g)
    go g n xs =
      let (i, g') = randomR (0, n - 1) g
          (pre, y : post) = splitAt i xs
      in let (rest, g'') = go g' (n - 1) (pre ++ post)
         in (y : rest, g'')

splitData :: Double -> StdGen -> [a] -> ([a], [a], StdGen)
splitData frac g xs =
  let (shuf, g') = shuffle g xs
      nTrain = round (frac * fromIntegral (length shuf))
  in (take nTrain shuf, drop nTrain shuf, g')
