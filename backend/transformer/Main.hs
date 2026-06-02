{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}

-- Training driver for the categorical (Conal-AD) transformer.  Reproduces the
-- historical grokking setup on the new stack: AdamW, per-global-step warmup+cosine
-- schedule, full checkpoint/resume.  Generic over model dims so the same loop runs
-- the p=5 overfit smoke test and the p=53 grokking run.
module Main where

import Data.List (foldl')
import System.Environment (getArgs)
import System.Directory (doesFileExist)
import System.IO (hFlush, stdout)
import Text.Printf (printf)
import System.Random (StdGen, mkStdGen, randomR)
import GHC.TypeNats (KnownNat, natVal)
import Data.Proxy (Proxy(..))

import Tensor
import Serialize
import Transformer
import Optimizer

-- ── init (Xavier on matrices; zero biases; LN γ=1, β=0) ──────────────────────────

uniformN :: Int -> Double -> StdGen -> ([Double], StdGen)
uniformN n b g = go n g []
  where go 0 g' acc = (reverse acc, g')
        go k g' acc = let (x, g'') = randomR (-b, b) g' in go (k - 1) g'' (x : acc)

-- Flat parameter list in the exact Serialize leaf order for Params v dM dF dK.
initFlat :: Int -> Int -> Int -> Int -> StdGen -> ([Double], StdGen)
initFlat v dM dF dK g0 =
  let xav r c = sqrt (6 / fromIntegral (r + c))
      mblock r c g = uniformN (r * c) (xav r c) g          -- Xavier matrix
      zeros k = replicate k 0
      ones  k = replicate k 1
      -- a linear layer: matrix (o×i) then zero bias (o)
      lin o i g = let (w, g') = mblock o i g in (w ++ zeros o, g')
      (tok, g1) = mblock v dM g0
      (pos, g2) = mblock 2 dM g1
      (wq,  g3) = lin dK dM g2
      (wk,  g4) = lin dK dM g3
      (wv,  g5) = lin dK dM g4
      (wo,  g6) = lin dM dK g5
      ln1       = ones dM ++ zeros dM
      (up,  g7) = lin dF dM g6
      (dn,  g8) = lin dM dF g7
      ln2       = ones dM ++ zeros dM
      (un,  g9) = lin v dM g8
      flat = tok ++ pos ++ wq ++ wk ++ wv ++ wo ++ ln1 ++ up ++ dn ++ ln2 ++ un
  in (flat, g9)

-- ── accuracy ────────────────────────────────────────────────────────────────────

argmaxList :: [Double] -> Int
argmaxList xs = snd (foldl' step (head xs, 0) (zip xs [0 ..]))
  where step (best, bi) (x, i) = if x > best then (x, i) else (best, bi)

accuracy :: forall v dM dF dK. ParamsC v dM dF dK
         => Params v dM dF dK -> [(Int, Int, Int)] -> Double
accuracy _ [] = 0
accuracy ps xs =
  let correct = length [ () | (a, b, t) <- xs
                            , argmaxList (vtoList (transformerLogitsVal a b ps)) == t ]
  in fromIntegral correct / fromIntegral (length xs)

-- ── batch / epoch ─────────────────────────────────────────────────────────────

-- accumulate gradient and loss over a batch (one chain-rule reverse pass per example)
batchGradLoss :: forall v dM dF dK. ParamsC v dM dF dK
              => Params v dM dF dK -> [(Int, Int, Int)] -> (Params v dM dF dK, Double)
batchGradLoss ps batch =
  foldl' step (zeroA, 0) batch
  where
    step (!gAcc, !lAcc) (a, b, t) =
      let (g, l) = transformerGradLoss a b t ps
      in (addA gAcc g, lAcc + l)

-- ── checkpoint ──────────────────────────────────────────────────────────────────

saveCkpt :: Serialize p => FilePath -> p -> AdamState p -> Int -> IO ()
saveCkpt path ps st epoch = do
  let header = unwords [ "CHECKPOINT_ADAMW", show epoch
                       , show (asT st), show (asB1Pow st), show (asB2Pow st) ]
      body   = toFloats ps ++ toFloats (asM st) ++ toFloats (asV st)
  writeFile path (unlines (header : map show body))

loadCkpt :: Serialize p => FilePath -> IO (Maybe (p, AdamState p, Int))
loadCkpt path = do
  exists <- doesFileExist path
  if not exists then pure Nothing else do
    contents <- readFile path
    length contents `seq` pure ()
    case lines contents of
      (header : rest) -> case words header of
        ("CHECKPOINT_ADAMW" : e : ts : b1 : b2 : _) ->
          let nums = map read rest
              (ps,  r1) = fromFloats nums
              (mm,  r2) = fromFloats r1
              (vv,  _)  = fromFloats r2
              st = AdamState (read ts) (read b1) (read b2) mm vv
          in pure (Just (ps, st, read e))
        _ -> pure Nothing
      _ -> pure Nothing

-- ── training loop ─────────────────────────────────────────────────────────────

data Cfg = Cfg
  { cP        :: Int
  , cFrac     :: Double
  , cBatch    :: Int
  , cSchedEp  :: Int      -- epochs assumed by the schedule (sets total steps)
  , cWarmup   :: Int
  , cBaseLR   :: Double
  , cMinLR    :: Double
  , cEvalEv   :: Int
  , cCkptEv   :: Int
  , cMaxRun   :: Int      -- max epochs to run THIS invocation
  , cCkpt     :: FilePath
  , cWD       :: Double   -- AdamW weight decay (matrices only)
  }

train :: forall v dM dF dK. (ParamsC v dM dF dK, Adam (Params v dM dF dK), Serialize (Params v dM dF dK))
      => Cfg -> Proxy '(v, dM, dF, dK) -> IO ()
train cfg _ = do
  let p   = cP cfg
      g0  = mkStdGen 42
      adamCfg = AdamConfig 0.9 0.999 1.0e-8 (cWD cfg)
      allData = generateData p
      (tr0, te0, _) = splitData (cFrac cfg) g0 allData
      batchesPer = length (chunksOf (cBatch cfg) tr0)
      totalSteps = cSchedEp cfg * max 1 batchesPer
      vI = fromIntegral (natVal (Proxy @v)) :: Int
      dMI = fromIntegral (natVal (Proxy @dM)) :: Int
      dFI = fromIntegral (natVal (Proxy @dF)) :: Int
      dKI = fromIntegral (natVal (Proxy @dK)) :: Int
  loaded <- loadCkpt (cCkpt cfg)
  (params0, st0, startEpoch) <- case loaded of
    Just (ps, st, e) -> do
      printf "Resumed %s at epoch %d (step %d)\n" (cCkpt cfg) e (asT st)
      pure (ps, st, e + 1)
    Nothing -> do
      let (flat, _) = initFlat vI dMI dFI dKI g0
          ps = fst (fromFloats flat) :: Params v dM dF dK
      printf "Fresh init (Xavier).\n"
      pure (ps, initAdam, 1)
  printf "p=%d dM=%d dF=%d dK=%d | train=%d test=%d batches/epoch=%d totalSteps=%d\n"
    p dMI dFI dKI (length tr0) (length te0) batchesPer totalSteps
  hFlush stdout

  let lastEpoch = startEpoch + cMaxRun cfg - 1
      gE = mkStdGen (1000 + startEpoch)

      loop !epoch !ps !st !g
        | epoch > lastEpoch = do
            saveCkpt (cCkpt cfg) ps st (epoch - 1)
            printf "Stopped after epoch %d (checkpoint saved).\n" (epoch - 1)
        | otherwise = do
            let (shuf, g') = shuffle g tr0
                batches = chunksOf (cBatch cfg) shuf
                (ps', st', lossSum) = foldl' batchStep (ps, st, 0) batches
                batchStep (!pp, !ss, !ls) batch =
                  let (gAvgRaw, l) = batchGradLoss pp batch
                      n = fromIntegral (length batch)
                      gAvg = scaleA (1 / n) gAvgRaw
                      step = asT ss + 1
                      lr = lrWarmupCosine (cWarmup cfg) totalSteps (cBaseLR cfg) (cMinLR cfg) step
                      (pp', ss') = adamStep adamCfg lr pp ss gAvg
                  in (pp', ss', ls + l / n)
                avgLoss = lossSum / fromIntegral (max 1 (length batches))
            -- force params before logging / next epoch (bound memory)
            (toFloats ps' `seq` pure ()) :: IO ()
            if epoch `mod` cEvalEv cfg == 0
              then do
                let tr = accuracy ps' tr0
                    te = accuracy ps' te0
                    lrNow = lrWarmupCosine (cWarmup cfg) totalSteps (cBaseLR cfg) (cMinLR cfg) (asT st')
                printf "%6d | step=%8d | lr=%.6f | loss=%.4f | train=%.1f%% | test=%.1f%%\n"
                  epoch (asT st') lrNow avgLoss (tr * 100) (te * 100)
                hFlush stdout
              else pure ()
            if epoch `mod` cCkptEv cfg == 0
              then saveCkpt (cCkpt cfg) ps' st' epoch
              else pure ()
            loop (epoch + 1) ps' st' g'

  loop startEpoch params0 st0 gE

-- ── entry ───────────────────────────────────────────────────────────────────────

main :: IO ()
main = do
  args <- getArgs
  case args of
    ("p5" : rest) ->
      let n = readDef 2000 rest
      in train (Cfg 5 1.0 25 5000 200 1.0e-3 1.0e-5 50 500 n "checkpoint-p5.ckpt" 1.0e-3)
               (Proxy @'(5, 16, 64, 16))
    -- accelerated grokking probe: identical to the canonical run but with 10× weight
    -- decay (the established lever that brings grokking onset earlier).  Flat lr
    -- (cosine over 500000 epochs ≈ constant 1e-3) so the transition isn't starved.
    ("p53hi" : rest) ->
      let n = readDef 1000 rest
      in train (Cfg 53 0.5 32 500000 2000 1.0e-3 1.0e-5 100 1000 n "checkpoint-p53hi.ckpt" 1.0e-2)
               (Proxy @'(53, 64, 256, 64))
    rest ->
      let n = readDef 1000 rest
      in train (Cfg 53 0.5 32 500000 2000 1.0e-3 1.0e-5 100 1000 n "checkpoint-p53.ckpt" 1.0e-3)
               (Proxy @'(53, 64, 256, 64))
  where readDef d xs = case xs of (x : _) -> maybe d id (readMaybeInt x); _ -> d
        readMaybeInt s = case reads s of [(x, "")] -> Just x; _ -> Nothing
