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
import Data.Time.Clock (getCurrentTime, diffUTCTime)
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
  , cSeed     :: Int      -- RNG seed: data split + Xavier init + per-epoch shuffle
  }

train :: forall v dM dF dK. (ParamsC v dM dF dK, Adam (Params v dM dF dK), Serialize (Params v dM dF dK))
      => Cfg -> Proxy '(v, dM, dF, dK) -> IO ()
train cfg _ = do
  let p   = cP cfg
      g0  = mkStdGen (cSeed cfg)
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
  -- ── detailed run header: what is being trained, how, and why it matters ──────
  let p2     = p * p
      nParam = length (toFloats params0)
      chance = 100 / fromIntegral p :: Double
  printf "════════════════════════════════════════════════════════════════════════\n"
  printf " Modular-arithmetic GROKKING — categorical (Conal-AD) transformer\n"
  printf "════════════════════════════════════════════════════════════════════════\n"
  printf " Task : learn (a + b) mod %d.  Two input tokens a,b in {0..%d} -> the model\n" p (p - 1)
  printf "        must output the residue (a+b) mod %d.  All %d ordered pairs are\n" p p2
  printf "        split %.0f%%/%.0f%% train/test, so it trains on only %d of %d facts and\n"
         (cFrac cfg * 100) (100 - cFrac cfg * 100) (length tr0) p2
  printf "        must GENERALIZE the rule to the %d pairs it never sees.\n" (length te0)
  printf " Why  : \"grokking\" — delayed generalization (Power et al. 2022, modulus p=97).\n"
  printf "        Train acc -> ~100%% within a few hundred epochs (memorization) while\n"
  printf "        test acc stays near chance (~%.1f%%) far longer, then jumps sharply once\n" chance
  printf "        weight decay drives the net onto a generalizing circuit.  Watch test%%.\n"
  printf " Model: 1-layer single-head transformer, seqLen=2, readout at position 0.\n"
  printf "        dModel=%d  dFF=%d  dK=%d  vocab=%d  |  %d parameters.\n" dMI dFI dKI p nParam
  printf " AD   : gradient = ONE reverse pass of Conal Elliott's AD-as-categories\n"
  printf "        (Wengert tape, Tape.hs); no hand-written backward.  Loss = cross-\n"
  printf "        entropy (softmax as a [0,1]-enriched copresheaf; relative entropy to\n"
  printf "        the Dirac truth — Tai-Danae Bradley).\n"
  printf " Optim: AdamW (b1=0.9 b2=0.999 eps=1e-8), decoupled weight decay wd=%g on\n" (cWD cfg)
  printf "        MATRICES ONLY — the grokking lever (higher wd groks sooner).  LR:\n"
  printf "        warmup %d steps -> ~flat %g (cosine over %d steps).  batch=%d;\n"
         (cWarmup cfg) (cBaseLR cfg) totalSteps (cBatch cfg)
  printf "        1 step = 1 batch gradient update (the axis the paper reports).\n"
  printf " Run  : seed=%d (deterministic per seed); checkpoint %s every %d epochs\n"
         (cSeed cfg) (cCkpt cfg) (cCkptEv cfg)
  printf "        (params+Adam+step+epoch, auto-resumes); eval every %d epochs.\n" (cEvalEv cfg)
  printf "────────────────────────────────────────────────────────────────────────\n"
  printf " Legend — banner fields:\n"
  printf "   p             modulus = vocab size; tokens are residues 0..p-1, task (a+b) mod p\n"
  printf "   dM            dModel: width of the residual stream / token-embedding vectors\n"
  printf "   dF            dFF: hidden width of the feed-forward (MLP) block\n"
  printf "   dK            attention query/key/value dimension (single head)\n"
  printf "   train/test    counts of training / held-out example pairs (sum = p*p)\n"
  printf "   batches/epoch minibatches per epoch = ceil(train / batch)\n"
  printf "   totalSteps    LR-schedule horizon (schedEpochs*batches/epoch); cosine denominator,\n"
  printf "                 set huge so lr stays ~flat\n"
  printf "   seed          RNG seed for split + init + shuffle (run is deterministic per seed)\n"
  printf " Legend — per-epoch columns:\n"
  printf "   epoch         one full pass over the training set\n"
  printf "   step          cumulative optimizer steps = batch gradient updates (the paper's x-axis)\n"
  printf "   lr            current learning rate (warmup -> ~flat cosine)\n"
  printf "   loss          mean cross-entropy over the epoch's batches (nats)\n"
  printf "   train%%        accuracy on the training set (argmax logit == target)\n"
  printf "   test%%         accuracy on the held-out set — the grokking signal to watch\n"
  printf "   elapsed       wall-clock seconds since this invocation started\n"
  printf "════════════════════════════════════════════════════════════════════════\n"
  printf "p=%d dM=%d dF=%d dK=%d | train=%d test=%d batches/epoch=%d totalSteps=%d seed=%d\n"
    p dMI dFI dKI (length tr0) (length te0) batchesPer totalSteps (cSeed cfg)
  -- column header for the per-epoch eval rows that follow
  printf "%7s | %9s | %8s | %8s | %7s | %7s | %8s\n"
    "epoch" "step" "lr" "loss" "train%" "test%" "elapsed"
  hFlush stdout
  t0 <- getCurrentTime

  let lastEpoch = startEpoch + cMaxRun cfg - 1
      gE = mkStdGen (cSeed cfg + 1000 + startEpoch)

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
                now <- getCurrentTime
                let elapsed = realToFrac (diffUTCTime now t0) :: Double
                printf "%7d | %9d | %8.6f | %8.4f | %7.1f | %7.1f | %7.1fs\n"
                  epoch (asT st') lrNow avgLoss (tr * 100) (te * 100) elapsed
                hFlush stdout
              else pure ()
            if epoch `mod` cCkptEv cfg == 0
              then saveCkpt (cCkpt cfg) ps' st' epoch
              else pure ()
            loop (epoch + 1) ps' st' g'

  loop startEpoch params0 st0 gE

-- ── entry ───────────────────────────────────────────────────────────────────────

-- Usage: backend-transformer-train [MODE] [MAX_EPOCHS] [SEED]
--   MODE       p5 | p53 | p53hi | p97 | p97hi   (default p53)
--   MAX_EPOCHS epochs to run THIS invocation (resumes from checkpoint)
--   SEED       RNG seed for split/init/shuffle (default 42; vary for non-identical runs)
main :: IO ()
main = do
  args <- getArgs
  let (mode, rest) = case args of (m : r) -> (m, r); [] -> ("p53", [])
      seed = readArg 1 42 rest
  case mode of
    "p5" ->
      train (Cfg 5 1.0 25 5000 200 1.0e-3 1.0e-5 50 500 (readArg 0 2000 rest) "checkpoint-p5.ckpt" 1.0e-3 seed)
            (Proxy @'(5, 16, 64, 16))
    -- accelerated grokking probe: identical to the canonical run but with 10× weight
    -- decay (the established lever that brings grokking onset earlier).  Flat lr
    -- (cosine over 500000 epochs ≈ constant 1e-3) so the transition isn't starved.
    "p53hi" ->
      train (Cfg 53 0.5 32 500000 2000 1.0e-3 1.0e-5 100 1000 (readArg 0 1000 rest) "checkpoint-p53hi.ckpt" 1.0e-2 seed)
            (Proxy @'(53, 64, 256, 64))
    -- p=97 (the grokking paper's modulus) on our 1-layer/single-head architecture;
    -- ~147 batches/epoch, so slower per epoch than p53.  "p97hi" is the accelerated
    -- recipe (wd=1e-2, the lever that brings grokking onset earlier — use this to
    -- grok); plain "p97" is the canonical wd=1e-3.
    "p97hi" ->
      train (Cfg 97 0.5 32 500000 2000 1.0e-3 1.0e-5 100 1000 (readArg 0 1000 rest) "checkpoint-p97hi.ckpt" 1.0e-2 seed)
            (Proxy @'(97, 64, 256, 64))
    "p97" ->
      train (Cfg 97 0.5 32 500000 2000 1.0e-3 1.0e-5 100 1000 (readArg 0 1000 rest) "checkpoint-p97.ckpt" 1.0e-3 seed)
            (Proxy @'(97, 64, 256, 64))
    -- default / "p53": canonical run, faithful hyperparameters (wd=1e-3).
    _ ->
      train (Cfg 53 0.5 32 500000 2000 1.0e-3 1.0e-5 100 1000 (readArg 0 1000 rest) "checkpoint-p53.ckpt" 1.0e-3 seed)
            (Proxy @'(53, 64, 256, 64))
  where readArg i d xs = case drop i xs of (x : _) -> maybe d id (readMaybeInt x); _ -> d
        readMaybeInt s = case reads s of [(x, "")] -> Just x; _ -> Nothing
