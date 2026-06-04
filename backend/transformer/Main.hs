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
import Data.Maybe (fromMaybe)
import Control.DeepSeq (NFData, deepseq)
import Control.Parallel.Strategies (parMap, rdeepseq, rseq, parListChunk, using)
import qualified Options.Applicative as O
import System.Directory (doesFileExist, renameFile)
import System.IO (hFlush, hPutStrLn, stdout, stderr)
import System.Exit (exitFailure)
import Data.Time.Clock (getCurrentTime, diffUTCTime)
import Text.Printf (printf)
import Text.Read (readMaybe)
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

-- Flat parameter list in the exact Serialize leaf order for Params2 (two stacked
-- blocks): tok, pos, block1, block2, unembed; each block = Wq,Wk,Wv,Wo, LN1, FFN
-- up, FFN down, LN2.
initFlat2 :: Int -> Int -> Int -> Int -> StdGen -> ([Double], StdGen)
initFlat2 v dM dF dK g0 =
  let xav r c = sqrt (6 / fromIntegral (r + c))
      mblock r c g = uniformN (r * c) (xav r c) g
      zeros k = replicate k 0
      ones  k = replicate k 1
      lin o i g = let (w, g') = mblock o i g in (w ++ zeros o, g')
      block g =                                            -- one transformer block
        let (wq, g1) = lin dK dM g
            (wk, g2) = lin dK dM g1
            (wv, g3) = lin dK dM g2
            (wo, g4) = lin dM dK g3
            ln1      = ones dM ++ zeros dM
            (up, g5) = lin dF dM g4
            (dn, g6) = lin dM dF g5
            ln2      = ones dM ++ zeros dM
        in (wq ++ wk ++ wv ++ wo ++ ln1 ++ up ++ dn ++ ln2, g6)
      (tok,  g1) = mblock v dM g0
      (pos,  g2) = mblock 2 dM g1
      (blk1, g3) = block g2
      (blk2, g4) = block g3
      (un,   g5) = lin v dM g4
      flat = tok ++ pos ++ blk1 ++ blk2 ++ un
  in (flat, g5)

-- ── accuracy ────────────────────────────────────────────────────────────────────

argmaxList :: [Double] -> Int
argmaxList xs = snd (foldl' step (head xs, 0) (zip xs [0 ..]))
  where step (best, bi) (x, i) = if x > best then (x, i) else (best, bi)

-- ── parallel batch primitives (model-agnostic) ─────────────────────────────────
--
-- Each example's reverse pass is an independent runST/tape, and `addA` is the
-- cotangent monoid (commutative+associative), so per-example gradients can be
-- evaluated concurrently and summed.  `parMap` preserves order and the fold runs
-- in that order, so the summed gradient is numerically identical to a sequential
-- fold — parallelism here changes wall-clock only, not the result.  `rdeepseq`
-- forces the whole gradient inside the spark (WHNF would leave matrices as thunks
-- and the work would leak back out sequentially — hence the NFData constraint).

parBatchGrad :: (Additive p, NFData p)
             => (Int -> Int -> Int -> p -> (p, Double)) -> p -> [(Int, Int, Int)] -> (p, Double)
parBatchGrad grad ps batch =
  foldl' (\(!gA, !lA) (g, l) -> (addA gA g, lA + l)) (zeroA, 0)
         (parMap rdeepseq (\(a, b, t) -> grad a b t ps) batch)

-- accuracy over a dataset, examples scored in parallel (chunked to bound spark count).
parAccuracy :: (Int -> Int -> p -> V v) -> p -> [(Int, Int, Int)] -> Double
parAccuracy _      _  [] = 0
parAccuracy logits ps xs =
  let hits = ([ if argmaxList (vtoList (logits a b ps)) == t then 1 else 0
              | (a, b, t) <- xs ] :: [Int]) `using` parListChunk 64 rseq
  in fromIntegral (sum hits) / fromIntegral (length xs)

accuracy :: forall v dM dF dK. ParamsC v dM dF dK
         => Params v dM dF dK -> [(Int, Int, Int)] -> Double
accuracy = parAccuracy transformerLogitsVal

-- ── batch / epoch ─────────────────────────────────────────────────────────────

-- accumulate gradient and loss over a batch (one chain-rule reverse pass per
-- example, examples evaluated in parallel — see parBatchGrad).
batchGradLoss :: forall v dM dF dK. ParamsC v dM dF dK
              => Params v dM dF dK -> [(Int, Int, Int)] -> (Params v dM dF dK, Double)
batchGradLoss = parBatchGrad transformerGradLoss

-- ── checkpoint ──────────────────────────────────────────────────────────────────

-- Atomic save: write a sibling temp file, then rename into place.  rename is
-- atomic on the same filesystem (the temp lives in the same dir), so a save
-- interrupted mid-write leaves only the partial `.tmp` — the resumable checkpoint
-- at `path` is never corrupted.
saveCkpt :: Serialize p => FilePath -> p -> AdamState p -> Int -> IO ()
saveCkpt path ps st epoch = do
  let tmp    = path ++ ".tmp"
      header = unwords [ "CHECKPOINT_ADAMW", show epoch
                       , show (asT st), show (asB1Pow st), show (asB2Pow st) ]
      body   = toFloats ps ++ toFloats (asM st) ++ toFloats (asV st)
  writeFile tmp (unlines (header : map show body))
  renameFile tmp path

-- Load + validate.  `nParam` is the expected per-section float count; a file whose
-- body isn't exactly 3*nParam floats (params + Adam m + v) is incomplete/corrupt
-- (e.g. an interrupted pre-atomic-save) — fail loudly instead of crashing deep in
-- `fromFloats` with a cryptic shape error.
loadCkpt :: Serialize p => Int -> FilePath -> IO (Maybe (p, AdamState p, Int))
loadCkpt nParam path = do
  exists <- doesFileExist path
  if not exists then pure Nothing else do
    contents <- readFile path
    length contents `seq` pure ()
    case lines contents of
      [] -> bad "empty checkpoint file"
      (header : rest) -> case words header of
        ["CHECKPOINT_ADAMW", eS, tsS, b1S, b2S] ->
          case (readMaybe eS, readMaybe tsS, readMaybe b1S, readMaybe b2S, traverse readMaybe rest) of
            (Just e, Just ts, Just b1, Just b2, Just nums)
              | length nums == 3 * nParam -> do
                  let (ps,  r1) = fromFloats nums
                      (mm,  r2) = fromFloats r1
                      (vv,  _)  = fromFloats r2
                      st = AdamState ts b1 b2 mm vv
                  pure (Just (ps, st, e))
              | otherwise -> bad $ "wrong float count: " ++ show (length nums)
                  ++ ", expected " ++ show (3 * nParam)
                  ++ " (likely an interrupted or wrong-model save)"
            _ -> bad "non-numeric header/body field"
        _ -> bad "unrecognized checkpoint header"
  where
    bad msg = do
      hPutStrLn stderr $ "loadCkpt: " ++ path ++ " is invalid: " ++ msg
        ++ ". Delete it to start fresh, or pass --checkpoint PATH for another file."
      exitFailure

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

-- Generic training loop, polymorphic over the parameter type `p`.  The
-- model-specific pieces — how to build fresh params, the batch gradient, and
-- accuracy — are passed in, so the same loop drives both the 1-layer and 2-layer
-- transformers (see train1 / train2W below).
train :: forall p. (Adam p, Serialize p, Additive p, Scale p, NFData p)
      => Cfg
      -> String                                    -- model description (header)
      -> (Int, Int, Int, Int)                      -- (vocab, dModel, dFF, dK) for the banner
      -> (StdGen -> p)                             -- build fresh params from the seed gen
      -> (p -> [(Int, Int, Int)] -> (p, Double))   -- batch gradient + summed loss
      -> (p -> [(Int, Int, Int)] -> Double)        -- accuracy on a dataset
      -> IO ()
train cfg modelDesc (vI, dMI, dFI, dKI) mkInit bgrad acc = do
  let p   = cP cfg
      g0  = mkStdGen (cSeed cfg)
      adamCfg = AdamConfig 0.9 0.999 1.0e-8 (cWD cfg)
      allData = generateData p
      (tr0, te0, _) = splitData (cFrac cfg) g0 allData
      batchesPer = length (chunksOf (cBatch cfg) tr0)
      totalSteps = cSchedEp cfg * max 1 batchesPer
  let freshP         = mkInit g0
      nParamExpected = length (toFloats freshP)
  loaded <- loadCkpt nParamExpected (cCkpt cfg)
  (params0, st0, startEpoch) <- case loaded of
    Just (ps, st, e) -> do
      printf "Resumed %s at epoch %d (step %d)\n" (cCkpt cfg) e (asT st)
      pure (ps, st, e + 1)
    Nothing -> do
      printf "Fresh init (Xavier).\n"
      pure (freshP, initAdam, 1)
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
  printf " Model: %s\n" modelDesc
  printf "        dModel=%d  dFF=%d  dK=%d  vocab=%d  |  %d parameters.\n" dMI dFI dKI vI nParam
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
            tEnd <- getCurrentTime
            let secs = realToFrac (diffUTCTime tEnd t0) :: Double
                epochsRun = epoch - startEpoch
            printf "Stopped after epoch %d | %d steps | %.1fs total" (epoch - 1) (asT st) secs
            printf " (%.3f s/epoch over %d epochs this run); checkpoint saved.\n"
              (if epochsRun > 0 then secs / fromIntegral epochsRun else 0) (max 0 epochsRun)
        | otherwise = do
            let (shuf, g') = shuffle g tr0
                batches = chunksOf (cBatch cfg) shuf
                (ps', st', lossSum) = foldl' batchStep (ps, st, 0) batches
                batchStep (!pp, !ss, !ls) batch =
                  let (gAvgRaw, l) = bgrad pp batch
                      n = fromIntegral (length batch)
                      gAvg = scaleA (1 / n) gAvgRaw
                      step = asT ss + 1
                      lr = lrWarmupCosine (cWarmup cfg) totalSteps (cBaseLR cfg) (cMinLR cfg) step
                      (pp', ss') = adamStep adamCfg lr pp ss gAvg
                  in (pp', ss', ls + l / n)
                avgLoss = lossSum / fromIntegral (max 1 (length batches))
            -- force params before logging / next epoch (bound memory)
            ps' `deepseq` pure ()
            if epoch `mod` cEvalEv cfg == 0
              then do
                let tr = acc ps' tr0
                    te = acc ps' te0
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

-- ── model wrappers (supply the model-specific pieces to the generic loop) ───────

train1 :: forall v dM dF dK.
          (ParamsC v dM dF dK, Adam (Params v dM dF dK), Serialize (Params v dM dF dK), Scale (Params v dM dF dK))
       => Cfg -> Proxy '(v, dM, dF, dK) -> IO ()
train1 cfg _ =
  train cfg "1-layer single-head transformer, seqLen=2, readout at position 0"
        (vI, dMI, dFI, dKI) mkInit batchGradLoss accuracy
  where vI  = fromIntegral (natVal (Proxy @v))  :: Int
        dMI = fromIntegral (natVal (Proxy @dM)) :: Int
        dFI = fromIntegral (natVal (Proxy @dF)) :: Int
        dKI = fromIntegral (natVal (Proxy @dK)) :: Int
        mkInit g = fst (fromFloats (fst (initFlat vI dMI dFI dKI g))) :: Params v dM dF dK

train2W :: forall v dM dF dK.
           (ParamsC2 v dM dF dK, Adam (Params2 v dM dF dK), Serialize (Params2 v dM dF dK), Scale (Params2 v dM dF dK))
        => Cfg -> Proxy '(v, dM, dF, dK) -> IO ()
train2W cfg _ =
  train cfg "2-layer transformer (two stacked single-head blocks), seqLen=2, readout at position 0"
        (vI, dMI, dFI, dKI) mkInit bgrad acc
  where vI  = fromIntegral (natVal (Proxy @v))  :: Int
        dMI = fromIntegral (natVal (Proxy @dM)) :: Int
        dFI = fromIntegral (natVal (Proxy @dF)) :: Int
        dKI = fromIntegral (natVal (Proxy @dK)) :: Int
        mkInit g = fst (fromFloats (fst (initFlat2 vI dMI dFI dKI g))) :: Params2 v dM dF dK
        bgrad = parBatchGrad transformerGradLoss2
        acc   = parAccuracy transformerLogitsVal2

-- ── entry ───────────────────────────────────────────────────────────────────────

-- Usage: backend-transformer-train [-m MODE] [-e N] [-s N] [-c PATH]
--   -m / --mode        p5 | p53 | p53hi | p97 | p97hi   (1-layer)
--                      p5l2 | p53l2 | p97l2             (2-layer — like the paper)
--                      (default p97l2 — the paper's modulus, 2 layers)
--   -e / --epochs      epochs to run THIS invocation (default: per-mode; resumes from checkpoint)
--   -s / --seed        RNG seed for split/init/shuffle (default 42)
--   -c / --checkpoint  checkpoint file (default checkpoint-<mode>.ckpt)
--   -h / --help        auto-generated usage (lists all flags + defaults)
data Opts = Opts
  { optMode   :: String          -- -m / --mode        (default "p97l2")
  , optEpochs :: Maybe Int       -- -e / --epochs      (Nothing → per-mode default)
  , optSeed   :: Int             -- -s / --seed        (default 42)
  , optCkpt   :: Maybe FilePath  -- -c / --checkpoint  (Nothing → checkpoint-<mode>.ckpt)
  }

optsP :: O.Parser Opts
optsP = Opts
  <$> O.strOption (O.long "mode" <> O.short 'm' <> O.metavar "MODE"
        <> O.value "p97l2" <> O.showDefault
        <> O.help "p5|p53|p53hi|p97|p97hi (1-layer); p5l2|p53l2|p97l2 (2-layer)")
  <*> O.optional (O.option O.auto (O.long "epochs" <> O.short 'e' <> O.metavar "N"
        <> O.help "epochs to run this invocation (default: per-mode — 1000, 2000 for p5)"))
  <*> O.option O.auto (O.long "seed" <> O.short 's' <> O.metavar "N"
        <> O.value 42 <> O.showDefault
        <> O.help "RNG seed for split/init/shuffle (run is deterministic per seed)")
  <*> O.optional (O.strOption (O.long "checkpoint" <> O.short 'c' <> O.metavar "PATH"
        <> O.help "checkpoint file (default checkpoint-<mode>.ckpt); resumes if present"))

main :: IO ()
main = do
  o <- O.execParser $ O.info (optsP O.<**> O.helper)
         ( O.fullDesc
           <> O.header "backend-transformer-train — denotational modular-arithmetic transformer"
           <> O.progDesc "Train (a+b) mod p via reverse-mode AD on a Wengert tape; reproduces grokking." )
  case optMode o of
    "p5"    -> train1  (mk o 5  1.0 25 5000   200  2000 "checkpoint-p5.ckpt"    1.0e-3) (Proxy @'(5, 16, 64, 16))
    -- accelerated grokking probe: identical to the canonical run but with 10× weight
    -- decay (the established lever that brings grokking onset earlier).  Flat lr
    -- (cosine over 500000 epochs ≈ constant 1e-3) so the transition isn't starved.
    "p53hi" -> train1  (mk o 53 0.5 32 500000 2000 1000 "checkpoint-p53hi.ckpt" 1.0e-2) (Proxy @'(53, 64, 256, 64))
    -- p=97 (the grokking paper's modulus) on our 1-layer/single-head architecture;
    -- ~147 batches/epoch, so slower per epoch than p53.  "p97hi" is the accelerated
    -- recipe (wd=1e-2, the lever that brings grokking onset earlier — use this to
    -- grok); plain "p97" is the canonical wd=1e-3.
    "p97hi" -> train1  (mk o 97 0.5 32 500000 2000 1000 "checkpoint-p97hi.ckpt" 1.0e-2) (Proxy @'(97, 64, 256, 64))
    "p97"   -> train1  (mk o 97 0.5 32 500000 2000 1000 "checkpoint-p97.ckpt"   1.0e-3) (Proxy @'(97, 64, 256, 64))
    "p53"   -> train1  (mk o 53 0.5 32 500000 2000 1000 "checkpoint-p53.ckpt"   1.0e-3) (Proxy @'(53, 64, 256, 64))
    -- ── two-layer variants (two stacked transformer blocks — closer to the paper) ──
    "p5l2"  -> train2W (mk o 5  1.0 25 5000   200  2000 "checkpoint-p5l2.ckpt"  1.0e-3) (Proxy @'(5, 16, 64, 16))
    "p53l2" -> train2W (mk o 53 0.5 32 500000 2000 1000 "checkpoint-p53l2.ckpt" 1.0e-2) (Proxy @'(53, 64, 256, 64))
    "p97l2" -> train2W (mk o 97 0.5 32 500000 2000 1000 "checkpoint-p97l2.ckpt" 1.0e-2) (Proxy @'(97, 64, 256, 64))
    badMode -> do
      hPutStrLn stderr $ "Unknown mode: " ++ badMode
      hPutStrLn stderr "Valid modes: p5, p53, p53hi, p97, p97hi, p5l2, p53l2, p97l2"
      exitFailure
  where
    -- per-mode literals + the three runtime overrides → a Cfg.
    -- (baseLR=1e-3, minLR=1e-5, and the eval/checkpoint cadence — print every 5
    --  epochs, checkpoint every 50 — are constant across all modes, so fixed here.)
    mk o p frac batch schedEp warmup epDef path wd =
      Cfg p frac batch schedEp warmup 1.0e-3 1.0e-5 5 50
          (fromMaybe epDef (optEpochs o))   -- cMaxRun
          (fromMaybe path  (optCkpt   o))   -- cCkpt
          wd (optSeed o)
