{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns #-}

-- market-train — train the VERIFIED sequence transformer (Layers/SeqTransformer
-- ↔ Transformer.ParamsSeq, conformance-gated) as a next-token language model
-- over quantized candle returns (README §10 Phase 4).
--
-- The language: candles → log returns → quantile bins (MarketTokenizer, the
-- realization of Semantics/MarketLanguage.agda's quantizer; edges fitted on the
-- TRAIN window only).  Each training sequence is ⊥ followed by n−1 bin tokens
-- (a sliding window over the return stream); the model's softmax at each prefix
-- is its meaning copresheaf π(·|prefix), and the loss is the mean relative
-- entropy to the actual continuations (seqGradLoss — the same morphism the Agda
-- spec defines and the oracle gates).
--
-- The training HARNESS (split, shuffle, schedule, clipping, checkpoint cadence)
-- is engineering, out of conformance scope — README §4.
--
-- Walk-forward discipline: train = first --train-frac of time, validation =
-- the rest, never shuffled across the boundary; the tokenizer artifact is
-- saved next to the checkpoint so the prompt decodes with the exact training
-- quantization.
module Main where

import Control.Monad (when, unless, forM_)
import Control.Parallel.Strategies (parMap, rdeepseq)
import Data.List (foldl1', sortOn, foldl')
import Data.Maybe (fromMaybe, mapMaybe)
import qualified Data.Map.Strict as Map
import System.Directory (doesFileExist)
import System.IO (hFlush, hPutStrLn, stdout, stderr, isEOF)
import System.Exit (exitFailure)
import System.Random (StdGen, mkStdGen, split)
import qualified System.Random as Rnd
import Data.Time.Clock (getCurrentTime, diffUTCTime)
import Text.Printf (printf)
import Text.Read (readMaybe)
import qualified Options.Applicative as O

import Tensor (Additive(..), Scale(..), V, vtoList)
import Serialize (Serialize, toFloats, fromFloats)
import Transformer (ParamsSeq, ParamsCSeq, seqLogitsVal, seqGradLoss)
import Optimizer (AdamConfig(..), AdamState(..), Adam, initAdam, adamStep,
                  lrWarmupCosine, chunksOf, shuffle)
import Checkpoint (CkptMeta(..), saveCkpt, loadCkpt)
import MarketTokenizer

-- ── CLI ───────────────────────────────────────────────────────────────────────

data Opts = Opts
  { optMode   :: String
  , optEpochs :: Int
  , optSeed   :: Int
  , optData   :: FilePath
  , optFrac   :: Double
  , optCkpt   :: Maybe FilePath
  , optResume :: Bool
  , optPrompt :: Bool
  , optList   :: Bool
  }

optsP :: O.Parser Opts
optsP = Opts
  <$> O.strOption (O.long "mode" <> O.short 'm' <> O.metavar "MODE"
        <> O.value "mkt-small" <> O.showDefault <> O.help "model mode (--list-modes)")
  <*> O.option O.auto (O.long "epochs" <> O.short 'e' <> O.metavar "N"
        <> O.value 50 <> O.showDefault <> O.help "training epochs")
  <*> O.option O.auto (O.long "seed" <> O.short 's' <> O.metavar "SEED"
        <> O.value 42 <> O.showDefault <> O.help "init/shuffle seed")
  <*> O.strOption (O.long "data" <> O.short 'd' <> O.metavar "CSV"
        <> O.value "data/BTC-1m.csv" <> O.showDefault
        <> O.help "candle CSV from chain-backfill (t,o,h,l,c,v,n)")
  <*> O.option O.auto (O.long "train-frac" <> O.metavar "F"
        <> O.value 0.7 <> O.showDefault <> O.help "walk-forward train fraction")
  <*> O.optional (O.strOption (O.long "checkpoint" <> O.short 'c' <> O.metavar "PATH"
        <> O.help "checkpoint path (default checkpoint-<mode>.ckpt)"))
  <*> O.switch (O.long "resume" <> O.help "resume from the checkpoint")
  <*> O.switch (O.long "prompt" <> O.help "interactive REPL over the checkpoint")
  <*> O.switch (O.long "list-modes" <> O.help "list available modes and exit")

listModesText :: String
listModesText = unlines
  [ "market-train modes (all are the conformance-gated architecture:"
  , "  1 block, causal, 2 heads — ParamsSeq v n dM dF dK):"
  , ""
  , "  mkt-small   128 bins (v=130), n=32,  dM=32, dF=128, dK=8    (~20k params, CPU smoke)"
  , "  mkt-base    512 bins (v=514), n=64,  dM=64, dF=256, dK=16   (~113k params, CPU overnight)"
  , ""
  , "Larger configs (dM 256+, n 256) wait for the GPU tier (README §10 Phase 2)."
  ]

-- ── candle CSV → returns ──────────────────────────────────────────────────────

loadCloses :: FilePath -> IO [Double]
loadCloses path = do
  ok <- doesFileExist path
  unless ok $ do
    hPutStrLn stderr ("market-train: no data file " ++ path
      ++ " — run chain-backfill first (see chain-query README)")
    exitFailure
  s <- readFile path
  let rows   = drop 1 (lines s)            -- skip header
      closeOf r = case splitOn ',' r of
        (_t : _o : _h : _l : c : _) -> readMaybe c :: Maybe Double
        _                           -> Nothing
      closes = mapMaybe closeOf rows
  when (length closes < length rows) $
    hPutStrLn stderr ("market-train: dropped "
      ++ show (length rows - length closes) ++ " malformed rows")
  pure closes
  where
    splitOn c xs = case break (== c) xs of
      (a, [])      -> [a]
      (a, _ : b)   -> a : splitOn c b

-- ── helpers ───────────────────────────────────────────────────────────────────

softmaxL :: [Double] -> [Double]
softmaxL xs = let m = maximum xs; es = map (\x -> exp (x - m)) xs; z = sum es
              in map (/ z) es

-- mean CE of a window under the model (forward only; same value as the loss
-- morphism — the detached max is shift-invariant)
windowCE :: ParamsCSeq v n dM dF dK
         => ParamsSeq v n dM dF dK -> [Int] -> Double
windowCE p toks =
  let logits = seqLogitsVal toks p
      pairs  = zip (init logits) (drop 1 toks)
      ce (lg, t) = let ps = softmaxL (vtoList lg) in negate (log (max 1e-300 (ps !! t)))
  in sum (map ce pairs) / fromIntegral (length pairs)

-- unconditional baseline: CE of the train bin distribution on the val targets
baselineCE :: Int -> [Int] -> [Int] -> Double
baselineCE v trainToks valTargets =
  let counts = Map.fromListWith (+) [ (t, 1 :: Double) | t <- trainToks ]
      total  = fromIntegral (length trainToks) + fromIntegral v   -- add-1 smoothing
      q t    = (Map.findWithDefault 0 t counts + 1) / total
  in negate (sum (map (log . q) valTargets)) / fromIntegral (length valTargets)

-- global-norm gradient clip (harness-side engineering, README §4)
clipGrad :: (Serialize p, Scale p) => Double -> p -> p
clipGrad c g =
  let n = sqrt (sum [ x * x | x <- toFloats g ])
  in if n > c then scaleA (c / n) g else g

uniformN :: Int -> Double -> StdGen -> ([Double], StdGen)
uniformN n b g = go n g []
  where go 0 g' acc = (reverse acc, g')
        go k g' acc = let (x, g'') = Rnd.randomR (-b, b) g' in go (k - 1) g'' (x : acc)

-- flat Xavier init in the exact ParamsSeq Serialize leaf order:
-- tok, pos, h1(Wq,Wk,Wv), h2(Wq,Wk,Wv), Wo, LN1, FFN up, FFN down, LN2, unembed.
initFlatSeq :: Int -> Int -> Int -> Int -> Int -> StdGen -> ([Double], StdGen)
initFlatSeq v n dM dF dK g0 =
  let xav r c = sqrt (6 / fromIntegral (r + c))
      mblock r c g = uniformN (r * c) (xav r c) g
      zeros k = replicate k (0 :: Double)
      ones  k = replicate k (1 :: Double)
      lin o i g = let (w, g') = mblock o i g in (w ++ zeros o, g')
      hd g = let (wq, g1) = lin dK dM g
                 (wk, g2) = lin dK dM g1
                 (wv, g3) = lin dK dM g2
             in (wq ++ wk ++ wv, g3)
      (tok, g1) = mblock v dM g0
      (pos, g2) = mblock n dM g1
      (h1,  g3) = hd g2
      (h2,  g4) = hd g3
      (wo,  g5) = lin dM (2 * dK) g4
      ln1       = ones dM ++ zeros dM
      (up,  g6) = lin dF dM g5
      (dn,  g7) = lin dM dF g6
      ln2       = ones dM ++ zeros dM
      (un,  g8) = lin v dM g7
  in (tok ++ pos ++ h1 ++ h2 ++ wo ++ ln1 ++ up ++ dn ++ ln2 ++ un, g8)

-- ── the generic runner (one code path for every mode) ─────────────────────────

run :: forall v n dM dF dK.
       ( ParamsCSeq v n dM dF dK
       , Serialize (ParamsSeq v n dM dF dK)
       , Adam (ParamsSeq v n dM dF dK)
       , Scale (ParamsSeq v n dM dF dK) )
    => Int -> Int -> Int -> Int -> Int   -- v n dM dF dK (value level)
    -> Opts -> IO ()
run vI nI dMI dFI dKI o = do
  let mode     = optMode o
      ckptPath = fromMaybe ("checkpoint-" ++ mode ++ ".ckpt") (optCkpt o)
      specPath = ckptPath ++ ".tok"
      nBins    = vI - 2
      nParam   = length (toFloats (zeroA :: ParamsSeq v n dM dF dK))
  if optPrompt o then promptMarket @v @n @dM @dF @dK nI ckptPath specPath nParam
  else do
    closes <- loadCloses (optData o)
    let rets             = logReturns closes
        (trainR, valR)   = walkForwardSplit (optFrac o) rets
    when (length trainR < 10 * nI) $ do
      hPutStrLn stderr "market-train: not enough training data for this context length"
      exitFailure
    -- tokenizer: fit on the TRAIN slice only; reuse the saved artifact on resume
    spec <- do
      have <- doesFileExist specPath
      if optResume o && have then loadSpec specPath
      else do
        let s = fitSpec nBins trainR
        saveSpec specPath s
        pure s
    let trainToks = map (encodeReturn spec) trainR
        valToks   = map (encodeReturn spec) valR
        -- each training sequence: ⊥ then n−1 bins (sliding window, stride 1)
        mkWindows ts = map (bosTok :) (contextWindows (nI - 1) ts)
        wTrain  = mkWindows trainToks
        wVal    = take 512 (mkWindows valToks)     -- eval sample (forward-only)
        base    = baselineCE vI trainToks (concatMap (drop 1) wVal)
        batchSz = 32
        epochs  = optEpochs o
        stepsPerEpoch = (length wTrain + batchSz - 1) `div` batchSz
        totalSteps    = epochs * stepsPerEpoch
        adamCfg = AdamConfig 0.9 0.999 1.0e-8 1.0e-3
        evalEvery = 1
        ckptEvery = 5
    printf "market-train %s: %d params, vocab %d (%d bins), n=%d\n" mode nParam vI nBins nI
    printf "data %s: %d candles -> %d train / %d val returns; %d train windows\n"
           (optData o) (length closes) (length trainR) (length valR) (length wTrain)
    printf "baseline (unconditional train distribution) val CE = %.4f (ppl %.1f)\n"
           base (exp base)
    -- init or resume
    let fresh = do
          let g0 = mkStdGen (optSeed o)
              (flat, _) = initFlatSeq vI nI dMI dFI dKI g0
          printf "Fresh init (Xavier), seed %d.\n" (optSeed o)
          pure (fst (fromFloats flat) :: ParamsSeq v n dM dF dK, initAdam, 0 :: Int)
    (pp0, st0, e0) <- if optResume o
      then loadCkpt nParam ckptPath >>= \r -> case r of
        Just (meta, pp, st) -> do
          when (ckMode meta /= mode) $
            hPutStrLn stderr ("WARNING: checkpoint mode " ++ ckMode meta
                              ++ " /= requested " ++ mode)
          printf "Resumed %s at epoch %d (t=%d).\n" ckptPath (ckEpoch meta) (asT st)
          pure (pp, st, ckEpoch meta)
        Nothing -> fresh
      else fresh
    t0 <- getCurrentTime
    let loop !epoch !pp !st !gSh !bestCE
          | epoch > e0 + epochs = pure (pp, st)
          | otherwise = do
              let (order, gSh') = shuffle gSh wTrain
                  batches = chunksOf batchSz order
                  stepBatch (!p, !s) b =
                    let grads = parMap rdeepseq (\w -> fst (seqGradLoss w p)) b
                        gAvg  = clipGrad 1.0
                                  (scaleA (1 / fromIntegral (length b)) (foldl1' addA grads))
                        lr    = lrWarmupCosine 500 totalSteps 1.0e-3 1.0e-4 (asT s + 1)
                    in adamStep adamCfg lr p s gAvg
                  (pp', st') = foldl' stepBatch (pp, st) batches
              best' <- if epoch `mod` evalEvery == 0
                then do
                  ce <- pure $! sum (map (windowCE pp') wVal) / fromIntegral (length wVal)
                  el <- (`diffUTCTime` t0) <$> getCurrentTime
                  printf "epoch %4d | step %6d | val CE %.4f (ppl %6.1f) | best %.4f | base %.4f | %s\n"
                         epoch (asT st') ce (exp ce) (min ce bestCE) base (show el)
                  hFlush stdout
                  when (ce < bestCE) $
                    saveCkpt (ckptPath ++ ".best") (CkptMeta "market" mode epoch) pp' st'
                  pure (min ce bestCE)
                else pure bestCE
              when (epoch `mod` ckptEvery == 0) $
                saveCkpt ckptPath (CkptMeta "market" mode epoch) pp' st'
              loop (epoch + 1) pp' st' gSh' best'
    (ppF, stF) <- loop (e0 + 1) pp0 st0 (snd (split (mkStdGen (optSeed o)))) (1 / 0)
    saveCkpt ckptPath (CkptMeta "market" mode (e0 + epochs)) ppF stF
    printf "Done. Checkpoint %s (+ .best, + .tok tokenizer artifact).\n" ckptPath

-- ── the prompt: talk to a trained model, constrained to its vocabulary ────────

promptMarket :: forall v n dM dF dK.
                ( ParamsCSeq v n dM dF dK
                , Serialize (ParamsSeq v n dM dF dK) )
             => Int -> FilePath -> FilePath -> Int -> IO ()
promptMarket nI ckptPath specPath nParam = do
  r <- loadCkpt nParam ckptPath
  (meta, pp) <- case r of
    Just (meta, pp, _st) -> pure (meta, pp :: ParamsSeq v n dM dF dK)
    Nothing -> do
      hPutStrLn stderr ("prompt: no checkpoint at " ++ ckptPath)
      exitFailure
  haveSpec <- doesFileExist specPath
  unless haveSpec $ do
    hPutStrLn stderr ("prompt: no tokenizer artifact at " ++ specPath)
    exitFailure
  spec <- loadSpec specPath
  printf "market prompt — %s (mode %s, epoch %d), %d bins, context up to %d returns\n"
         ckptPath (ckMode meta) (ckEpoch meta) (tsNBins spec) (nI - 1)
  putStrLn "enter recent returns (space-separated decimals, e.g. `0.001 -0.0007 0.002`); :q quits"
  let go = do
        putStr "> " >> hFlush stdout
        eof <- isEOF
        unless eof $ do
          line <- getLine
          case line of
            ":q" -> pure ()
            _ -> do
              case traverse readMaybe (words line) :: Maybe [Double] of
                Nothing -> putStrLn "  parse error: space-separated decimal returns only" >> go
                Just rs
                  | null rs -> go
                  | length rs > nI - 1 ->
                      putStrLn ("  too many: max " ++ show (nI - 1) ++ " returns") >> go
                  | otherwise -> do
                      let toks   = bosTok : map (encodeReturn spec) rs
                          logits = last (seqLogitsVal toks pp)
                          probs  = softmaxL (vtoList logits)
                          binsP  = [ (t, p') | (t, p') <- zip [0 ..] probs ]
                          top5   = take 5 (sortOn (negate . snd) binsP)
                          expRet = sum [ p' * ret
                                       | (t, p') <- binsP, Just ret <- [decodeTok spec t] ]
                      putStrLn ("  context bins: " ++ show (map (encodeReturn spec) rs))
                      forM_ top5 $ \(t, p') ->
                        case decodeTok spec t of
                          Just ret -> printf "  bin %3d  p=%.4f  ~return %+.5f\n" t p' ret
                          Nothing  -> printf "  %s  p=%.4f\n"
                                        (if t == bosTok then "⊥" else "†" :: String) p'
                      printf "  expected next return = %+.6f\n" expRet
                      go
  go

-- ── dispatch ──────────────────────────────────────────────────────────────────

main :: IO ()
main = do
  o <- O.execParser (O.info (O.helper <*> optsP)
        (O.fullDesc <> O.progDesc "next-token transformer over quantized candle returns"))
  if optList o then putStr listModesText
  else case optMode o of
    "mkt-small" -> run @130 @32 @32 @128 @8  130 32 32 128 8  o
    "mkt-base"  -> run @514 @64 @64 @256 @16 514 64 64 256 16 o
    m -> do
      hPutStrLn stderr ("unknown mode " ++ m)
      putStr listModesText
      exitFailure
