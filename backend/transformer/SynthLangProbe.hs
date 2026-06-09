{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# OPTIONS_GHC -Wno-simplifiable-class-constraints #-}

-- Enriched-semantics probe, Phase B: a synthetic *next-token language* designed so
-- Tai-Danae Bradley's enriched structure is ground-truth, then a test that the
-- trained model realizes it.
--
-- Language ("ℤ/m rotation walk"): tokens 0..m-1; the continuation distribution after
-- a prefix depends only on the running sum s = (Σ tokens) mod m and is a fixed base
-- shape B rotated by s:   D_s(g) = B((g - s) mod m).   By construction:
--   * synonyms  = prefixes with equal sum         (Yoneda: meaning = the copresheaf);
--   * meanings  = the orbit {Rotate_s(B)} ≅ ℤ/m   (the group acts by rotation);
--   * composition: appending a token adds to s, i.e. ROTATES the meaning — the
--     [0,1]-enriched chain rule made concrete (meaning is a monoid homomorphism).
-- Unlike the mod-p probe (Phase A), the copresheaf here is a genuine NON-degenerate
-- distribution, so we can also test the rotation/composition law directly.
--
-- We train a small transformer with SOFT labels = the true D_s (exact supervision,
-- so cross-entropy floors at H(D_s)), then probe.  prefix length 2 = the proven-
-- learnable modular-addition setting; m=12 gives a clean 12-point circle.
module Main where

import Control.Monad (when)
import Control.Monad.ST (ST)
import Data.Complex (Complex(..), magnitude, mkPolar)
import Data.List (foldl', maximumBy)
import Data.Maybe (fromMaybe)
import Data.Ord (comparing)
import GHC.TypeNats (KnownNat)
import qualified Numeric.LinearAlgebra as LA
import qualified Options.Applicative as O
import System.Exit (die)
import System.IO (isEOF)
import System.Random (mkStdGen, randomR, StdGen)
import Text.Printf (printf)
import Text.Read (readMaybe)

import AD (fstL, sndL, (.<))
import Checkpoint (CkptMeta(..), saveCkpt, loadCkpt)
import Optimizer
import Serialize
import Tape
import Tensor
import Transformer (Lin, Block)

-- ── fixed task dimensions ───────────────────────────────────────────────────────
mI, nI, dMI, dFI, dKI :: Int
mI  = 12    -- ℤ/12  (also the vocab size)
nI  = 2     -- prefix length (2-token modular addition)
dMI = 32
dFI = 64
dKI = 32

type SP = (M 12 32, (M 2 32, (Block 32 64 32, Lin 12 32)))   -- tok, pos, block, unembed
type Vm = V 12

-- ── the language ────────────────────────────────────────────────────────────────

baseWeights :: [Double]                       -- circular bump peaked at 0, ±1, ±2
baseWeights = [0.45, 0.22, 0.06] ++ replicate (mI - 5) 0.0 ++ [0.06, 0.22]

baseDist :: [Double]
baseDist = let z = sum baseWeights in map (/ z) baseWeights

rotate :: Int -> [Double] -> [Double]         -- (Rotate_s xs)(g) = xs[(g-s) mod m]
rotate s xs = [ xs !! ((g - s) `mod` mI) | g <- [0 .. mI - 1] ]

trueDist :: Int -> [Double]
trueDist s = rotate s baseDist

stateOf :: [Int] -> Int
stateOf toks = sum toks `mod` mI

allPrefixes :: [[Int]]
allPrefixes = go nI
  where go 0 = [[]]
        go k = [ t : ts | t <- [0 .. mI - 1], ts <- go (k - 1) ]

-- ── init (Xavier; leaf order tok, pos, block, unembed) ──────────────────────────

uniformN :: Int -> Double -> StdGen -> ([Double], StdGen)
uniformN k b g = go k g []
  where go 0 g' acc = (reverse acc, g')
        go j g' acc = let (x, g'') = randomR (-b, b) g' in go (j - 1) g'' (x : acc)

initFlat :: StdGen -> ([Double], StdGen)
initFlat g0 =
  let (tok, g1) = mblock mI dMI g0
      (pos, g2) = mblock nI dMI g1
      (blk, g3) = block g2
      (un,  g4) = lin mI dMI g3
  in (tok ++ pos ++ blk ++ un, g4)
  where
    xav r c = sqrt (6 / fromIntegral (r + c))
    mblock r c g = uniformN (r * c) (xav r c) g
    zeros k = replicate k 0
    ones  k = replicate k 1
    lin o i g = let (w, g') = mblock o i g in (w ++ zeros o, g')
    block g =
      let (wq, g1) = lin dKI dMI g
          (wk, g2) = lin dKI dMI g1
          (wv, g3) = lin dKI dMI g2
          (wo, g4) = lin dMI dKI g3
          ln1      = ones dMI ++ zeros dMI
          (up, g5) = lin dFI dMI g4
          (dn, g6) = lin dMI dFI g5
          ln2      = ones dMI ++ zeros dMI
      in (wq ++ wk ++ wv ++ wo ++ ln1 ++ up ++ dn ++ ln2, g6)

-- ── tape forward (1-layer, full attention, read the last position) ──────────────

affineT :: (KnownNat o, KnownNat i)
        => Tape s p -> R s (M o i) -> R s (V o) -> R s (V i) -> ST s (R s (V o))
affineT tp rw rb rx = tMatvec tp rw rx >>= \wx -> tVadd tp wx rb

layerNormT :: KnownNat d => Tape s p -> R s (V d) -> R s (V d) -> R s (V d) -> ST s (R s (V d))
layerNormT tp rg rb rx = do
  xc <- tCenter tp rx; sq <- tSquareV tp xc; var <- tMeanV tp sq
  var' <- tAddC tp 1.0e-5 var; inv <- tRsqrt tp var'; norm <- tScaleV tp inv xc
  gn <- tHadamard tp rg norm; tVadd tp gn rb

sumR :: Tape s p -> [R s Double] -> ST s (R s Double)
sumR tp (x : xs) = foldl' (\m y -> m >>= \a -> tAdd tp a y) (pure x) xs
sumR _  []       = error "sumR: empty"

sumV :: KnownNat d => Tape s p -> [R s (V d)] -> ST s (R s (V d))
sumV tp (x : xs) = foldl' (\m y -> m >>= \a -> tVadd tp a y) (pure x) xs
sumV _  []       = error "sumV: empty"

reprLast :: forall s. Tape s SP -> [Int] -> SP -> ST s (R s (V 32))
reprLast tp toks p = do
  let lTok = fstL; lPos = sndL .< fstL
      lB    = sndL .< sndL .< fstL
      lAttn = lB .< fstL; lLn1 = lB .< sndL .< fstL
      lFfn  = lB .< sndL .< sndL .< fstL; lLn2 = lB .< sndL .< sndL .< sndL
  rTok <- tInput tp lTok p
  rPos <- tInput tp lPos p
  xs <- sequence [ do et <- tEmbedRow tp t rTok; ep <- tEmbedRow tp i rPos; tVadd tp et ep
                 | (i, t) <- zip [0 ..] toks ]
  qW <- tInput tp (lAttn .< fstL .< fstL .< fstL) p
  qB <- tInput tp (lAttn .< fstL .< fstL .< sndL) p
  kW <- tInput tp (lAttn .< fstL .< sndL .< fstL) p
  kB <- tInput tp (lAttn .< fstL .< sndL .< sndL) p
  vW <- tInput tp (lAttn .< sndL .< fstL .< fstL) p
  vB <- tInput tp (lAttn .< sndL .< fstL .< sndL) p
  oW <- tInput tp (lAttn .< sndL .< sndL .< fstL) p
  oB <- tInput tp (lAttn .< sndL .< sndL .< sndL) p
  g1 <- tInput tp (lLn1 .< fstL) p; b1 <- tInput tp (lLn1 .< sndL) p
  upW <- tInput tp (lFfn .< fstL .< fstL) p; upB <- tInput tp (lFfn .< fstL .< sndL) p
  dnW <- tInput tp (lFfn .< sndL .< fstL) p; dnB <- tInput tp (lFfn .< sndL .< sndL) p
  g2 <- tInput tp (lLn2 .< fstL) p; b2 <- tInput tp (lLn2 .< sndL) p
  ks <- mapM (affineT tp kW kB) xs
  vs <- mapM (affineT tp vW vB) xs
  let sc = 1.0 / sqrt (fromIntegral dKI)
  q <- affineT tp qW qB (last xs)            -- read the LAST position
  rawScores <- mapM (tVdot tp q) ks
  scores <- mapM (tScaleC tp sc) rawScores
  let sm = maximum (map primalR scores)
  shifted <- mapM (tAddC tp (negate sm)) scores
  exps <- mapM (tExp tp) shifted
  z <- sumR tp exps; rz <- tRecip tp z
  weights <- mapM (\e -> tMul tp e rz) exps
  pieces <- sequence [ tScaleV tp w v | (w, v) <- zip weights vs ]
  att <- sumV tp pieces
  ao <- affineT tp oW oB att
  r1 <- tVadd tp (last xs) ao
  n1 <- layerNormT tp g1 b1 r1
  hh <- affineT tp upW upB n1; hr <- tReluV tp hh; ff <- affineT tp dnW dnB hr
  r2 <- tVadd tp n1 ff
  layerNormT tp g2 b2 r2
  where
    -- attention sub-lenses below are relative to lAttn (= Attn = ((Wq,Wk),(Wv,Wo)));
    -- qW/qB live under fstL.<fstL, kW/kB under fstL.<sndL — handled inline above.

forwardLogits :: forall s. Tape s SP -> [Int] -> SP -> ST s (R s Vm)
forwardLogits tp toks p = do
  rep <- reprLast tp toks p
  let lUn = sndL .< sndL .< sndL
  uW <- tInput tp (lUn .< fstL) p; uB <- tInput tp (lUn .< sndL) p
  affineT tp uW uB rep

-- soft cross-entropy to the true continuation distribution d:
--   lse(logits) - <d, logits>   (minimised at logits ↦ log d, value = H(d))
lossSoftT :: Tape s p -> [Double] -> R s Vm -> ST s (R s Double)
lossSoftT tp d logits = do
  shifted <- tDetachMax tp logits
  e <- tExpV tp shifted; s <- tVsum tp e; lse <- tLog tp s
  dC <- tConst tp (vfromList d :: Vm)
  wsum <- tVdot tp dC shifted              -- <d, logits - max>; the max cancels in lse - wsum
  tSub tp lse wsum

gradLoss :: [Int] -> SP -> (SP, Double)
gradLoss toks p = tGradLoss (\tp -> forwardLogits tp toks p >>= lossSoftT tp (trueDist (stateOf toks)))

logitsOf :: [Int] -> SP -> Vm
logitsOf toks p = tEval (\tp -> forwardLogits tp toks p)

reprOf :: [Int] -> SP -> V 32
reprOf toks p = tEval (\tp -> reprLast tp toks p)

-- ── training (full data, soft labels) ───────────────────────────────────────────

batchGrad :: SP -> [[Int]] -> (SP, Double)
batchGrad p batch =
  let (g, l) = foldl' step (zeroA, 0) batch
      k = fromIntegral (max 1 (length batch))
  in (scaleA (1 / k) g, l / k)
  where step (!ga, !la) toks = let (gx, lx) = gradLoss toks p in (addA ga gx, la + lx)

train :: FilePath -> Int -> Int -> Double -> Double -> Int -> SP -> AdamState SP -> [[Int]] -> IO SP
train ckpt epochs batch lr wd startEp p0 st0 dat = go startEp p0 st0 (mkStdGen (7 + startEp))
  where
    adamCfg = AdamConfig 0.9 0.999 1.0e-8 wd
    go e p st g
      | e > epochs = do saveCkpt ckpt (CkptMeta "synth" "synth" (e - 1)) p st; pure p
      | otherwise = do
          let (shuf, g') = shuffle g dat
              batches = chunksOf batch shuf
              (p', st', lsum) = foldl' stepB (p, st, 0) batches
              stepB (!pp, !ss, !ls) b =
                let (gAvg, l) = batchGrad pp b
                    (pp', ss') = adamStep adamCfg lr pp ss gAvg
                in (pp', ss', ls + l)
              avgL = lsum / fromIntegral (max 1 (length batches))
          (toFloats p' `seq` pure ()) :: IO ()
          if e == 1 || e `mod` 50 == 0 || e == epochs
            then printf "  epoch %4d | soft-CE %.4f\n" e avgL else pure ()
          when (e `mod` 50 == 0) $ saveCkpt ckpt (CkptMeta "synth" "synth" e) p' st'
          go (e + 1) p' st' g'

-- Interactive REPL: enter an n-token prefix in {0..m-1}; print the model's
-- next-token distribution, its argmax, and the true D_s for comparison.
promptSynth :: FilePath -> Int -> IO ()
promptSynth ckpt nParam = do
  loaded <- loadCkpt nParam ckpt
  case loaded of
    Nothing -> die $ "No checkpoint at " ++ ckpt ++ " — train first (run without --prompt)."
    Just (meta, ps, _st) -> do
      printf "Loaded %s (mode %s, epoch %d).\n" ckpt (ckMode meta) (ckEpoch meta)
      printf "Prompt: enter %d tokens in {0..%d} (e.g. \"3 5\"); :q to quit.\n" nI (mI - 1)
      let r3 x = (fromIntegral (round (x * 1000) :: Int) / 1000) :: Double
          repl = do
            eof <- isEOF
            if eof then pure () else do
              line <- getLine
              if line == ":q" then pure () else do
                case mapM readMaybe (words line) :: Maybe [Int] of
                  Just toks | length toks == nI && all (\t -> t >= 0 && t < mI) toks -> do
                    let dist  = softmax (vtoList (logitsOf toks (ps :: SP)))
                        s     = sum toks `mod` mI
                        predI = snd (maximumBy (comparing fst) (zip dist [0 :: Int ..]))
                    printf "  state s=%d   next-token argmax=%d (p=%.3f)\n" s predI (dist !! predI)
                    printf "    model D : %s\n" (show (map r3 dist))
                    printf "    true  D_s: %s   KL(true‖model)=%.4g\n"
                      (show (map r3 (trueDist s))) (klDiv (trueDist s) dist)
                  _ -> printf "  expected %d integers in 0..%d; :q to quit\n" nI (mI - 1)
                repl
      repl

-- ── numeric helpers ──────────────────────────────────────────────────────────────

softmax :: [Double] -> [Double]
softmax xs = let m = maximum xs; es = map (\x -> exp (x - m)) xs; z = sum es in map (/ z) es

klDiv :: [Double] -> [Double] -> Double
klDiv ps qs = sum [ if p <= 0 then 0 else p * log (p / max q 1e-12) | (p, q) <- zip ps qs ]

entropy :: [Double] -> Double
entropy ps = negate (sum [ if p <= 0 then 0 else p * log p | p <- ps ])

mean :: [Double] -> Double
mean xs = sum xs / fromIntegral (length xs)

meanVec :: [[Double]] -> [Double]
meanVec vss = map (/ fromIntegral (length vss)) (foldl1 (zipWith (+)) vss)

sqDist :: [Double] -> [Double] -> Double
sqDist a b = sum [ (x - y) * (x - y) | (x, y) <- zip a b ]

-- ── the probes ──────────────────────────────────────────────────────────────────

probe :: SP -> IO ()
probe p = do
  let prefixes = allPrefixes
      copre toks = softmax (vtoList (logitsOf toks p))
      states = [0 .. mI - 1]
      byState f = [ [ f toks | toks <- prefixes, stateOf toks == s ] | s <- states ]
      copsByState = byState copre
      cenCop = map meanVec copsByState
      reprByState = byState (\t -> vtoList (reprOf t p))
      cenRep = map meanVec reprByState
      gMean = meanVec (concat reprByState)

  printf "════════════════════════════════════════════════════════════════════════\n"
  printf " Synthetic-language enriched probe — ℤ/%d rotation walk, prefix length %d\n" mI nI
  printf "════════════════════════════════════════════════════════════════════════\n"
  printf " Language: next token ~ D_s(g)=B((g-s) mod %d), s=(Σ prefix) mod %d; %d prefixes.\n"
    mI mI (length prefixes)
  printf " base B ≈ %s  (entropy floor H(B)=%.4f nats)\n\n"
    (show (map (\x -> fromIntegral (round (x*100)::Int)/100) baseDist)) (entropy baseDist)

  -- fidelity: did it learn the true continuation distributions?
  let fidKL = mean [ klDiv (trueDist s) (cenCop !! s) | s <- states ]
      ce    = mean [ klDiv (trueDist (stateOf t)) (copre t) + entropy (trueDist (stateOf t)) | t <- prefixes ]
  printf " Fidelity: mean KL(true D_s ‖ learned)=%.4g   soft cross-entropy=%.4f (floor %.4f)\n\n"
    fidKL ce (entropy baseDist)

  -- Metric 1 — Yoneda synonymy collapse
  let intraKL = mean [ klDiv c cen | (cs, cen) <- zip copsByState cenCop, c <- cs ]
      interKL = mean [ klDiv (cenCop !! i) (cenCop !! j) | i <- states, j <- states, i /= j ]
      ssW = sum [ sqDist r cen | (rs, cen) <- zip reprByState cenRep, r <- rs ]
      ssT = sum [ sqDist r gMean | rs <- reprByState, r <- rs ]
      varExpl = 1 - ssW / ssT
  printf " Metric 1 — Yoneda synonymy collapse (equal-sum prefixes are synonyms)\n"
  printf "   copresheaf KL: intra-state=%.4g  inter-state=%.4g  ratio=%.1fx\n"
    intraKL interKL (interKL / max intraKL 1e-12)
  printf "   meaning-vector variance explained by state: %.4f  (1.0 = perfect)\n\n" varExpl

  -- Metric 2 — composition / group action: meaning(s) = Rotate_s(meaning(0))
  let rotResid = mean [ klDiv (cenCop !! s) (rotate s (head cenCop)) | s <- states ]
      centered = [ zipWith (-) c gMean | c <- cenRep ]
      matA = LA.fromLists centered :: LA.Matrix Double
      (_u, _s, vmat) = LA.svd matA
      proj = matA LA.<> LA.takeColumns 2 vmat
      pts = [ (LA.atIndex r 0, LA.atIndex r 1) | r <- LA.toRows proj ]
      zs = [ x :+ y | (x, y) <- pts ]
      energy k = magnitude (sum [ z * mkPolar 1 (-2*pi*fromIntegral (k*c)/fromIntegral mI)
                                | (c, z) <- zip [0 ..] zs ]) ^ (2 :: Int)
      ks = [1 .. mI - 1]
      domK = maximumBy (comparing energy) ks
      domShare = energy domK / sum (map energy ks)
  printf " Metric 2 — composition: meaning(s) = Rotate_s(meaning(0))  (the group acts)\n"
  printf "   rotation residual: mean KL(learned D_s ‖ rotate_s(learned D_0)) = %.4g  (0 = exact)\n" rotResid
  printf "   meaning-space DFT: dominant frequency k=%d  energy share=%.1f%%  (1 peak ⇒ circular)\n"
    domK (100 * domShare)

-- ── CLI / main ──────────────────────────────────────────────────────────────────

data Opts = Opts
  { optEpochs :: Int, optBatch :: Int, optLR :: Double, optWD :: Double, optSeed :: Int
  , optCkpt :: Maybe FilePath, optResume :: Bool, optPrompt :: Bool, optList :: Bool }

listModesText :: String
listModesText = unlines
  [ "synth-lang-probe has a single built-in language (no -m selector):"
  , "  ℤ/" ++ show mI ++ " rotation walk, prefix length " ++ show nI
  , "Train with --epochs/--batch/--lr/--weight-decay/--seed; query with --prompt."
  ]

optsP :: O.Parser Opts
optsP = Opts
  <$> O.option O.auto (O.long "epochs" <> O.short 'e' <> O.value 500 <> O.showDefault <> O.metavar "N")
  <*> O.option O.auto (O.long "batch" <> O.short 'b' <> O.value 32 <> O.showDefault <> O.metavar "N")
  <*> O.option O.auto (O.long "lr" <> O.value 2.0e-3 <> O.showDefault <> O.metavar "LR")
  <*> O.option O.auto (O.long "weight-decay" <> O.value 1.0e-4 <> O.showDefault <> O.metavar "WD")
  <*> O.option O.auto (O.long "seed" <> O.short 's' <> O.value 42 <> O.showDefault <> O.metavar "N")
  <*> O.optional (O.strOption (O.long "checkpoint" <> O.short 'c' <> O.metavar "PATH"
        <> O.help "checkpoint file (default checkpoint-synth.ckpt)"))
  <*> O.switch (O.long "resume" <> O.help "resume training from the checkpoint if it exists")
  <*> O.switch (O.long "prompt" <> O.help "load the checkpoint and query the model interactively (no training)")
  <*> O.switch (O.long "list-modes" <> O.help "list all modes and exit")

main :: IO ()
main = do
  o <- O.execParser $ O.info (optsP O.<**> O.helper)
    (O.fullDesc <> O.header "synth-lang-probe — enriched semantics on a synthetic next-token language")
  let p0fresh = fst (fromFloats (fst (initFlat (mkStdGen (optSeed o))))) :: SP
      nParam  = length (toFloats p0fresh)
      ckpt    = fromMaybe "checkpoint-synth.ckpt" (optCkpt o)
  if optList o then putStr listModesText
  else if optPrompt o
    then promptSynth ckpt nParam
    else do
      (p0, st0, startEp) <- if optResume o
        then do
          l <- loadCkpt nParam ckpt
          case l of
            Just (m, p, st) -> do printf "Resumed %s at epoch %d\n" ckpt (ckEpoch m); pure (p, st, ckEpoch m + 1)
            Nothing -> pure (p0fresh, initAdam, 1)
        else pure (p0fresh, initAdam, 1)
      printf "Training on %d prefixes (ℤ/%d rotation walk, length %d), soft labels…\n"
        (length allPrefixes) mI nI
      p <- train ckpt (optEpochs o) (optBatch o) (optLR o) (optWD o) startEp p0 st0 allPrefixes
      printf "\n"
      probe p
