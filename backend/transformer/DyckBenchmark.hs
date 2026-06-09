{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE BangPatterns #-}
{-# OPTIONS_GHC -Wno-simplifiable-class-constraints #-}

-- Harder algorithmic benchmark than modular arithmetic: fixed-length Dyck-1
-- balanced-parentheses classification.  This is intentionally separate from the
-- Agda-conformant modular trainer so architecture experiments can move faster.
module Main where

import Control.DeepSeq (NFData, deepseq)
import Control.Monad (foldM, when)
import Control.Monad.ST (ST)
import Data.Char (isSpace)
import Data.List (foldl', intercalate, nub, sort)
import Data.Maybe (fromMaybe)
import Data.Proxy (Proxy(..))
import GHC.TypeNats (KnownNat, natVal)
import qualified Options.Applicative as O
import System.Exit (die)
import System.IO (isEOF)
import System.Random (StdGen, mkStdGen, randomR)
import Text.Printf (printf)

import AD (Lens, fstL, sndL, (.<))
import Checkpoint (CkptMeta(..), saveCkpt, loadCkpt)
import Optimizer
import Serialize
import Tape
import Tensor
import Transformer (Lin, LN, Attn, FFN, Block)

type Example = ([Int], Int) -- token sequence including CLS, target class 0/1

data EvalFormat = Compact | Full deriving Eq

type SeqParams1 n v dM dF dK =
  ( M v dM
  , ( M n dM
  , ( Block dM dF dK
  ,   Lin 2 dM )))

type SeqParams2 n v dM dF dK =
  ( M v dM
  , ( M n dM
  , ( Block dM dF dK
  , ( Block dM dF dK
  ,   Lin 2 dM ))))

type SeqC n v dM dF dK p =
  ( KnownNat n, KnownNat v, KnownNat dM, KnownNat dF, KnownNat dK
  , Additive (M v dM), Additive (M n dM), Additive (V 2)
  , Additive (M dK dM), Additive (V dK), Additive (M dM dK), Additive (V dM)
  , Additive (M dF dM), Additive (V dF), Additive (M dM dF), Additive (M 2 dM)
  , Additive p, Scale p, Adam p, Serialize p, NFData p )

data BenchCfg = BenchCfg
  { bEpochs :: !Int
  , bBatch  :: !Int
  , bFrac   :: !Double
  , bSeed   :: !Int
  , bLR     :: !Double
  , bWD     :: !Double
  , bEvalEv :: !Int
  , bFormat :: !EvalFormat
  , bCkpt   :: !FilePath   -- checkpoint path (latest written here; best to <path>.best)
  , bResume :: !Bool       -- resume training from bCkpt if it exists
  , bPrompt :: !Bool       -- load bCkpt and classify strings interactively (no training)
  , bMode   :: !String     -- the -m mode string (recorded in the checkpoint header)
  }

uniformN :: Int -> Double -> StdGen -> ([Double], StdGen)
uniformN n b g = go n g []
  where
    go 0 g' acc = (reverse acc, g')
    go k g' acc = let (x, g'') = randomR (-b, b) g' in go (k - 1) g'' (x : acc)

initFlatSeq1 :: Int -> Int -> Int -> Int -> Int -> StdGen -> ([Double], StdGen)
initFlatSeq1 n v dM dF dK g0 =
  let (tok, g1) = mblock v dM g0
      (pos, g2) = mblock n dM g1
      (b1,  g3) = block g2
      (un,  g4) = lin 2 dM g3
  in (tok ++ pos ++ b1 ++ un, g4)
  where
    xav r c = sqrt (6 / fromIntegral (r + c))
    mblock r c g = uniformN (r * c) (xav r c) g
    zeros k = replicate k 0
    ones  k = replicate k 1
    lin o i g = let (w, g') = mblock o i g in (w ++ zeros o, g')
    block g =
      let (wq, g1) = lin dK dM g
          (wk, g2) = lin dK dM g1
          (wv, g3) = lin dK dM g2
          (wo, g4) = lin dM dK g3
          ln1      = ones dM ++ zeros dM
          (up, g5) = lin dF dM g4
          (dn, g6) = lin dM dF g5
          ln2      = ones dM ++ zeros dM
      in (wq ++ wk ++ wv ++ wo ++ ln1 ++ up ++ dn ++ ln2, g6)

initFlatSeq2 :: Int -> Int -> Int -> Int -> Int -> StdGen -> ([Double], StdGen)
initFlatSeq2 n v dM dF dK g0 =
  let (tok, g1) = mblock v dM g0
      (pos, g2) = mblock n dM g1
      (b1,  g3) = block g2
      (b2,  g4) = block g3
      (un,  g5) = lin 2 dM g4
  in (tok ++ pos ++ b1 ++ b2 ++ un, g5)
  where
    xav r c = sqrt (6 / fromIntegral (r + c))
    mblock r c g = uniformN (r * c) (xav r c) g
    zeros k = replicate k 0
    ones  k = replicate k 1
    lin o i g = let (w, g') = mblock o i g in (w ++ zeros o, g')
    block g =
      let (wq, g1) = lin dK dM g
          (wk, g2) = lin dK dM g1
          (wv, g3) = lin dK dM g2
          (wo, g4) = lin dM dK g3
          ln1      = ones dM ++ zeros dM
          (up, g5) = lin dF dM g4
          (dn, g6) = lin dM dF g5
          ln2      = ones dM ++ zeros dM
      in (wq ++ wk ++ wv ++ wo ++ ln1 ++ up ++ dn ++ ln2, g6)

affineT :: (KnownNat o, KnownNat i)
        => Tape s p -> R s (M o i) -> R s (V o) -> R s (V i) -> ST s (R s (V o))
affineT tp rw rb rx = tMatvec tp rw rx >>= \wx -> tVadd tp wx rb

layerNormT :: KnownNat dM
           => Tape s p -> R s (V dM) -> R s (V dM) -> R s (V dM) -> ST s (R s (V dM))
layerNormT tp rg rb rx = do
  xc   <- tCenter tp rx
  sq   <- tSquareV tp xc
  var  <- tMeanV tp sq
  var' <- tAddC tp 1.0e-5 var
  inv  <- tRsqrt tp var'
  norm <- tScaleV tp inv xc
  gn   <- tHadamard tp rg norm
  tVadd tp gn rb

sumR :: Tape s p -> [R s Double] -> ST s (R s Double)
sumR _  []       = error "sumR: empty list"
sumR tp (x : xs) = foldM (tAdd tp) x xs

sumV :: KnownNat dM => Tape s p -> [R s (V dM)] -> ST s (R s (V dM))
sumV _  []       = error "sumV: empty list"
sumV tp (x : xs) = foldM (tVadd tp) x xs

blockSeq :: forall s p dM dF dK. (KnownNat dM, KnownNat dF, KnownNat dK)
         => Tape s p -> p
         -> Lens p (Attn dM dK) -> Lens p (LN dM) -> Lens p (FFN dM dF) -> Lens p (LN dM)
         -> [Int] -> [R s (V dM)] -> ST s [R s (V dM)]
blockSeq tp p lAttn lLn1 lFfn lLn2 toks xs = do
  qW <- tInput tp (lAttn .< fstL .< fstL .< fstL) p
  qB <- tInput tp (lAttn .< fstL .< fstL .< sndL) p
  kW <- tInput tp (lAttn .< fstL .< sndL .< fstL) p
  kB <- tInput tp (lAttn .< fstL .< sndL .< sndL) p
  vW <- tInput tp (lAttn .< sndL .< fstL .< fstL) p
  vB <- tInput tp (lAttn .< sndL .< fstL .< sndL) p
  oW <- tInput tp (lAttn .< sndL .< sndL .< fstL) p
  oB <- tInput tp (lAttn .< sndL .< sndL .< sndL) p
  g1 <- tInput tp (lLn1 .< fstL) p
  b1 <- tInput tp (lLn1 .< sndL) p
  upW <- tInput tp (lFfn .< fstL .< fstL) p
  upB <- tInput tp (lFfn .< fstL .< sndL) p
  dnW <- tInput tp (lFfn .< sndL .< fstL) p
  dnB <- tInput tp (lFfn .< sndL .< sndL) p
  g2 <- tInput tp (lLn2 .< fstL) p
  b2 <- tInput tp (lLn2 .< sndL) p
  qs <- mapM (affineT tp qW qB) xs
  ks <- mapM (affineT tp kW kB) xs
  vs <- mapM (affineT tp vW vB) xs
  let sc = 1.0 / sqrt (fromIntegral (natVal (Proxy @dK)))
      keyIsReal = map (/= 3) toks
      finish q xres = do
        rawScores <- mapM (tVdot tp q) ks
        scores <- mapM (tScaleC tp sc) rawScores
        maskedScores <- sequence [ if keep then pure score else tConst tp (-1.0e9)
                                 | (score, keep) <- zip scores keyIsReal ]
        let sm = maximum (map primalR maskedScores)
        shifted <- mapM (tAddC tp (negate sm)) maskedScores
        exps <- mapM (tExp tp) shifted
        z <- sumR tp exps
        rz <- tRecip tp z
        weights <- mapM (\e -> tMul tp e rz) exps
        pieces <- sequence [ tScaleV tp w v | (w, v) <- zip weights vs ]
        att <- sumV tp pieces
        ao <- affineT tp oW oB att
        r1 <- tVadd tp xres ao
        n1 <- layerNormT tp g1 b1 r1
        hh <- affineT tp upW upB n1
        hr <- tReluV tp hh
        ff <- affineT tp dnW dnB hr
        r2 <- tVadd tp n1 ff
        layerNormT tp g2 b2 r2
  sequence [ finish q x | (q, x) <- zip qs xs ]

forward1 :: forall s n v dM dF dK. SeqC n v dM dF dK (SeqParams1 n v dM dF dK)
         => Tape s (SeqParams1 n v dM dF dK) -> [Int] -> SeqParams1 n v dM dF dK -> ST s (R s (V 2))
forward1 tp toks p = do
  let lTok = fstL
      lPos = sndL .< fstL
      lB1  = sndL .< sndL .< fstL
      lUn  = sndL .< sndL .< sndL
  xs <- embedSeq tp p lTok lPos toks
  hs <- blockSeq tp p (lB1 .< fstL) (lB1 .< sndL .< fstL)
                      (lB1 .< sndL .< sndL .< fstL) (lB1 .< sndL .< sndL .< sndL) toks xs
  uW <- tInput tp (lUn .< fstL) p
  uB <- tInput tp (lUn .< sndL) p
  case hs of
    cls : _ -> affineT tp uW uB cls
    []      -> error "forward1: empty sequence"

forward2 :: forall s n v dM dF dK. SeqC n v dM dF dK (SeqParams2 n v dM dF dK)
         => Tape s (SeqParams2 n v dM dF dK) -> [Int] -> SeqParams2 n v dM dF dK -> ST s (R s (V 2))
forward2 tp toks p = do
  let lTok = fstL
      lPos = sndL .< fstL
      lB1  = sndL .< sndL .< fstL
      lB2  = sndL .< sndL .< sndL .< fstL
      lUn  = sndL .< sndL .< sndL .< sndL
  xs <- embedSeq tp p lTok lPos toks
  h1 <- blockSeq tp p (lB1 .< fstL) (lB1 .< sndL .< fstL)
                       (lB1 .< sndL .< sndL .< fstL) (lB1 .< sndL .< sndL .< sndL) toks xs
  h2 <- blockSeq tp p (lB2 .< fstL) (lB2 .< sndL .< fstL)
                       (lB2 .< sndL .< sndL .< fstL) (lB2 .< sndL .< sndL .< sndL) toks h1
  uW <- tInput tp (lUn .< fstL) p
  uB <- tInput tp (lUn .< sndL) p
  case h2 of
    cls : _ -> affineT tp uW uB cls
    []      -> error "forward2: empty sequence"

embedSeq :: forall s p n v dM. (KnownNat n, KnownNat v, KnownNat dM)
         => Tape s p -> p -> Lens p (M v dM) -> Lens p (M n dM) -> [Int] -> ST s [R s (V dM)]
embedSeq tp p lTok lPos toks
  | length toks /= nI = error $ "embedSeq: expected " ++ show nI ++ " tokens, got " ++ show (length toks)
  | otherwise = do
      rTok <- tInput tp lTok p
      rPos <- tInput tp lPos p
      sequence [ do et <- tEmbedRow tp tok rTok
                    ep <- tEmbedRow tp i rPos
                    tVadd tp et ep
               | (i, tok) <- zip [0 ..] toks ]
  where
    nI = fromIntegral (natVal (Proxy @n)) :: Int

lossFromLogits :: Tape s p -> Int -> R s (V 2) -> ST s (R s Double)
lossFromLogits tp target logits = do
  shifted <- tDetachMax tp logits
  e <- tExpV tp shifted
  s <- tVsum tp e
  lse <- tLog tp s
  sel <- tSelect tp target shifted
  tSub tp lse sel

gradLoss1 :: forall n v dM dF dK. SeqC n v dM dF dK (SeqParams1 n v dM dF dK)
          => [Int] -> Int -> SeqParams1 n v dM dF dK -> (SeqParams1 n v dM dF dK, Double)
gradLoss1 toks target p = tGradLoss (\tp -> forward1 tp toks p >>= lossFromLogits tp target)

gradLoss2 :: forall n v dM dF dK. SeqC n v dM dF dK (SeqParams2 n v dM dF dK)
          => [Int] -> Int -> SeqParams2 n v dM dF dK -> (SeqParams2 n v dM dF dK, Double)
gradLoss2 toks target p = tGradLoss (\tp -> forward2 tp toks p >>= lossFromLogits tp target)

logits1 :: forall n v dM dF dK. SeqC n v dM dF dK (SeqParams1 n v dM dF dK)
        => [Int] -> SeqParams1 n v dM dF dK -> V 2
logits1 toks p = tEval (\tp -> forward1 tp toks p)

logits2 :: forall n v dM dF dK. SeqC n v dM dF dK (SeqParams2 n v dM dF dK)
        => [Int] -> SeqParams2 n v dM dF dK -> V 2
logits2 toks p = tEval (\tp -> forward2 tp toks p)

argmaxList :: [Double] -> Int
argmaxList [] = error "argmaxList: empty list"
argmaxList (x : xs) = snd (foldl' step (x, 0) (zip xs [1 ..]))
  where
    step (best, bi) (x, i) = if x > best then (x, i) else (best, bi)

data Metrics = Metrics
  { mTotal   :: !Int
  , mCorrect :: !Int
  , mTP      :: !Int
  , mTN      :: !Int
  , mFP      :: !Int
  , mFN      :: !Int
  }

zeroMetrics :: Metrics
zeroMetrics = Metrics 0 0 0 0 0 0

effectiveLen :: [Int] -> Int
effectiveLen = length . filter (`elem` [1, 2])

metrics :: (p -> [Int] -> V 2) -> p -> [Example] -> Metrics
metrics logits p xs = metricsFromPredictions
  [ (y, argmaxList (vtoList (logits p toks))) | (toks, y) <- xs ]

metricsFromPredictions :: [(Int, Int)] -> Metrics
metricsFromPredictions = foldl' step zeroMetrics
  where
    step (Metrics total correct tp tn fp fn) (y, predY) =
      let correct' = correct + if predY == y then 1 else 0
      in case (y, predY) of
           (1, 1) -> Metrics (total + 1) correct' (tp + 1) tn fp fn
           (0, 0) -> Metrics (total + 1) correct' tp (tn + 1) fp fn
           (0, 1) -> Metrics (total + 1) correct' tp tn (fp + 1) fn
           (1, 0) -> Metrics (total + 1) correct' tp tn fp (fn + 1)
           _      -> error "metrics: class labels must be 0 or 1"

metricsAccuracy :: Metrics -> Double
metricsAccuracy (Metrics total correct _ _ _ _) =
  fromIntegral correct / fromIntegral (max 1 total)

validAccuracy :: Metrics -> Double
validAccuracy (Metrics _ _ tp _ _ fn) =
  fromIntegral tp / fromIntegral (max 1 (tp + fn))

invalidAccuracy :: Metrics -> Double
invalidAccuracy (Metrics _ _ _ tn fp _) =
  fromIntegral tn / fromIntegral (max 1 (tn + fp))

metricsByLength :: (p -> [Int] -> V 2) -> p -> [Example] -> [(Int, Metrics)]
metricsByLength logits p xs = snd (testDiagnostics logits p xs)

testDiagnostics :: (p -> [Int] -> V 2) -> p -> [Example] -> (Metrics, [(Int, Metrics)])
testDiagnostics logits p xs =
  let scored = [ (effectiveLen toks, y, argmaxList (vtoList (logits p toks))) | (toks, y) <- xs ]
      lens = sort (nub [ n | (n, _, _) <- scored ])
      allM = metricsFromPredictions [ (y, predY) | (_, y, predY) <- scored ]
      byLen = [ (n, metricsFromPredictions [ (y, predY) | (n', y, predY) <- scored, n' == n ])
              | n <- lens ]
  in (allM, byLen)

formatLengthMetrics :: [(Int, Metrics)] -> String
formatLengthMetrics xs = intercalate " "
  [ printf "len%d=%.1f%%" n (100 * metricsAccuracy m) | (n, m) <- xs ]

formatLengthRow :: [(Int, Metrics)] -> String
formatLengthRow xs = intercalate " | "
  [ printf "%5.1f" (100 * metricsAccuracy m) | (_, m) <- xs ]

formatLengthHeader :: [Int] -> String
formatLengthHeader lens = intercalate " | " [ printf "len%-2d" n | n <- lens ]

data EvalResult = EvalResult
  { erEpoch  :: !Int
  , erStep   :: !Int
  , erLoss   :: !Double
  , erTrain  :: !Double
  , erTestM  :: !Metrics
  , erByLen  :: ![(Int, Metrics)]
  }

erTest :: EvalResult -> Double
erTest = metricsAccuracy . erTestM

printFullEvalWithLR :: Double -> EvalResult -> IO ()
printFullEvalWithLR lr r = do
  printf "%7d | %9d | %8.6f | %8.4f | %7.1f | %7.1f\n"
    (erEpoch r) (erStep r) lr (erLoss r) (100 * erTrain r) (100 * erTest r)
  printf "        test class : valid=%5.1f%% invalid=%5.1f%% tp=%d tn=%d fp=%d fn=%d\n"
    (100 * validAccuracy (erTestM r)) (100 * invalidAccuracy (erTestM r))
    (mTP (erTestM r)) (mTN (erTestM r)) (mFP (erTestM r)) (mFN (erTestM r))
  printf "        test length: %s\n" (formatLengthMetrics (erByLen r))

printCompactEval :: EvalResult -> Double -> Bool -> IO ()
printCompactEval r bestTest isNewBest =
  printf "%7d | %9d | %8.4f | %5.1f | %5.1f | %5.1f | %5.1f | %7.1f | %s%s\n"
    (erEpoch r) (erStep r) (erLoss r) (100 * erTrain r) (100 * erTest r) (100 * bestTest)
    (100 * validAccuracy (erTestM r)) (100 * invalidAccuracy (erTestM r))
    (formatLengthRow (erByLen r)) (if isNewBest then " *" else "")

printSummary :: String -> EvalResult -> IO ()
printSummary label r = do
  printf " %s: epoch=%d step=%d test=%.1f%% train=%.1f%% valid=%.1f%% invalid=%.1f%% loss=%.4f\n"
    label (erEpoch r) (erStep r) (100 * erTest r) (100 * erTrain r)
    (100 * validAccuracy (erTestM r)) (100 * invalidAccuracy (erTestM r)) (erLoss r)
  printf "        lengths: %s\n" (formatLengthMetrics (erByLen r))

accuracy :: (p -> [Int] -> V 2) -> p -> [Example] -> Double
accuracy _ _ [] = 0
accuracy logits p xs = metricsAccuracy (metrics logits p xs)

allBits :: Int -> [[Int]]
allBits 0 = [[]]
allBits n = [ b : bs | b <- [0, 1], bs <- allBits (n - 1) ]

isDyck :: [Int] -> Bool
isDyck = go 0
  where
    go bal [] = bal == 0
    go bal (x : xs) =
      let bal' = if x == 1 then bal + 1 else bal - 1
      in bal' >= 0 && go bal' xs

dyckData :: Int -> Int -> [Example]
dyckData seed parenLen = dyckDataPadded seed parenLen parenLen

dyckDataPadded :: Int -> Int -> Int -> [Example]
dyckDataPadded seed maxParenLen parenLen = valids ++ take (length valids) invalids
  where
    examples =
      [ (0 : map tok bits ++ replicate (maxParenLen - parenLen) 3, if isDyck bits then 1 else 0)
      | bits <- allBits parenLen ]
    valids = filter ((== 1) . snd) examples
    (invalids, _) = shuffle (mkStdGen (seed + 2000)) (filter ((== 0) . snd) examples)
    tok 1 = 1 -- '('
    tok _ = 2 -- ')'

dyckDataPaddedMany :: Int -> Int -> [Int] -> [Example]
dyckDataPaddedMany seed maxParenLen parenLens = concat
  [ dyckDataPadded (seed + 997 * i) maxParenLen parenLen
  | (i, parenLen) <- zip [0 ..] parenLens ]

stratifiedSplit :: Double -> StdGen -> [Example] -> ([Example], [Example])
stratifiedSplit frac g xs = (tr, te)
  where
    positives = filter ((== 1) . snd) xs
    negatives = filter ((== 0) . snd) xs
    (posS, g1) = shuffle g positives
    (negS, g2) = shuffle g1 negatives
    nTrain ys = max 0 (min (length ys) (floor (frac * fromIntegral (length ys))))
    (trP, teP) = splitAt (nTrain posS) posS
    (trN, teN) = splitAt (nTrain negS) negS
    (tr, g3) = shuffle g2 (trP ++ trN)
    (te, _) = shuffle g3 (teP ++ teN)

batchGrad :: (Additive p, Scale p)
          => (Example -> p -> (p, Double)) -> p -> [Example] -> (p, Double)
batchGrad grad p batch =
  let (g, l) = foldl' step (zeroA, 0) batch
      n = fromIntegral (max 1 (length batch))
  in (scaleA (1 / n) g, l / n)
  where
    step (!ga, !la) ex = let (g, l) = grad ex p in (addA ga g, la + l)

softmaxD :: [Double] -> [Double]
softmaxD xs = let m = maximum xs; es = map (\x -> exp (x - m)) xs; z = sum es in map (/ z) es

-- Interactive Dyck REPL.  Vocabulary: '(' and ')'; the model input prepends CLS=0
-- and pads with PAD=3 to the mode's seqLen.  Prints the model's valid/invalid call
-- (with P(valid)) alongside the ground-truth balance check.
-- `usesPad` distinguishes the padded modes (OOD: vocab includes PAD=3, variable
-- length) from the fixed-length modes (vocab {CLS,'(',')'}; every string is exactly
-- maxParenLen parens, no PAD).  Padding a fixed-length (no-PAD) model would index a
-- non-existent embedding row, so we constrain the input to match the mode's data.
promptDyck :: forall p. Serialize p => BenchCfg -> (p -> [Int] -> V 2) -> Int -> Bool -> Int -> IO ()
promptDyck cfg logits seqLen usesPad nParam = do
  loaded <- loadCkpt nParam (bCkpt cfg)
  case loaded of
    Nothing -> die $ "No checkpoint at " ++ bCkpt cfg ++ " — train this mode first (run without --prompt)."
    Just (meta, ps, _st) -> do
      let maxLen = seqLen - 1
          tok c  = if c == '(' then 1 else 2 :: Int
      printf "Loaded %s (mode %s, epoch %d).\n" (bCkpt cfg) (ckMode meta) (ckEpoch meta)
      if usesPad
        then printf "Prompt: enter a '(' ')' string (length <= %d, shorter is PAD-filled); :q to quit.\n" maxLen
        else printf "Prompt: fixed-length mode — enter exactly %d of '(' ')'; :q to quit.\n" maxLen
      let repl = do
            eof <- isEOF
            if eof then pure () else do
              line <- getLine
              if line == ":q" then pure () else do
                let s  = filter (not . isSpace) line
                    ok = all (`elem` "()") s && (if usesPad then length s <= maxLen else length s == maxLen)
                if ok
                  then do
                    let toks   = 0 : map tok s ++ (if usesPad then replicate (maxLen - length s) 3 else [])
                        dist   = softmaxD (vtoList (logits ps toks))
                        pValid = dist !! 1
                        truth  = isDyck (map (\c -> if c == '(' then 1 else 2) s)
                    printf "  %-7s P(valid)=%.3f   [actually balanced? %s]\n"
                      (if pValid >= 0.5 then "VALID" else "INVALID" :: String) pValid
                      (if truth then "yes" else "no" :: String)
                  else if usesPad
                    then printf "  expected '(' ')' chars, length <= %d; :q to quit\n" maxLen
                    else printf "  fixed-length mode: enter exactly %d of '(' ')'; :q to quit\n" maxLen
                repl
      repl

trainBenchSplit :: forall p. (Adam p, Additive p, Scale p, NFData p, Serialize p)
           => BenchCfg -> String -> p -> (Example -> p -> (p, Double)) -> (p -> [Int] -> V 2) -> [Example] -> [Example] -> IO ()
trainBenchSplit cfg desc params0 grad logits tr0 te0
  | bPrompt cfg = promptDyck cfg logits seqLen usesPad nParam
  | otherwise = do
  let adamCfg = AdamConfig 0.9 0.999 1.0e-8 (bWD cfg)
      batchesPer = length (chunksOf (bBatch cfg) tr0)
      totalSteps = max 1 (bEpochs cfg * max 1 batchesPer)
      validRate xs = 100 * fromIntegral (length (filter ((== 1) . snd) xs)) / fromIntegral (max 1 (length xs)) :: Double
  printf "════════════════════════════════════════════════════════════════════════\n"
  printf " Dyck-1 balanced-parentheses benchmark — %s\n" desc
  printf "════════════════════════════════════════════════════════════════════════\n"
  printf " Task : classify balanced valid/invalid Dyck-1 strings. Token 0 is CLS; tokens 1/2 are '(' / ')'; token 3 is PAD and is masked from attention.\n"
  printf " Data : total=%d train=%d test=%d valid-rate(train)=%.1f%% valid-rate(test)=%.1f%%\n"
    (length tr0 + length te0) (length tr0) (length te0) (validRate tr0) (validRate te0)
  printf " Model: %d parameters, batch=%d, epochs=%d, lr=%g, wd=%g, seed=%d\n"
    (length (toFloats params0)) (bBatch cfg) (bEpochs cfg) (bLR cfg) (bWD cfg) (bSeed cfg)
  let testLens = sort (nub (map (effectiveLen . fst) te0))
  case bFormat cfg of
    Compact -> printf "%7s | %9s | %8s | %5s | %5s | %5s | %5s | %7s | %s\n"
      "epoch" "step" "loss" "train" "test" "best" "valid" "invalid" (formatLengthHeader testLens)
    Full -> printf "%7s | %9s | %8s | %8s | %7s | %7s\n" "epoch" "step" "lr" "loss" "train%" "test%"
  let loop :: Int -> Maybe EvalResult -> Maybe EvalResult -> p -> AdamState p -> StdGen -> IO (Maybe EvalResult, Maybe EvalResult)
      loop !epoch !best !finalEval !ps !st !g
        | epoch > bEpochs cfg = do
            saveCkpt (bCkpt cfg) (CkptMeta "dyck" (bMode cfg) (epoch - 1)) ps st
            pure (best, finalEval)
        | otherwise = do
            let (shuf, g') = shuffle g tr0
                batches = chunksOf (bBatch cfg) shuf
                (ps', st', lossSum) = foldl' stepBatch (ps, st, 0) batches
                stepBatch (!p0, !s0, !ls) batch =
                  let (gAvg, l) = batchGrad grad p0 batch
                      stepN = asT s0 + 1
                      lr = lrWarmupCosine 0 totalSteps (bLR cfg) (bLR cfg) stepN
                      (p1, s1) = adamStep adamCfg lr p0 s0 gAvg
                  in (p1, s1, ls + l)
                avgLoss = lossSum / fromIntegral (max 1 (length batches))
            ps' `deepseq` pure ()
            (best', finalEval') <- if epoch == 1 || epoch `mod` bEvalEv cfg == 0 || epoch == bEpochs cfg
              then do
                let tr = accuracy logits ps' tr0
                    (teM, teByLen) = testDiagnostics logits ps' te0
                    te = metricsAccuracy teM
                    result = EvalResult epoch (asT st') avgLoss tr teM teByLen
                    bestNext = case best of
                      Nothing -> Just result
                      Just bestResult
                        | te > erTest bestResult -> Just result
                        | otherwise   -> best
                    bestTest = maybe te erTest bestNext
                    isNewBest = case best of
                      Nothing -> True
                      Just bestResult -> te > erTest bestResult
                case bFormat cfg of
                  Compact -> printCompactEval result bestTest isNewBest
                  Full -> printFullEvalWithLR (bLR cfg) result
                saveCkpt (bCkpt cfg) (CkptMeta "dyck" (bMode cfg) epoch) ps' st'  -- latest (resumable)
                when isNewBest $                                                  -- best-by-test
                  saveCkpt (bCkpt cfg ++ ".best") (CkptMeta "dyck" (bMode cfg) epoch) ps' st'
                pure (bestNext, Just result)
              else pure (best, finalEval)
            loop (epoch + 1) best' finalEval' ps' st' g'
  (ps0, st0, startEp) <- if bResume cfg
    then do
      l <- loadCkpt nParam (bCkpt cfg)
      case l of
        Just (m, ps, st) -> do
          printf "Resumed %s at epoch %d (step %d)\n" (bCkpt cfg) (ckEpoch m) (asT st)
          pure (ps, st, ckEpoch m + 1)
        Nothing -> pure (params0, initAdam, 1)
    else pure (params0, initAdam, 1)
  (best, finalEval) <- loop startEp Nothing Nothing ps0 st0 (mkStdGen (bSeed cfg + 1000))
  printf "\nSummary\n"
  case best of
    Nothing -> printf " No evals were run.\n"
    Just r -> printSummary "Best " r
  case finalEval of
    Nothing -> pure ()
    Just r -> printSummary "Final" r
  where
    nParam  = length (toFloats params0)
    seqLen  = case tr0 ++ te0 of ((toks, _) : _) -> length toks; _ -> 0
    usesPad = any ((3 `elem`) . fst) (tr0 ++ te0)   -- True for padded (OOD) modes

trainBench :: forall p. (Adam p, Additive p, Scale p, NFData p, Serialize p)
           => BenchCfg -> String -> p -> (Example -> p -> (p, Double)) -> (p -> [Int] -> V 2) -> [Example] -> IO ()
trainBench cfg desc params0 grad logits allData =
  let (tr0, te0) = stratifiedSplit (bFrac cfg) (mkStdGen (bSeed cfg)) allData
  in trainBenchSplit cfg desc params0 grad logits tr0 te0

runDyck1 :: BenchCfg -> IO ()
runDyck1 cfg = trainBench cfg "1-layer single-head, seqLen=13, dModel=32, dFF=128"
  params grad logits (dyckData (bSeed cfg) 12)
  where
    params = fst (fromFloats (fst (initFlatSeq1 13 3 32 128 32 (mkStdGen (bSeed cfg))))) :: SeqParams1 13 3 32 128 32
    grad (toks, y) p = gradLoss1 @13 @3 @32 @128 @32 toks y p
    logits p toks = logits1 @13 @3 @32 @128 @32 toks p

runDyck2 :: BenchCfg -> IO ()
runDyck2 cfg = trainBench cfg "2-layer single-head, seqLen=13, dModel=32, dFF=128"
  params grad logits (dyckData (bSeed cfg) 12)
  where
    params = fst (fromFloats (fst (initFlatSeq2 13 3 32 128 32 (mkStdGen (bSeed cfg))))) :: SeqParams2 13 3 32 128 32
    grad (toks, y) p = gradLoss2 @13 @3 @32 @128 @32 toks y p
    logits p toks = logits2 @13 @3 @32 @128 @32 toks p

runDyck16_1 :: BenchCfg -> IO ()
runDyck16_1 cfg = trainBench cfg "1-layer single-head, seqLen=17, dModel=32, dFF=128"
  params grad logits (dyckData (bSeed cfg) 16)
  where
    params = fst (fromFloats (fst (initFlatSeq1 17 3 32 128 32 (mkStdGen (bSeed cfg))))) :: SeqParams1 17 3 32 128 32
    grad (toks, y) p = gradLoss1 @17 @3 @32 @128 @32 toks y p
    logits p toks = logits1 @17 @3 @32 @128 @32 toks p

runDyck16_2 :: BenchCfg -> IO ()
runDyck16_2 cfg = trainBench cfg "2-layer single-head, seqLen=17, dModel=32, dFF=128"
  params grad logits (dyckData (bSeed cfg) 16)
  where
    params = fst (fromFloats (fst (initFlatSeq2 17 3 32 128 32 (mkStdGen (bSeed cfg))))) :: SeqParams2 17 3 32 128 32
    grad (toks, y) p = gradLoss2 @17 @3 @32 @128 @32 toks y p
    logits p toks = logits2 @17 @3 @32 @128 @32 toks p

runDyck18_1 :: BenchCfg -> IO ()
runDyck18_1 cfg = trainBench cfg "1-layer single-head, seqLen=19, dModel=32, dFF=128"
  params grad logits (dyckData (bSeed cfg) 18)
  where
    params = fst (fromFloats (fst (initFlatSeq1 19 3 32 128 32 (mkStdGen (bSeed cfg))))) :: SeqParams1 19 3 32 128 32
    grad (toks, y) p = gradLoss1 @19 @3 @32 @128 @32 toks y p
    logits p toks = logits1 @19 @3 @32 @128 @32 toks p

runDyck18_2 :: BenchCfg -> IO ()
runDyck18_2 cfg = trainBench cfg "2-layer single-head, seqLen=19, dModel=32, dFF=128"
  params grad logits (dyckData (bSeed cfg) 18)
  where
    params = fst (fromFloats (fst (initFlatSeq2 19 3 32 128 32 (mkStdGen (bSeed cfg))))) :: SeqParams2 19 3 32 128 32
    grad (toks, y) p = gradLoss2 @19 @3 @32 @128 @32 toks y p
    logits p toks = logits2 @19 @3 @32 @128 @32 toks p

runDyckOod12_16_1 :: BenchCfg -> IO ()
runDyckOod12_16_1 cfg = trainBenchSplit cfg "1-layer single-head, OOD train lengths <=12 test lengths 14/16, seqLen=17, dModel=32, dFF=128"
  params grad logits trainData testData
  where
    params = fst (fromFloats (fst (initFlatSeq1 17 4 32 128 32 (mkStdGen (bSeed cfg))))) :: SeqParams1 17 4 32 128 32
    grad (toks, y) p = gradLoss1 @17 @4 @32 @128 @32 toks y p
    logits p toks = logits1 @17 @4 @32 @128 @32 toks p
    trainData = dyckDataPaddedMany (bSeed cfg) 16 [2, 4, 6, 8, 10, 12]
    testData = dyckDataPaddedMany (bSeed cfg + 10000) 16 [14, 16]

runDyckOod12_16_2 :: BenchCfg -> IO ()
runDyckOod12_16_2 cfg = trainBenchSplit cfg "2-layer single-head, OOD train lengths <=12 test lengths 14/16, seqLen=17, dModel=32, dFF=128"
  params grad logits trainData testData
  where
    params = fst (fromFloats (fst (initFlatSeq2 17 4 32 128 32 (mkStdGen (bSeed cfg))))) :: SeqParams2 17 4 32 128 32
    grad (toks, y) p = gradLoss2 @17 @4 @32 @128 @32 toks y p
    logits p toks = logits2 @17 @4 @32 @128 @32 toks p
    trainData = dyckDataPaddedMany (bSeed cfg) 16 [2, 4, 6, 8, 10, 12]
    testData = dyckDataPaddedMany (bSeed cfg + 10000) 16 [14, 16]

data Opts = Opts
  { optMode :: String
  , optEpochs :: Int
  , optBatch :: Int
  , optSeed :: Int
  , optFrac :: Double
  , optLR :: Double
  , optWD :: Double
  , optEvalEvery :: Int
  , optEvalFormat :: String
  , optCkpt :: Maybe FilePath
  , optResume :: Bool
  , optPrompt :: Bool
  , optList :: Bool
  }

listModesText :: String
listModesText = unlines
  [ "Modes (-m) for transformer-benchmarks — Dyck-1 balanced-parentheses:"
  , "  dyck12-1 / dyck12-2          fixed length 12 (seqLen 13), 1- / 2-layer"
  , "  dyck16-1 / dyck16-2          fixed length 16 (seqLen 17), 1- / 2-layer"
  , "  dyck18-1 / dyck18-2          fixed length 18 (seqLen 19), 1- / 2-layer   [heaviest]"
  , "  dyck-ood-12-16-1 / -2        OOD: train lengths <=12, test 14/16, 1- / 2-layer"
  ]

optsP :: O.Parser Opts
optsP = Opts
  <$> O.strOption (O.long "mode" <> O.short 'm' <> O.value "dyck12-2" <> O.showDefault
        <> O.help "dyck12-1|dyck12-2|dyck16-1|dyck16-2|dyck18-1|dyck18-2|dyck-ood-12-16-1|dyck-ood-12-16-2")
  <*> O.option O.auto (O.long "epochs" <> O.short 'e' <> O.value 50 <> O.showDefault <> O.metavar "N")
  <*> O.option O.auto (O.long "batch" <> O.short 'b' <> O.value 64 <> O.showDefault <> O.metavar "N")
  <*> O.option O.auto (O.long "seed" <> O.short 's' <> O.value 42 <> O.showDefault <> O.metavar "N")
  <*> O.option O.auto (O.long "train-frac" <> O.value 0.5 <> O.showDefault <> O.metavar "F")
  <*> O.option O.auto (O.long "lr" <> O.value 1.0e-3 <> O.showDefault <> O.metavar "LR")
  <*> O.option O.auto (O.long "weight-decay" <> O.value 1.0e-3 <> O.showDefault <> O.metavar "WD")
  <*> O.option O.auto (O.long "eval-every" <> O.value 5 <> O.showDefault <> O.metavar "N")
  <*> O.strOption (O.long "eval-format" <> O.value "compact" <> O.showDefault <> O.metavar "compact|full")
  <*> O.optional (O.strOption (O.long "checkpoint" <> O.short 'c' <> O.metavar "PATH"
        <> O.help "checkpoint file (default checkpoint-dyck-<mode>.ckpt); best model also saved to <path>.best"))
  <*> O.switch (O.long "resume" <> O.help "resume training from the checkpoint if it exists")
  <*> O.switch (O.long "prompt" <> O.help "load the checkpoint and classify strings interactively (no training)")
  <*> O.switch (O.long "list-modes" <> O.help "list all -m modes and exit")

main :: IO ()
main = do
  o <- O.execParser $ O.info (optsP O.<**> O.helper)
    (O.fullDesc <> O.header "transformer-benchmarks — harder algorithmic tasks")
  when (optEpochs o < 0) $ die "--epochs must be non-negative"
  when (optBatch o <= 0) $ die "--batch must be positive"
  when (optFrac o < 0 || optFrac o > 1) $ die "--train-frac must be between 0 and 1"
  when (optEvalEvery o <= 0) $ die "--eval-every must be positive"
  evalFormat <- case optEvalFormat o of
    "compact" -> pure Compact
    "full" -> pure Full
    bad -> die $ "Unknown --eval-format: " ++ bad ++ "\nValid formats: compact, full"
  let ckpt = fromMaybe ("checkpoint-dyck-" ++ optMode o ++ ".ckpt") (optCkpt o)
      cfg = BenchCfg (optEpochs o) (optBatch o) (optFrac o) (optSeed o) (optLR o) (optWD o)
                     (optEvalEvery o) evalFormat ckpt (optResume o) (optPrompt o) (optMode o)
  if optList o then putStr listModesText else case optMode o of
    "dyck12-1" -> runDyck1 cfg
    "dyck12-2" -> runDyck2 cfg
    "dyck16-1" -> runDyck16_1 cfg
    "dyck16-2" -> runDyck16_2 cfg
    "dyck18-1" -> runDyck18_1 cfg
    "dyck18-2" -> runDyck18_2 cfg
    "dyck-ood-12-16-1" -> runDyckOod12_16_1 cfg
    "dyck-ood-12-16-2" -> runDyckOod12_16_2 cfg
    bad -> die $ "Unknown benchmark mode: " ++ bad ++ "\nValid modes: dyck12-1, dyck12-2, dyck16-1, dyck16-2, dyck18-1, dyck18-2, dyck-ood-12-16-1, dyck-ood-12-16-2"
