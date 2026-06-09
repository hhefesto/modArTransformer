{-# LANGUAGE ScopedTypeVariables #-}

-- The market tokenizer: the backend realization of the quantizer DENOTATION in
-- ModArTransformer/Semantics/MarketLanguage.agda.
--
--   * Vocabulary layout mirrors the Agda module exactly:
--       ⊥ (begin) = 0, † (end) = 1, bin i = 2 + i;  vocab = nBins + 2.
--   * `quantize edges r` = the number of edges ≤ r — literally the Agda
--     definition (`if r f< e then zero else suc (quantize es r)`), so the two
--     sides agree pointwise on every return.
--   * Edges are QUANTILES OF THE TRAINING WINDOW ONLY — the no-lookahead rule
--     (the side condition stated in MarketLanguage.agda).  `fitSpec` takes the
--     train slice; the walk-forward splitter hands it nothing else.
--   * Decoding maps a bin to the median training return that fell in it.
--
-- Pure; file I/O is limited to saving/loading the fitted spec artifact.
module MarketTokenizer
  ( TokenizerSpec(..)
  , bosTok, eosTok, binTok, unBinTok, vocabSize
  , logReturns
  , fitSpec
  , quantize
  , encodeReturn, encodeSeries
  , decodeTok
  , walkForwardSplit, contextWindows
  , saveSpec, loadSpec
  ) where

import Data.List (sort)
import Text.Read (readMaybe)

-- ── the fitted artifact ───────────────────────────────────────────────────────

data TokenizerSpec = TokenizerSpec
  { tsNBins  :: Int        -- number of bins (vocab = nBins + 2)
  , tsMean   :: Double     -- train-window mean of log returns (normalization)
  , tsStd    :: Double     -- train-window std of log returns
  , tsEdges  :: [Double]   -- nBins−1 sorted interior edges, in NORMALIZED units
  , tsReps   :: [Double]   -- nBins per-bin representatives (median train return,
                           --   normalized units) — the decoder
  } deriving (Show, Eq)

-- ── vocabulary layout (must mirror Semantics/MarketLanguage.agda) ─────────────

bosTok, eosTok :: Int
bosTok = 0      -- ⊥
eosTok = 1      -- †

binTok :: Int -> Int
binTok i = 2 + i

unBinTok :: Int -> Maybe Int
unBinTok t | t >= 2    = Just (t - 2)
           | otherwise = Nothing

vocabSize :: TokenizerSpec -> Int
vocabSize ts = tsNBins ts + 2

-- ── returns ───────────────────────────────────────────────────────────────────

-- log returns of a (time-ordered, positive) price series.
logReturns :: [Double] -> [Double]
logReturns ps = zipWith (\a b -> log (b / a)) ps (drop 1 ps)

-- ── fitting (TRAIN WINDOW ONLY) ───────────────────────────────────────────────

-- Fit nBins quantile bins on the training returns: normalize (z-score), take
-- the i/nBins quantiles (i = 1..nBins−1) as edges, and record per-bin median
-- representatives.  Edges are sorted by construction (quantiles of a sorted
-- sample), satisfying the Sorted premise of the Agda QuantizeMono obligation.
fitSpec :: Int -> [Double] -> TokenizerSpec
fitSpec nBins trainReturns
  | nBins < 2 = error "fitSpec: need at least 2 bins"
  | length trainReturns < nBins =
      error ("fitSpec: " ++ show (length trainReturns) ++ " returns < " ++ show nBins ++ " bins")
  | otherwise =
      let m   = mean trainReturns
          s0  = stddev m trainReturns
          s   = if s0 <= 0 then 1 else s0
          zs  = map (\x -> (x - m) / s) trainReturns
          srt = sort zs
          n   = length srt
          q i = srt !! min (n - 1) ((i * n) `div` nBins)
          edges = [ q i | i <- [1 .. nBins - 1] ]
          reps  = [ binMedian b | b <- [0 .. nBins - 1] ]
            where binMedian b =
                    let xs = [ z | z <- zs, quantize edges z == b ]
                    in if null xs then fallback b else median xs
                  fallback 0 = head edges - 1
                  fallback b | b == nBins - 1 = last edges + 1
                             | otherwise      = (edges !! (b - 1) + edges !! b) / 2
      in TokenizerSpec nBins m s edges reps
  where
    mean xs = sum xs / fromIntegral (length xs)
    stddev m xs = sqrt (sum [ (x - m) ^ (2 :: Int) | x <- xs ] / fromIntegral (length xs))
    median xs = let ys = sort xs in ys !! (length ys `div` 2)

-- ── the quantizer (the Agda denotation, verbatim) ─────────────────────────────

-- quantize edges r = number of edges ≤ r.  Identical to the Agda recursion:
--   quantize []       _ = zero
--   quantize (e ∷ es) r = if r f< e then zero else suc (quantize es r)
quantize :: [Double] -> Double -> Int
quantize []       _ = 0
quantize (e : es) r = if r < e then 0 else 1 + quantize es r

-- a raw (unnormalized) return → its token.
encodeReturn :: TokenizerSpec -> Double -> Int
encodeReturn ts r =
  let z = (r - tsMean ts) / tsStd ts
  in binTok (quantize (tsEdges ts) z)

-- a raw return series → ⊥ : bin tokens : †   (a full "sentence")
encodeSeries :: TokenizerSpec -> [Double] -> [Int]
encodeSeries ts rs = bosTok : map (encodeReturn ts) rs ++ [eosTok]

-- a token → its representative raw return (Nothing for ⊥/†).
decodeTok :: TokenizerSpec -> Int -> Maybe Double
decodeTok ts t = do
  b <- unBinTok t
  if b < tsNBins ts
    then Just (tsReps ts !! b * tsStd ts + tsMean ts)
    else Nothing

-- ── walk-forward dataset building (no temporal leakage) ──────────────────────

-- Split a time-ordered series at a fraction: everything before the boundary is
-- train, everything after is test.  NEVER shuffled.
walkForwardSplit :: Double -> [a] -> ([a], [a])
walkForwardSplit frac xs =
  let k = floor (frac * fromIntegral (length xs))
  in splitAt k xs

-- All length-n context windows (stride 1) over a token stream.
contextWindows :: Int -> [Int] -> [[Int]]
contextWindows n toks
  | n <= 0    = []
  | otherwise = [ take n (drop i toks) | i <- [0 .. length toks - n] ]

-- ── spec artifact I/O ─────────────────────────────────────────────────────────

-- header: MKTOK <nBins> <mean> <std>, then nBins−1 edges, then nBins reps,
-- one float per line.
saveSpec :: FilePath -> TokenizerSpec -> IO ()
saveSpec path ts =
  writeFile path $ unlines $
    unwords ["MKTOK", show (tsNBins ts), show (tsMean ts), show (tsStd ts)]
    : map show (tsEdges ts ++ tsReps ts)

loadSpec :: FilePath -> IO TokenizerSpec
loadSpec path = do
  s <- readFile path
  case lines s of
    (h : rest) -> case words h of
      ["MKTOK", nS, mS, sS]
        | Just n <- readMaybe nS, Just m <- readMaybe mS, Just sd <- readMaybe sS
        , Just nums <- traverse readMaybe rest
        , length nums == (n - 1) + n ->
            let (es, reps) = splitAt (n - 1) nums
            in pure (TokenizerSpec n m sd es reps)
      _ -> bad
    _ -> bad
  where bad = error ("loadSpec: " ++ path ++ " is not a valid tokenizer spec")
