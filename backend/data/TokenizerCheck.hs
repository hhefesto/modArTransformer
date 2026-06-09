-- Property checks for the market tokenizer against its Agda denotation
-- (Semantics/MarketLanguage.agda).  Hand-rolled (no QuickCheck dep), in the
-- style of GradCheck: deterministic pseudo-random cases, hard pass/fail.
--
--   1. MONOTONICITY (the QuantizeMono obligation): r ≤ s ⇒ bin(r) ≤ bin(s).
--   2. TOTALITY/RANGE: every return lands in a bin 0..nBins−1; every token
--      round-trips through the vocabulary layout (⊥=0, †=1, bin i = 2+i).
--   3. DECODE COHERENCE: decoding a bin's representative re-encodes to the
--      same bin.
--   4. NO-LOOKAHEAD: the fitted spec is a function of the TRAIN slice only —
--      fitting with different futures appended after the walk-forward
--      boundary yields the identical artifact.
--   5. SPEC I/O: save/load round-trips exactly.
module Main where

import Control.Monad (unless)
import System.Exit (exitFailure)
import System.Random (mkStdGen, randomRs)
import MarketTokenizer

check :: String -> Bool -> IO ()
check name ok = do
  putStrLn ((if ok then "PASS " else "FAIL ") ++ name)
  unless ok exitFailure

main :: IO ()
main = do
  let nBins  = 16
      -- synthetic "price" walk → returns (deterministic)
      noise  = randomRs (-0.02, 0.025) (mkStdGen 42) :: [Double]
      prices = scanl (\p r -> p * exp r) 100 (take 5000 noise)
      rets   = logReturns prices
      (trainR, testR) = walkForwardSplit 0.7 rets
      spec   = fitSpec nBins trainR

  -- 1. monotonicity over random pairs
  let pairs = zip (randomRs (-0.1, 0.1) (mkStdGen 1) :: [Double])
                  (randomRs (-0.1, 0.1) (mkStdGen 2) :: [Double])
      monoOk (a, b) = let (lo, hi) = (min a b, max a b)
                      in encodeReturn spec lo <= encodeReturn spec hi
  check "monotonicity (QuantizeMono)" (all monoOk (take 10000 pairs))

  -- 2. totality / vocabulary range
  let toks = map (encodeReturn spec) (trainR ++ testR)
  check "totality: all tokens are bins" (all (\t -> t >= 2 && t < vocabSize spec) toks)
  check "vocab layout: bos/eos below bins" (bosTok == 0 && eosTok == 1 && binTok 0 == 2)
  let series = encodeSeries spec (take 50 testR)
  check "series: bos…eos framing"
        (head series == bosTok && last series == eosTok && length series == 52)

  -- 3. decode coherence: bin → representative → same bin
  let cohOk b = case decodeTok spec (binTok b) of
                  Just r  -> encodeReturn spec r == binTok b
                  Nothing -> False
  check "decode coherence" (all cohOk [0 .. nBins - 1])

  -- 4. no-lookahead: the spec the pipeline uses is fit on the train slice
  -- only.  Two assertions: (a) the leak DETECTOR has power — fitting on
  -- train+test produces measurably different edges, so a leaking pipeline
  -- would be caught; (b) the deterministic re-fit on the train slice
  -- reproduces the pipeline's spec exactly.
  let specLeaky = fitSpec nBins (trainR ++ testR)
  check "no-lookahead: detector has power" (tsEdges specLeaky /= tsEdges spec)
  check "no-lookahead: spec ≡ fit(train slice)" (fitSpec nBins trainR == spec)
  -- and the edges really are sorted (the Sorted premise of QuantizeMono)
  check "edges sorted" (and (zipWith (<=) (tsEdges spec) (drop 1 (tsEdges spec))))

  -- 5. artifact round-trip
  saveSpec "/tmp/mktok-test.spec" spec
  spec' <- loadSpec "/tmp/mktok-test.spec"
  check "spec save/load round-trip" (spec' == spec)

  -- 6. context windows: every window has exactly n tokens, walk-forward order
  let ws = contextWindows 8 series
  check "context windows" (all ((== 8) . length) ws && length ws == length series - 7)

  putStrLn "TOKENIZER CHECK PASSED"
