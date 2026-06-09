{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- Finite-difference gradient check for the Cont/Dual transformer.  This is the
-- correctness gate (plan Phase 4a): if structural cotangent accumulation over the
-- product tree is right, the chain-rule gradient must match central differences.
module Main where

import System.Random (mkStdGen, randomRs)
import Text.Printf (printf)
import Tensor (Additive(..))
import Serialize
import Transformer

-- tiny model
type V' = 3
type DM = 4
type DF = 8
type DK = 4
type P = Params V' DM DF DK

-- tiny SEQUENCE model (the conformance-oracle shape: causal, 2 heads, n=4)
type VS = 5
type NS = 4
type DMS = 4
type DFS = 8
type DKS = 2
type PS = ParamsSeq VS NS DMS DFS DKS

mkParams :: [Double] -> P
mkParams = fst . fromFloats

loss :: Int -> Int -> Int -> P -> Double
loss a b t p = snd (transformerGradLoss a b t p)

-- generic finite-difference check over sampled coordinates
fdCheck :: String -> Int -> [Double] -> [Double] -> ([Double] -> Double) -> IO Double
fdCheck name n flat gFlat lossAt = do
  let eps  = 1e-5
      idxs = [ (j * 7 + 3) `mod` n | j <- [0 .. 29] ]
      with xs (i, x) = [ if k == i then x else v | (k, v) <- zip [0 ..] xs ]
      fd j = let bump d = flat `with` (j, (flat !! j) + d)
                 lp = lossAt (bump eps)
                 lm = lossAt (bump (-eps))
             in (lp - lm) / (2 * eps)
  printf "%-6s %14s %14s %12s\n" "idx" "analytic" "finite-diff" "absErr"
  maxErr <- go idxs (\j -> gFlat !! j) fd 0
  printf "[%s] max abs error over %d sampled coords = %.3e\n" name (length idxs) maxErr
  pure maxErr
  where
    go :: [Int] -> (Int -> Double) -> (Int -> Double) -> Double -> IO Double
    go [] _ _ acc = pure acc
    go (j : js) g fd acc = do
      let a = g j
          f = fd j
          e = abs (a - f)
      printf "%-6d %14.6f %14.6f %12.3e\n" j a f e
      go js g fd (max acc e)

main :: IO ()
main = do
  -- ── check 1: the seqLen-2 single-head model (the original gate) ──
  let n      = length (toFloats (zeroA :: P))
      vals   = take n (randomRs (-0.4, 0.4) (mkStdGen 7))
      params = mkParams vals
      flat   = toFloats params
      (a, b, t) = (1, 2, (1 + 2) `mod` 3)
      (gParams, l0) = transformerGradLoss a b t params
  printf "params=%d  loss=%.6f\n" n l0
  err1 <- fdCheck "modular" n flat (toFloats gParams)
                  (\fs -> loss a b t (mkParams fs))
  -- ── check 2: the generalized sequence model (causal, 2 heads, n=4) ──
  let nS      = length (toFloats (zeroA :: PS))
      valsS   = take nS (randomRs (-0.4, 0.4) (mkStdGen 11))
      paramsS = fst (fromFloats valsS) :: PS
      flatS   = toFloats paramsS
      toks    = [1, 4, 2, 3] :: [Int]
      (gS, lS) = seqGradLoss toks paramsS
  printf "seq params=%d  loss=%.6f\n" nS lS
  err2 <- fdCheck "sequence" nS flatS (toFloats gS)
                  (\fs -> snd (seqGradLoss toks (fst (fromFloats fs) :: PS)))
  let maxErr = max err1 err2
  printf "overall max abs error = %.3e\n" maxErr
  if maxErr < 1e-4
    then putStrLn "GRADCHECK PASSED"
    else putStrLn "GRADCHECK FAILED"
