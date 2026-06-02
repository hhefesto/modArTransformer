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

mkParams :: [Double] -> P
mkParams = fst . fromFloats

loss :: Int -> Int -> Int -> P -> Double
loss a b t p = snd (transformerGradLoss a b t p)

main :: IO ()
main = do
  let n      = length (toFloats (zeroA :: P))
      vals   = take n (randomRs (-0.4, 0.4) (mkStdGen 7))
      params = mkParams vals
      flat   = toFloats params
      (a, b, t) = (1, 2, (1 + 2) `mod` 3)
      (gParams, l0) = transformerGradLoss a b t params
      gFlat  = toFloats gParams
      eps    = 1e-5
      -- sample coordinates spread across the parameter vector
      idxs   = [ (j * 7 + 3) `mod` n | j <- [0 .. 29] ]
      fd j   = let bump d = flat `with` (j, (flat !! j) + d)
                   lp = loss a b t (mkParams (bump eps))
                   lm = loss a b t (mkParams (bump (-eps)))
               in (lp - lm) / (2 * eps)
  printf "params=%d  loss=%.6f\n" n l0
  printf "%-6s %14s %14s %12s\n" "idx" "analytic" "finite-diff" "absErr"
  maxErr <- go idxs gFlat fd 0
  printf "max abs error over %d sampled coords = %.3e\n" (length idxs) maxErr
  if maxErr < 1e-4
    then putStrLn "GRADCHECK PASSED"
    else putStrLn "GRADCHECK FAILED"
  where
    with xs (i, x) = [ if k == i then x else v | (k, v) <- zip [0 ..] xs ]
    go :: [Int] -> [Double] -> (Int -> Double) -> Double -> IO Double
    go [] _ _ acc = pure acc
    go (j : js) g fd acc = do
      let a = g !! j
          f = fd j
          e = abs (a - f)
      printf "%-6d %14.6f %14.6f %12.3e\n" j a f e
      go js g fd (max acc e)
