{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- Haskell side of the Agda↔backend conformance oracle.  Builds a fixed tiny model
-- (Params 3 4 8 4 — same dims as the Agda oracle), writes its parameters to a
-- shared flat-float file (which the Agda oracle reads), and writes the backend's
-- logits ++ [loss] ++ gradient for the same fixed (a,b,target) cases.  A flake
-- check then tolerance-diffs this against the Agda oracle's output.
module Main where

import System.Random (mkStdGen, randomRs)
import Tensor (Additive(..), vtoList)
import Serialize
import Transformer

-- tiny model, matching modArConformanceOracle.agda (vocab 3, dM 4, dF 8, dK 4)
type V' = 3
type DM = 4
type DF = 8
type DK = 4
type P  = Params V' DM DF DK

-- tiny SEQUENCE model, matching the oracle's second section
-- (vocab 5, n 4, dM 4, dF 8, dK 2 — causal, two heads)
type VS  = 5
type NS  = 4
type DMS = 4
type DFS = 8
type DKS = 2
type PS  = ParamsSeq VS NS DMS DFS DKS

-- same three cases as the Agda oracle: (a, b, (a+b) mod 3)
cases :: [(Int, Int, Int)]
cases = [ (1, 2, 0), (0, 1, 1), (2, 2, 1) ]

-- same two token sequences as the Agda oracle
seqCases :: [[Int]]
seqCases = [ [1, 4, 2, 3], [0, 2, 4, 1] ]

-- logits ++ [loss] ++ gradient, in the same order the Agda oracle emits.
perCase :: P -> (Int, Int, Int) -> [Double]
perCase params (a, b, t) =
  let logits   = vtoList (transformerLogitsVal a b params)
      (g, l)   = transformerGradLoss a b t params
  in logits ++ (l : toFloats g)

-- per-position logits ++ [loss] ++ gradient for the sequence model.
perSeqCase :: PS -> [Int] -> [Double]
perSeqCase params toks =
  let logits = concatMap vtoList (seqLogitsVal toks params)
      (g, l) = seqGradLoss toks params
  in logits ++ (l : toFloats g)

main :: IO ()
main = do
  let n      = length (toFloats (zeroA :: P))
      flat   = take n (randomRs (-0.4, 0.4) (mkStdGen 7)) :: [Double]
      params = fst (fromFloats flat) :: P
      hs     = concatMap (perCase params) cases
      nS      = length (toFloats (zeroA :: PS))
      flatS   = take nS (randomRs (-0.4, 0.4) (mkStdGen 13)) :: [Double]
      paramsS = fst (fromFloats flatS) :: PS
      hsS     = concatMap (perSeqCase paramsS) seqCases
  writeFile "conformance-params.txt"     (unlines (map show flat))
  writeFile "conformance-seq-params.txt" (unlines (map show flatS))
  writeFile "conformance-hs.txt"         (unlines (map show (hs ++ hsS)))
  putStrLn ("[conformance] wrote conformance-params.txt (" ++ show n
            ++ " params), conformance-seq-params.txt (" ++ show nS
            ++ " params) and conformance-hs.txt ("
            ++ show (length hs + length hsS) ++ " floats)")
