{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

-- One successful backend training epoch, deliberately tiny.
--
-- This executable is the next milestone after the scalar/matvec Cont playground:
-- a shape-indexed Haskell backend trains a small modular-addition softmax model
-- for one epoch.  It is not the full transformer and not yet the real CTC plugin;
-- it is a CTC-shaped backend target using Cont reverse AD over typed tensors.

module Main where

import AD
import Text.Printf (printf)
import qualified Tensor as T

type P = 5
type InDim = 25

type Params = (T.Mat P InDim, T.Vec P)

data Example = Example Int Int Int
  deriving Show

examples :: [Example]
examples = [Example a b ((a + b) `mod` 5) | a <- [0..4], b <- [0..4]]

zeroParams :: Params
zeroParams =
  ( T.mat @P @InDim (replicate 5 (replicate 25 0))
  , T.vec @P (replicate 5 0)
  )

oneHot :: Int -> Int -> [Double]
oneHot n i = [if j == i then 1 else 0 | j <- [0 .. n - 1]]

inputVec :: Int -> Int -> T.Vec InDim
inputVec a b = T.vec @InDim (oneHot 25 (a * 5 + b))

targetVec :: Int -> T.Vec P
targetVec t = T.vec @P (oneHot 5 t)

logSumExp :: T.Vec P -> Double
logSumExp (T.Vec xs) =
  let mx = maximum xs
  in mx + log (sum [exp (x - mx) | x <- xs])

softmax :: T.Vec P -> T.Vec P
softmax (T.Vec xs) =
  let mx = maximum xs
      exps = [exp (x - mx) | x <- xs]
      z = sum exps
  in T.Vec [e / z | e <- exps]

select :: Int -> T.Vec P -> Double
select i (T.Vec xs) = xs !! i

logits :: Params -> T.Vec InDim -> T.Vec P
logits (w, b) x = T.vadd (T.matvec w x) b

lossD :: Example -> DCont Double Params Double
lossD (Example a b t) = DCont $ \params ->
  let x = inputVec a b
      y = logits params x
      loss = logSumExp y - select t y
      dLogits = T.vadd (softmax y) (T.vscale (-1) (targetVec t))
      cont = Cont $ \(AddFun k) -> AddFun $ \(dw, db) ->
        k (T.vdot dLogits (T.vadd (T.matvec dw x) db))
  in (loss, cont)

paramsDot :: Params -> Params -> Double
paramsDot (w1, b1) (w2, b2) = matDot w1 w2 + T.vdot b1 b2
  where
    matDot (T.Mat rowsA) (T.Mat rowsB) = sum (zipWith rowDot rowsA rowsB)
    rowDot xs ys = sum (zipWith (*) xs ys)

matBasis :: Int -> Int -> T.Mat P InDim
matBasis row col = T.mat @P @InDim
  [ [if r == row && c == col then 1 else 0 | c <- [0..24]] | r <- [0..4] ]

vecBasis :: Int -> T.Vec P
vecBasis ix = T.vec @P [if i == ix then 1 else 0 | i <- [0..4]]

materializeGrad :: (Params -> Double) -> Params
materializeGrad lin =
  ( T.mat @P @InDim [[lin (matBasis r c, T.vec @P (replicate 5 0)) | c <- [0..24]] | r <- [0..4]]
  , T.vec @P [lin (T.mat @P @InDim (replicate 5 (replicate 25 0)), vecBasis i) | i <- [0..4]]
  )

gradAndLoss :: Example -> Params -> (Params, Double)
gradAndLoss ex params =
  let (loss, cont) = runDCont (lossD ex) params
      AddFun lin = runCont cont idL
  in (materializeGrad lin, loss)

scaleParams :: Double -> Params -> Params
scaleParams s (T.Mat rows, T.Vec b) =
  (T.Mat (map (map (s *)) rows), T.Vec (map (s *) b))

addParams :: Params -> Params -> Params
addParams (T.Mat a, T.Vec b) (T.Mat c, T.Vec d) =
  (T.Mat (zipWith (zipWith (+)) a c), T.Vec (zipWith (+) b d))

sgdStep :: Double -> Params -> Example -> (Params, Double)
sgdStep lr params ex =
  let (grad, loss) = gradAndLoss ex params
  in (addParams params (scaleParams (-lr) grad), loss)

epoch :: Double -> Params -> [Example] -> (Params, Double)
epoch lr = go 0 0
  where
    go total n params [] = (params, total / fromIntegral n)
    go total n params (ex:xs) =
      let (params', loss) = sgdStep lr params ex
      in go (total + loss) (n + 1 :: Int) params' xs

averageLoss :: Params -> [Example] -> Double
averageLoss params xs =
  sum [snd (gradAndLoss ex params) | ex <- xs] / fromIntegral (length xs)

main :: IO ()
main = do
  let lr = 0.25
      before = averageLoss zeroParams examples
      (params1, trainLoss) = epoch lr zeroParams examples
      after = averageLoss params1 examples
      deltaNorm = paramsDot params1 params1
  putStrLn "Backend Cont training smoke test"
  putStrLn "model: typed pair-feature linear softmax classifier for p=5 modular addition"
  putStrLn "AD: Cont reverse-mode, materialized by basis evaluation for this tiny model"
  printf "examples = %d\n" (length examples)
  printf "loss before epoch = %.9f\n" before
  printf "mean online epoch loss = %.9f\n" trainLoss
  printf "loss after epoch = %.9f\n" after
  printf "parameter squared norm after epoch = %.9f\n" deltaNorm
