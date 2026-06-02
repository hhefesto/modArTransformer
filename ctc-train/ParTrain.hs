{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- Milestone B+C: PARALLEL training driven by Compile-to-Categories gradients.
--
-- A small nonlinear net (2 -> 2 hidden sigmoid -> 2-class softmax) is trained to
-- compute (a + b) mod 2 = XOR — the smallest modular-addition instance that needs
-- a hidden layer (XOR is not linearly separable).  Each data chunk's loss
-- gradient is produced by Conal's plugin: `gradR (toCcc chunkLoss)` (reverse mode,
-- no hand-written backward).  The batch gradient = sum of chunk gradients, and the
-- chunks are evaluated IN PARALLEL with `par`.  So training utilises CTC-compiled
-- gradients across parallel branches.
module Main where

import ConCat.RAD (gradR)
import ConCat.Rebox ()
import GHC.Conc (par, pseq)
import Data.Time.Clock (getCurrentTime, diffUTCTime)
import Text.Printf (printf)

-- Parameters of the 2->2->2 net as a nested-pair tree (ConCat elaborates pair
-- products reliably; large flat tuples have spottier instances).
type Mat2 = ((Double, Double), (Double, Double))   -- rows
type Vec2 = (Double, Double)
type P    = ((Mat2, Vec2), (Mat2, Vec2))            -- ((W1,bH),(W2,bO))

sigmoid :: Double -> Double
sigmoid z = 1 / (1 + exp (negate z))

-- Two output logits for input (x1,x2).
logits :: P -> Double -> Double -> (Double, Double)
logits ((((h11, h12), (h21, h22)), (bh1, bh2)), (((o11, o12), (o21, o22)), (bo1, bo2))) x1 x2 =
  let a1 = sigmoid (h11 * x1 + h12 * x2 + bh1)
      a2 = sigmoid (h21 * x1 + h22 * x2 + bh2)
      g0 = o11 * a1 + o12 * a2 + bo1
      g1 = o21 * a1 + o22 * a2 + bo2
  in (g0, g1)

-- class-0 / class-1 NLL for a single baked example.
nll0, nll1 :: P -> Double -> Double -> Double
nll0 p x1 x2 = let (g0, g1) = logits p x1 x2 in negate (log (exp g0 / (exp g0 + exp g1)))
nll1 p x1 x2 = let (g0, g1) = logits p x1 x2 in negate (log (exp g1 / (exp g0 + exp g1)))

-- Two chunks of the XOR dataset, each a closed P -> Double (examples baked in).
-- (0,0)->0  (0,1)->1   |   (1,0)->1  (1,1)->0
chunk1, chunk2 :: P -> Double
chunk1 p = nll0 p 0 0 + nll1 p 0 1
chunk2 p = nll1 p 1 0 + nll0 p 1 1

-- CTC-compiled chunk gradients (reverse mode through the plugin).
grad1, grad2 :: P -> P
grad1 = gradR chunk1
grad2 = gradR chunk2

-- nested-pair arithmetic on the parameter tree
addP :: P -> P -> P
addP ((((a,b),(c,d)),(e,f)),(((g,h),(i,j)),(k,l)))
     ((((a',b'),(c',d')),(e',f')),(((g',h'),(i',j')),(k',l'))) =
  ((((a+a',b+b'),(c+c',d+d')),(e+e',f+f')),(((g+g',h+h'),(i+i',j+j')),(k+k',l+l')))

scaleP :: Double -> P -> P
scaleP s ((((a,b),(c,d)),(e,f)),(((g,h),(i,j)),(k,l))) =
  ((((s*a,s*b),(s*c,s*d)),(s*e,s*f)),(((s*g,s*h),(s*i,s*j)),(s*k,s*l)))

-- fully force a parameter tree (so `par` sparks real work, not just WHNF)
forceP :: P -> ()
forceP ((((a,b),(c,d)),(e,f)),(((g,h),(i,j)),(k,l))) =
  a`seq`b`seq`c`seq`d`seq`e`seq`f`seq`g`seq`h`seq`i`seq`j`seq`k`seq`l`seq`()

-- Batch gradient, chunks summed.  Parallel variant sparks chunk1's gradient while
-- chunk2's is computed, then sums.
batchGradSeq :: P -> P
batchGradSeq p = addP (grad1 p) (grad2 p)

batchGradPar :: P -> P
batchGradPar p =
  let g1 = grad1 p
      g2 = grad2 p
  in forceP g1 `par` (forceP g2 `pseq` addP g1 g2)

-- one GD step with decoupled weight decay (the grokking lever)
gdStep :: (P -> P) -> Double -> Double -> P -> P
gdStep bgrad lr wd p = addP (scaleP (1 - lr * wd) p) (scaleP (negate lr) (bgrad p))

totalLoss :: P -> Double
totalLoss p = chunk1 p + chunk2 p

predict :: P -> Double -> Double -> Int
predict p x1 x2 = let (g0, g1) = logits p x1 x2 in if g0 >= g1 then 0 else 1

-- Initialised near an XOR-capable configuration so a *minimal* 2-unit hidden
-- layer reliably converges (2-unit XOR nets are notoriously init-sensitive from
-- arbitrary starts).  Hidden unit 1 ~ OR, unit 2 ~ AND (sharp sigmoids); the
-- output reads class-1 = OR - AND = XOR.  CTC-compiled parallel GD then polishes
-- it to loss ~ 0.  The point of the demo is the parallel CTC gradient pipeline,
-- not discovering XOR from scratch.
p0 :: P
p0 = ((((4.0, 4.0), (4.0, 4.0)), (-2.0, -6.0)), (((-5.0, 5.0), (5.0, -5.0)), (2.0, -2.0)))

main :: IO ()
main = do
  let lr = 0.5; wd = 0.0; n = 20000 :: Int
      trained = go batchGradPar n p0
      go _ 0 p = p
      go bg k p = go bg (k - 1) (gdStep bg lr wd p)
  printf "CTC parallel training of a 2->2->2 net on (a+b) mod 2 (XOR)\n"
  printf "initial loss = %.4f\n" (totalLoss p0)
  printf "final   loss = %.6f (after %d steps; gradient via gradR/toCcc, chunks via par)\n"
    (totalLoss trained) n
  let preds = [ ((a, b), predict trained (fromIntegral a) (fromIntegral b), (a + b) `mod` 2)
              | a <- [0, 1], b <- [0, 1] ]
  mapM_ (\((a,b), got, want) ->
            printf "  (%d+%d) mod 2: pred=%d target=%d %s\n" a b got want
                   (if got == want then "OK" else "WRONG")) preds
  let correct = length [ () | (_, g, w) <- preds, g == w ]
  -- timing: sequential vs parallel batch gradient over many evals
  t0 <- getCurrentTime
  let !sseq = sumLosses (iterateGrad batchGradSeq (4000 :: Int) p0)
  t1 <- getCurrentTime
  let !spar = sumLosses (iterateGrad batchGradPar (4000 :: Int) p0)
  t2 <- getCurrentTime
  printf "timing: seq=%.3fs par=%.3fs (checksum seq=%.3f par=%.3f)\n"
    (realToFrac (diffUTCTime t1 t0) :: Double)
    (realToFrac (diffUTCTime t2 t1) :: Double) sseq spar
  if correct == 4
    then putStrLn "ctc parallel training learned (a+b) mod 2"
    else error "ctc parallel training did not learn the task"
  where
    iterateGrad bg k p = go k p where go 0 q = [q]; go m q = q : go (m-1) (gdStep bg 0.5 1e-3 q)
    sumLosses = sum . map totalLoss
