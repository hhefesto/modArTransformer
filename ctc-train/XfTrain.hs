{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- ⚠ NOT BUILT — see the NOTE in ctc-train.cabal.  This self-attention block
-- trainer is kept as a documented attempt: reverse-mode `gradR` over softmax
-- self-attention does not compile in practical time/memory at this concat/ghc948
-- pin (1h27m, RSS climbing past 6 GB for a d=2 / 20-param instance).  The forward
-- attention elaborates fine (ctc-smoke Stage 8) and reverse-mode MLPs compile
-- fine (ctc-partrain); reverse-mode attention is the wall.
--
-- Parallel CTC training of a self-ATTENTION block on modular addition.
--
-- The model is a single-head self-attention layer over two tokens, with a
-- nonlinear (sigmoid) readout — the core transformer mechanism:
--   embed -> Q/K/V proj (Wk,Wv shared across positions) -> scaled-dot attention
--   (softmax) -> hidden sigmoid -> unembed -> class softmax.
-- It learns (a + b) mod 2.
--
-- NOTE: the full block (extra Wo + residual + two sqrt-based LayerNorms + FFN
-- down-proj) makes `gradR`'s Core blow up unboundedly at this concat/ghc948 pin
-- (a tiny d=2 instance ran 67 min / 16 GB before we killed it). The LayerNorms'
-- `sqrt`/variance are the cost amplifiers. This attention+readout keeps the
-- defining transformer op (softmax self-attention) with an op-count close to a
-- plain MLP, so its gradient compiles.
--
-- CTC on two axes: (1) each chunk's gradient is `gradR (toCcc chunkLoss)`
-- (reverse mode, no hand-written backward); (2) the chunk gradients are summed
-- with the CTC-compiled chunks evaluated concurrently via `par`/`pseq` (+RTS -N).
module Main where

import ConCat.RAD (gradR)
import ConCat.Rebox ()
import GHC.Conc (par, pseq)
import Data.Time.Clock (getCurrentTime, diffUTCTime)
import Text.Printf (printf)

class Vec a where
  vadd   :: a -> a -> a
  vscale :: Double -> a -> a
  vforce :: a -> ()
instance Vec Double where
  vadd = (+); vscale = (*); vforce = (`seq` ())
instance (Vec a, Vec b) => Vec (a, b) where
  vadd   (a, b) (c, d) = (vadd a c, vadd b d)
  vscale s (a, b)      = (vscale s a, vscale s b)
  vforce (a, b)        = vforce a `seq` vforce b `seq` ()

type D2  = (Double, Double)
type M22 = (D2, D2)
type Emb = (D2, D2)                  -- embeddings for tokens 0,1
type Attn = (M22, (M22, M22))        -- Wq,(Wk,Wv)
type Un   = (D2, D2)                  -- unembed rows for classes 0,1
type P    = (Emb, (Attn, (M22, Un)))  -- (emb,(attn,(Wup, unembed)))

mv :: M22 -> D2 -> D2
mv ((a, b), (c, d)) (x, y) = (a * x + b * y, c * x + d * y)

dot :: D2 -> D2 -> Double
dot (a, b) (c, d) = a * c + b * d

sigmoid :: Double -> Double
sigmoid z = 1 / (1 + exp (negate z))

-- Self-attention over (ea = position 0, eb = position 1), position-0 readout.
attnLogits :: P -> D2 -> D2 -> (Double, Double)
attnLogits (_emb, (attn, (wup, un))) ea eb =
  let (wq, (wk, wv)) = attn
      (u0, u1)       = un
      q  = mv wq ea
      k0 = mv wk ea; k1 = mv wk eb
      v0 = mv wv ea; v1 = mv wv eb
      s0 = dot q k0; s1 = dot q k1
      a0 = exp s0;  a1 = exp s1;  az = a0 + a1
      w0 = a0 / az; w1 = a1 / az
      att = (w0 * fst v0 + w1 * fst v1, w0 * snd v0 + w1 * snd v1)
      hp  = mv wup att
      h   = (sigmoid (fst hp), sigmoid (snd hp))   -- nonlinear readout
  in (dot u0 h, dot u1 h)

-- Squared-error loss to a one-hot target (like ConCat.Deep's errSqr).  Using MSE
-- rather than softmax-NLL keeps the ONLY softmax in the model the attention one
-- — matching the nonlinearity budget of the MLP that compiled fast (a second,
-- loss-side softmax tipped gradR's Core into an unbounded blow-up at this pin).
err0, err1 :: P -> D2 -> D2 -> Double
err0 p ea eb = let (g0, g1) = attnLogits p ea eb in (g0 - 1) * (g0 - 1) + g1 * g1
err1 p ea eb = let (g0, g1) = attnLogits p ea eb in g0 * g0 + (g1 - 1) * (g1 - 1)

-- (a+b) mod 2: (0,0)->0 (0,1)->1 | (1,0)->1 (1,1)->0
chunk1, chunk2 :: P -> Double
chunk1 p = let ((e0, e1), _) = p in err0 p e0 e0 + err1 p e0 e1
chunk2 p = let ((e0, e1), _) = p in err1 p e1 e0 + err0 p e1 e1

grad1, grad2 :: P -> P
grad1 = gradR chunk1
grad2 = gradR chunk2

batchGradSeq :: P -> P
batchGradSeq p = vadd (grad1 p) (grad2 p)

batchGradPar :: P -> P
batchGradPar p =
  let g1 = grad1 p; g2 = grad2 p
  in vforce g1 `par` (vforce g2 `pseq` vadd g1 g2)

gdStep :: (P -> P) -> Double -> P -> P
gdStep bgrad lr p = vadd p (vscale (negate lr) (bgrad p))

totalLoss :: P -> Double
totalLoss p = chunk1 p + chunk2 p

predict :: P -> D2 -> D2 -> Int
predict p ea eb = let (g0, g1) = attnLogits p ea eb in if g0 >= g1 then 0 else 1

p0 :: P
p0 = ( ((0.5, -0.3), (-0.2, 0.6))                       -- emb e0,e1
     , ( ( ((0.4, -0.1), (0.2, 0.3))                    -- Wq
         , ( ((-0.2, 0.5), (0.1, -0.4))                 -- Wk
           , ((0.3, 0.2), (-0.5, 0.1)) ) )              -- Wv
       , ( ((0.3, -0.2), (-0.4, 0.5))                   -- Wup
         , ((0.6, -0.5), (-0.4, 0.7)) ) ) )             -- unembed u0,u1

main :: IO ()
main = do
  let lr = 0.3; n = 40000 :: Int
      go _ 0 p = p
      go bg k p = go bg (k - 1) (gdStep bg lr p)
      trained = go batchGradPar n p0
  printf "Parallel CTC training of a self-attention block on (a+b) mod 2\n"
  printf "params=20 (emb + Wq/Wk/Wv + Wup + unembed, single-head attn, d=2)\n"
  printf "initial loss = %.4f\n" (totalLoss p0)
  printf "final   loss = %.6f (after %d steps; gradient via gradR/toCcc, chunks via par)\n"
    (totalLoss trained) n
  let ((e0, e1), _) = trained
      cases = [ ((0,0), predict trained e0 e0, 0)
              , ((0,1), predict trained e0 e1, 1)
              , ((1,0), predict trained e1 e0, 1)
              , ((1,1), predict trained e1 e1, 0) ]
  mapM_ (\((a,b), got, want) ->
            printf "  (%d+%d) mod 2: pred=%d target=%d %s\n" (a::Int) (b::Int) got want
                   (if got == want then "OK" else "WRONG")) cases
  let correct = length [ () | (_, g, w) <- cases, g == w ]
  t0 <- getCurrentTime
  let !sseq = sum (map totalLoss (iterateGrad batchGradSeq 3000 p0))
  t1 <- getCurrentTime
  let !spar = sum (map totalLoss (iterateGrad batchGradPar 3000 p0))
  t2 <- getCurrentTime
  printf "timing: seq=%.3fs par=%.3fs (checksum seq=%.4f par=%.4f)\n"
    (realToFrac (diffUTCTime t1 t0) :: Double)
    (realToFrac (diffUTCTime t2 t1) :: Double) sseq spar
  if correct == 4
    then putStrLn "ctc self-attention parallel training learned (a+b) mod 2"
    else error "ctc self-attention training did not learn the task"
  where
    iterateGrad bg k p = goi k p where goi 0 q = [q]; goi m q = q : goi (m - 1) (gdStep bg 0.3 q)
