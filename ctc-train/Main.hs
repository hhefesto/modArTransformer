{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- Milestone A: end-to-end *training* driven by a Compile-to-Categories gradient.
--
-- The per-step gradient of the loss is produced by Conal's plugin: `gradR (toCcc
-- loss)` (reverse-mode AD via GD (Dual AdditiveFun)), the path unblocked in
-- ctc-grad-smoke.  The optimisation loop itself is ordinary Haskell that simply
-- *calls* the compiled gradient — so the learning is genuinely CTC-driven.
--
-- Task: least-squares fit of a line y = w*x + b to points on y = 2x.  Params are
-- (w, b); we expect w -> 2, b -> 0, loss -> 0.
module Main where

import ConCat.RAD (gradR)
import ConCat.Rebox ()

-- Training points (x, y) lying on y = 2x.
pts :: [(Double, Double)]
pts = [(0, 0), (1, 2), (2, 4), (3, 6), (4, 8)]

-- Loss as a pure function of the parameters (w, b).  This is the lambda the
-- plugin elaborates into the differentiable category.
loss :: (Double, Double) -> Double
loss (w, b) =
  let sq e = e * e
  in sq (w * 0 + b - 0)
   + sq (w * 1 + b - 2)
   + sq (w * 2 + b - 4)
   + sq (w * 3 + b - 6)
   + sq (w * 4 + b - 8)

-- The gradient, compiled through toCcc (reverse mode).  No hand-written backward.
lossGrad :: (Double, Double) -> (Double, Double)
lossGrad = gradR loss

-- One gradient-descent step (ordinary Haskell calling the compiled gradient).
step :: Double -> (Double, Double) -> (Double, Double)
step lr (w, b) =
  let (gw, gb) = lossGrad (w, b)
  in (w - lr * gw, b - lr * gb)

main :: IO ()
main = do
  let lr     = 0.02
      p0     = (0.0, 0.0)
      iters  = iterate (step lr) p0
      report k = let p = iters !! k in
                   putStrLn ("iter " ++ show k ++ ": (w,b)=" ++ show p
                             ++ " loss=" ++ show (loss p))
  putStrLn "CTC-gradient training (line fit y=2x); gradient via gradR (toCcc):"
  mapM_ report [0, 50, 100, 200, 400, 800]
  let (wf, bf) = iters !! 800
  if abs (wf - 2) < 1e-3 && abs bf < 1e-3
    then putStrLn "ctc train converged (w~2, b~0)"
    else error "ctc train did not converge"
