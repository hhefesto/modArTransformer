module Main where

import ConCat.AD (gradient)
import ConCat.Rebox ()

-- This currently documents the heavier CTC-gradient acceptance gate.  The main
-- ctc-smoke package stays as the green Stage 0-2 forward/numeric-kernel gate.
ctcGrad :: (Double, Double) -> (Double, Double)
ctcGrad = gradient (\(x, y) -> x * x + y * y)

main :: IO ()
main = do
  let got = ctcGrad (3, 4)
      want = (6, 8)
  putStrLn ("stage3 grad: ctc=" ++ show got ++ " direct=" ++ show want)
  if got == want then putStrLn "ctc gradient smoke passed" else error "ctc gradient mismatch"
