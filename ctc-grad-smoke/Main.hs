{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Main where

-- Reverse-mode gradient via ConCat.RAD = GD (Dual (-+>)) (Dual AdditiveFun),
-- the path ConCat's own tests use (andGradR/andGrad2R).  This avoids the
-- LinearRow `L s` row-matrix representation that ConCat.AD.gradient uses, whose
-- free-vector-space Pointed instances ($fPointed:*:/$fPointedPar1) made the GHC
-- simplifier loop.  gradR only needs `Num s`.
import ConCat.RAD (gradR)
import ConCat.Rebox ()

-- grad (x^2 + y^2) = (2x, 2y); at (3,4) -> (6,8).
ctcGrad :: (Double, Double) -> (Double, Double)
ctcGrad = gradR (\(x, y) -> x * x + y * y)

main :: IO ()
main = do
  let got = ctcGrad (3, 4)
      want = (6, 8)
  putStrLn ("stage3 grad: ctc=" ++ show got ++ " direct=" ++ show want)
  if got == want then putStrLn "ctc gradient smoke passed" else error "ctc gradient mismatch"
