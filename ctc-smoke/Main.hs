module Main where

import ConCat.AltCat (toCcc)

-- Stage 0: Bool projection (structural; no numerics).
ctcFirst :: (Bool, Bool) -> Bool
ctcFirst = toCcc @(->) @(Bool, Bool) @Bool (\(x, _y) -> x)

-- Stage 1: scalar Double arithmetic — exercises NumCat/FloatingCat elaboration
-- (this is where the unboxed Double# panic is expected, if any).
ctcAffine :: (Double, Double) -> Double
ctcAffine = toCcc @(->) @(Double, Double) @Double (\(x, y) -> x * y + 1)

-- Stage 2: a numeric kernel — a 2-D dot product (mul + add over Doubles).  This
-- is the smallest thing shaped like the transformer's real ops (vdot/matvec).
ctcDot2 :: ((Double, Double), (Double, Double)) -> Double
ctcDot2 = toCcc @(->) (\((a, b), (c, d)) -> a * c + b * d)

check :: (Eq a, Show a) => String -> a -> a -> IO ()
check name got want = do
  putStrLn (name ++ ": ctc=" ++ show got ++ " direct=" ++ show want)
  if got == want then pure () else error (name ++ " mismatch")

main :: IO ()
main = do
  check "stage0 first"  (ctcFirst (True, False))      (fst (True, False))
  check "stage1 affine" (ctcAffine (3, 4))            (3 * 4 + 1)
  check "stage2 dot2"   (ctcDot2 ((1, 2), (3, 4)))    (1 * 3 + 2 * 4)
  putStrLn "ctc smoke passed"
