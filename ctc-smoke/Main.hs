module Main where

import ConCat.AltCat (toCcc)

ctcFirst :: (Bool, Bool) -> Bool
ctcFirst = toCcc @(->) @(Bool, Bool) @Bool (\(x, _y) -> x)

main :: IO ()
main = do
  let input = (True, False)
      y = ctcFirst input
      expected = fst input
  putStrLn ("ctc first = " ++ show y)
  putStrLn ("direct first = " ++ show expected)
  if y == expected
    then putStrLn "ctc smoke passed"
    else error "ctc smoke mismatch"
