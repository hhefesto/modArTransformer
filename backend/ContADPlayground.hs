{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}

-- A small executable playground for the next backend direction.
--
-- The Agda code remains the formal specification.  This file is deliberately a
-- tiny Haskell runtime experiment for Conal Elliott's representation choice:
-- compare ordinary Dual pullbacks with continuation-style reverse AD.
--
-- Current Agda shape:
--   D A B = A -> (B, Dual A B)
--   Dual A B = AddFun B A
--
-- Continuation shape from Conal's notes:
--   Cont r A B = (B -o r) -> (A -o r)
--
-- For a scalar loss A -> Double and r = Double, the final continuation is the
-- identity linear functional.  Applying the resulting input linear functional to
-- the scalar tangent 1 extracts the gradient.

module Main where

import Text.Printf (printf)
import AD
import qualified Cat as C
import qualified Tensor as T

ctcShapedToy :: Double -> (Double, Double)
ctcShapedToy = C.forkC (\x -> x * x) (\x -> x + 1)

squareDual :: DDual Double Double
squareDual = DDual $ \x ->
  (x * x, Dual (AddFun (\dy -> (2 * x) * dy)))

addConstDual :: Double -> DDual Double Double
addConstDual c = DDual $ \x ->
  (x + c, Dual idL)

squareCont :: DCont Double Double Double
squareCont = DCont $ \x ->
  ( x * x
  , Cont $ \(AddFun k) -> AddFun (\dx -> k ((2 * x) * dx))
  )

addConstCont :: Double -> DCont Double Double Double
addConstCont c = DCont $ \x ->
  ( x + c
  , Cont id
  )

matvecDual :: DDual (T.Mat m n, T.Vec n) (T.Vec m)
matvecDual = DDual $ \(m, x) ->
  ( T.matvec m x
  , Dual (AddFun (\dy -> (T.outer dy x, T.matvec (T.transpose m) dy)))
  )

matvecCont :: DCont Double (T.Mat m n, T.Vec n) (T.Vec m)
matvecCont = DCont $ \(m, x) ->
  ( T.matvec m x
  , Cont $ \(AddFun k) -> AddFun $ \(dm, dx) ->
      k (T.vadd (T.matvec dm x) (T.matvec m dx))
  )

pairDot :: (T.Mat m n, T.Vec n) -> (T.Mat m n, T.Vec n) -> Double
pairDot (m1, v1) (m2, v2) = matDot m1 m2 + T.vdot v1 v2
  where
    matDot (T.Mat rowsA) (T.Mat rowsB) = sum (zipWith rowDot rowsA rowsB)
    rowDot xs ys = sum (zipWith (*) xs ys)

-- h x = (x^2 + 1)^2
programDual :: DDual Double Double
programDual = squareDual `composeDDual` addConstDual 1 `composeDDual` squareDual

programCont :: DCont Double Double Double
programCont = squareCont `composeDCont` addConstCont 1 `composeDCont` squareCont

gradDual :: DDual Double Double -> Double -> (Double, Double)
gradDual (DDual f) x =
  let (y, Dual pb) = f x
  in (y, applyL pb 1)

gradCont :: DCont Double Double Double -> Double -> (Double, Double)
gradCont (DCont f) x =
  let (y, cont) = f x
      inputFunctional = runCont cont idL
  in (y, applyL inputFunctional 1)

analytic :: Double -> (Double, Double)
analytic x =
  let y = (x * x + 1) * (x * x + 1)
      dy = 4 * x * (x * x + 1)
  in (y, dy)

row :: Double -> IO ()
row x = do
  let (yd, gd) = gradDual programDual x
      (yc, gc) = gradCont programCont x
      (ya, ga) = analytic x
  printf "%6.2f  dual=(%10.6f,%10.6f)  cont=(%10.6f,%10.6f)  analytic=(%10.6f,%10.6f)\n"
    x yd gd yc gc ya ga

matvecRow :: IO ()
matvecRow = do
  let m = T.mat @2 @3 [[1, 2, 3], [4, 5, 6]]
      x = T.vec @3 [0.5, -1, 2]
      seed = T.vec @2 [3, -2]
      tangent = (T.mat @2 @3 [[0.1, 0.2, -0.3], [0.4, -0.5, 0.6]], T.vec @3 [0.7, -0.8, 0.9])
      (yD, Dual pb) = runDDual matvecDual (m, x)
      gradInput = applyL pb seed
      directionalDual = pairDot gradInput tangent
      (yC, cont) = runDCont matvecCont (m, x)
      inputFunctional = runCont cont (AddFun (T.vdot seed))
      directionalCont = applyL inputFunctional tangent
      directDirectional = T.vdot seed (T.vadd (T.matvec (fst tangent) x) (T.matvec m (snd tangent)))
  putStrLn ""
  putStrLn "typed tensor primitive: matvec : Mat 2 3 -> Vec 3 -> Vec 2"
  putStrLn $ "matvec output Dual = " <> show yD
  putStrLn $ "matvec output Cont = " <> show yC
  printf "directional derivative: dual=%10.6f  cont=%10.6f  direct=%10.6f\n"
    directionalDual directionalCont directDirectional

main :: IO ()
main = do
  putStrLn "Conal-style continuation AD playground"
  putStrLn $ "CTC-shaped cartesian target smoke test at 2: " <> show (ctcShapedToy 2)
  putStrLn "program: h x = (x^2 + 1)^2"
  putStrLn "columns: x, (value, gradient) for Dual, Cont, analytic"
  mapM_ row [-3, -1, 0, 0.5, 2]
  matvecRow
