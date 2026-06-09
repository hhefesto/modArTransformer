{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -Wno-simplifiable-class-constraints #-}

-- Enriched-semantics probe (Phase A): does the TRAINED (a+b) mod p model realize
-- Tai-Danae Bradley's enriched-category structure?
--
--   * Meaning = copresheaf (Yoneda): contexts (a,b) with the same sum c=(a+b)%p are
--     EXACT synonyms (identical continuation distribution = Dirac on c). We measure
--     whether the model's copresheaf softmax(logits) collapses within each sum-class.
--   * The space of meanings = the group ℤ/p (the "grokking circle"): we average the
--     pre-unembed meaning vector per sum-class and show the p class-means lie on a
--     circle (PCA→2D + circle fit + a single dominant DFT frequency), i.e. the
--     learned meaning space carries the cyclic-group structure.
--
-- This is read-only over a trained checkpoint; it changes nothing about the model.
module Main where

import Data.Complex (Complex(..), magnitude, mkPolar)
import Data.List (foldl', maximumBy)
import Data.Ord (comparing)
import Data.Proxy (Proxy(..))
import GHC.TypeNats (natVal)
import qualified Numeric.LinearAlgebra as LA
import qualified Options.Applicative as O
import Text.Printf (printf)

import Serialize (fromFloats)
import Tensor (vtoList)
import Transformer (Params2, transformerLogitsVal2, transformerReprVal2)

-- The trained production model: p=97, dModel=64, dFF=256, dK=64.
type P = Params2 97 64 256 64

pI :: Int
pI = fromIntegral (natVal (Proxy @97))

-- ── load just the parameters from a checkpoint (header line + flat floats) ──────

loadParams :: FilePath -> IO P
loadParams path = do
  contents <- readFile path
  let rows = drop 1 (lines contents)          -- drop the CHECKPOINT_ADAMW header
      nums = map read rows :: [Double]
      (params, _) = fromFloats nums            -- fromFloats consumes the param prefix
  params `seq` pure params

-- ── small numeric helpers ───────────────────────────────────────────────────────

softmax :: [Double] -> [Double]
softmax xs = let m = maximum xs; es = map (\x -> exp (x - m)) xs; z = sum es
             in map (/ z) es

-- KL(p ‖ q) in nats, with a floor on q to stay finite.
klDiv :: [Double] -> [Double] -> Double
klDiv ps qs = sum [ if p <= 0 then 0 else p * log (p / max q 1e-12)
                  | (p, q) <- zip ps qs ]

mean :: [Double] -> Double
mean xs = sum xs / fromIntegral (length xs)

-- elementwise mean of a list of equal-length vectors
meanVec :: [[Double]] -> [Double]
meanVec vss = map (/ fromIntegral (length vss)) (foldl1 (zipWith (+)) vss)

sqDist :: [Double] -> [Double] -> Double
sqDist a b = sum [ (x - y) * (x - y) | (x, y) <- zip a b ]

-- ── the probe ───────────────────────────────────────────────────────────────────

data Ctx = Ctx { cxA :: !Int, cxB :: !Int, cxClass :: !Int }

run :: FilePath -> FilePath -> IO ()
run ckpt csvOut = do
  params <- loadParams ckpt
  let ctxs = [ Ctx a b ((a + b) `mod` pI) | a <- [0 .. pI - 1], b <- [0 .. pI - 1] ]
      -- per context: copresheaf (softmax of logits) and meaning vector (repr)
      copre c = softmax (vtoList (transformerLogitsVal2 (cxA c) (cxB c) params))
      repr  c = vtoList (transformerReprVal2 (cxA c) (cxB c) params)
      classes = [0 .. pI - 1]
      byClass f = [ [ f c | c <- ctxs, cxClass c == k ] | k <- classes ]

  printf "════════════════════════════════════════════════════════════════════════\n"
  printf " Enriched-semantics probe — (a+b) mod %d   [checkpoint: %s]\n" pI ckpt
  printf "════════════════════════════════════════════════════════════════════════\n"

  -- sanity: is this model actually grokked?
  let acc = mean [ if argmax (vtoList (transformerLogitsVal2 (cxA c) (cxB c) params)) == cxClass c
                     then 1 else 0 | c <- ctxs ]
      loss = mean [ negate (log (max (copre c !! cxClass c) 1e-12)) | c <- ctxs ]
  printf " Model: accuracy=%.2f%%  mean cross-entropy (relative entropy to Dirac truth)=%.4f nats\n\n"
    (100 * acc) loss

  -- ── Metric 1: Yoneda synonymy collapse ─────────────────────────────────────────
  let copsByClass  = byClass copre                      -- [[copresheaf]] grouped by sum
      centroidCop  = map meanVec copsByClass            -- mean copresheaf per class
      intraKL = mean [ klDiv m cen
                     | (ms, cen) <- zip copsByClass centroidCop, m <- ms ]
      interKL = mean [ klDiv (centroidCop !! i) (centroidCop !! j)
                     | i <- classes, j <- classes, i /= j ]
      reprByClass = byClass repr
      reprCentroid = map meanVec reprByClass
      globalMean   = meanVec (concat reprByClass)
      ssWithin = sum [ sqDist r cen | (rs, cen) <- zip reprByClass reprCentroid, r <- rs ]
      ssTotal  = sum [ sqDist r globalMean | rs <- reprByClass, r <- rs ]
      varExplained = 1 - ssWithin / ssTotal             -- fraction of repr variance due to the sum
  printf " Metric 1 — Yoneda synonymy collapse (equal-sum contexts are synonyms)\n"
  printf "   copresheaf KL: intra-class=%.4g  inter-class=%.4g  ratio=%.1fx\n"
    intraKL interKL (interKL / max intraKL 1e-12)
  printf "   meaning-vector variance explained by sum-class: %.4f  (1.0 = perfect collapse)\n\n"
    varExplained

  -- ── Metric 2: the meaning space is ℤ/p (the grokking circle) ───────────────────
  -- PCA the p class-mean meaning vectors to 2D.
  let n = length classes
      d = length globalMean
      centeredRows = [ zipWith (-) cen globalMean | cen <- reprCentroid ]
      matA = LA.fromLists centeredRows :: LA.Matrix Double   -- p × d
      (_u, svals, vmat) = LA.svd matA                         -- columns of v = principal axes
      pcs = LA.takeColumns 2 vmat                             -- d × 2
      proj = matA LA.<> pcs                                   -- p × 2
      pts = [ (LA.atIndex r 0, LA.atIndex r 1) | r <- LA.toRows proj ]  -- [(x,y)] per class
      -- circle fit (Kåsa algebraic): x²+y² = A x + B y + C
      lhs = LA.fromLists [ [x, y, 1] | (x, y) <- pts ]
      rhsv = LA.fromList [ x*x + y*y | (x, y) <- pts ]
      sol = LA.flatten (LA.linearSolveLS lhs (LA.asColumn rhsv))
      (aa, bb, cc) = (LA.atIndex sol 0, LA.atIndex sol 1, LA.atIndex sol 2)
      cx = aa / 2; cy = bb / 2
      radius = sqrt (cc + (aa*aa + bb*bb) / 4)
      dists = [ sqrt ((x - cx)^(2::Int) + (y - cy)^(2::Int)) | (x, y) <- pts ]
      radiusCV = stddev dists / mean dists                   -- 0 = perfect circle
      -- dominant DFT frequency across class index (group order) of z_c = x + iy
      zs = [ x :+ y | (x, y) <- pts ]
      energy k = magnitude (sum [ z * mkPolar 1 (-2 * pi * fromIntegral (k*c) / fromIntegral n)
                                | (c, z) <- zip [0 ..] zs ]) ^ (2::Int)
      ks = [1 .. n - 1]
      domK = maximumBy (comparing energy) ks
      totalE = sum (map energy ks)
      domShare = energy domK / totalE
  printf " Metric 2 — meaning space recovers ℤ/%d (the grokking circle)\n" pI
  printf "   PCA(2D) of the %d sum-class meanings: top-2 of %d dims capture %.1f%% of variance\n"
    n d (100 * dimVarFrac svals)
  printf "   circle fit: radius=%.4f  radius CV=%.4f  (0 = points equidistant from centre)\n"
    radius radiusCV
  printf "   dominant DFT frequency across sum order: k=%d  energy share=%.1f%%  (1 peak ⇒ circular)\n\n"
    domK (100 * domShare)

  -- CSV for plotting the circle
  writeFile csvOut $ unlines $
    "class,x,y" : [ printf "%d,%.6f,%.6f" c x y | (c, (x, y)) <- zip [0 :: Int ..] pts ]
  printf " Wrote 2D class-mean projection to %s (columns: class,x,y) — plot to see the circle.\n" csvOut
  where
    argmax xs = snd (maximumBy (comparing fst) (zip xs [0 :: Int ..]))
    stddev xs = let m = mean xs in sqrt (mean [ (x - m)^(2::Int) | x <- xs ])
    -- fraction of total variance in the top-2 singular values
    dimVarFrac s = let sv = LA.toList s; e = map (^(2::Int)) sv
                   in sum (take 2 e) / sum e

-- ── CLI ─────────────────────────────────────────────────────────────────────────

data Opts = Opts { optCkpt :: FilePath, optCsv :: FilePath }

optsP :: O.Parser Opts
optsP = Opts
  <$> O.strOption (O.long "checkpoint" <> O.short 'c' <> O.value "checkpoint-p97l2.ckpt"
        <> O.showDefault <> O.metavar "PATH" <> O.help "trained p97l2 checkpoint to probe")
  <*> O.strOption (O.long "csv" <> O.value "enriched-modp-circle.csv" <> O.showDefault
        <> O.metavar "PATH" <> O.help "output CSV of the 2D meaning projection")

main :: IO ()
main = do
  o <- O.execParser $ O.info (optsP O.<**> O.helper)
    (O.fullDesc <> O.header "enriched-probe — does the trained model realize enriched semantics?")
  run (optCkpt o) (optCsv o)
