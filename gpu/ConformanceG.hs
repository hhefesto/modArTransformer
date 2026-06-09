{-# LANGUAGE BangPatterns #-}

-- ConformanceG — the GPU tier's FIRST milestone (README §10 Phase 2):
-- the generalized sequence transformer with **our categorical reverse-mode AD
-- over libtorch kernels** — no torch autograd anywhere — reproducing the CPU
-- tape's conformance floats.
--
--   * Same tape design as backend/transformer/Tape.hs: every node owns a
--     mutable cotangent cell (IORef of a torch Tensor); consumers ADD into it;
--     backprop runs the recorded actions once in reverse creation order.
--     Every local adjoint below is the same Dual-category pullback the CPU
--     tape and the Agda spec use — libtorch supplies KERNELS ONLY.
--   * float64, CPU device: this is the conformance flavor (the gate is
--     ≤ 1e-9 against conformance-hs.txt).  The same code runs CUDA by
--     switching the tensor options' device — nothing else changes.
--   * Parameters are the flat leaf list of Transformer.ParamsSeq in Serialize
--     order, so the shared conformance-seq-params.txt deserializes identically.
--
-- Usage:  ConformanceG <conformance-seq-params.txt> <out.txt>
-- Emits, for the same two token sequences as the CPU harness:
--   per-position logits ++ [loss] ++ gradient (flat leaf order).
module Main where

import Control.Monad (forM, foldM, zipWithM, unless)
import Data.IORef
import System.Environment (getArgs)
import System.Exit (exitFailure)
import System.IO (hPutStrLn, stderr)

import Torch.Tensor (Tensor, asTensor, asValue, toDouble, reshape, shape, sliceDim)
import Torch.TensorFactories (onesLike)
import Torch.DType (DType (Double))
import qualified Torch.Functional as F
import Torch.Functional (Dim (..), matmul, transpose2D)

-- outer product from matmul kernels (a ⊗ b = (k×1)·(1×m))
outerT :: Tensor -> Tensor -> Tensor
outerT a b = matmul (reshape [head (shape a), 1] a) (reshape [1, head (shape b)] b)

-- ── model dims (the oracle shape: ParamsSeq 5 4 4 8 2) ────────────────────────

vDim, nDim, dM, dF, dK :: Int
vDim = 5; nDim = 4; dM = 4; dF = 8; dK = 2

-- leaf shapes of ParamsSeq in Serialize order:
-- tok, pos, h1(WqW,WqB,WkW,WkB,WvW,WvB), h2(same), WoW, WoB,
-- LN1γ, LN1β, upW, upB, dnW, dnB, LN2γ, LN2β, unW, unB
leafShapes :: [[Int]]
leafShapes =
  [ [vDim, dM], [nDim, dM] ]
  ++ headShapes ++ headShapes
  ++ [ [dM, 2 * dK], [dM] ]
  ++ [ [dM], [dM] ]
  ++ [ [dF, dM], [dF], [dM, dF], [dM] ]
  ++ [ [dM], [dM] ]
  ++ [ [vDim, dM], [vDim] ]
  where headShapes = [ [dK, dM], [dK], [dK, dM], [dK], [dK, dM], [dK] ]

-- token sequences, mirroring the CPU harness
seqCases :: [[Int]]
seqCases = [ [1, 4, 2, 3], [0, 2, 4, 1] ]

-- ── float64 tensor helpers (kernels only) ─────────────────────────────────────

toT :: [Int] -> [Double] -> Tensor
toT [r, c] xs = reshape [r, c] (asTensor xs)
toT [_]    xs = asTensor xs
toT sh     _  = error ("toT: shape " ++ show sh)

zerosLike :: Tensor -> Tensor
zerosLike t = F.mulScalar (0 :: Double) t

scalarT :: Double -> Tensor
scalarT x = asTensor x

flatList :: Tensor -> [Double]
flatList t = asValue (F.flattenAll t)

-- ── the tape (same design as Tape.hs; IO/IORef instead of ST/STRef) ───────────

data RG = RG { primalG :: !Tensor, adjG :: !(IORef Tensor) }

data TapeG = TapeG
  { actsG :: !(IORef [IO ()])      -- backprop actions, newest first
  , gradG :: !(IORef [Tensor])     -- per-leaf gradient accumulators
  }

addTo :: IORef Tensor -> Tensor -> IO ()
addTo ref d = modifyIORef' ref (`F.add` d)

node :: TapeG -> Tensor -> (Tensor -> IO ()) -> IO RG
node tp !primal propagate = do
  adj <- newIORef (zerosLike primal)
  modifyIORef' (actsG tp) ((readIORef adj >>= propagate) :)
  pure (RG primal adj)

-- parameter leaf input: cotangent flows into the per-leaf accumulator
tInput :: TapeG -> Int -> [Tensor] -> IO RG
tInput tp i leaves =
  node tp (leaves !! i)
          (\d -> modifyIORef' (gradG tp)
                   (\gs -> [ if j == i then F.add g d else g | (j, g) <- zip [0 ..] gs ]))

-- ── primitives: forward kernel + Dual pullback (mirrors Tape.hs 1:1) ──────────

tVadd :: TapeG -> RG -> RG -> IO RG
tVadd tp (RG a ra) (RG b rb) = node tp (F.add a b) (\dy -> addTo ra dy >> addTo rb dy)

tMatvec :: TapeG -> RG -> RG -> IO RG           -- W (o×i) · x (i) → (o)
tMatvec tp (RG w rw) (RG x rx) =
  node tp (matmul w x)
          (\dy -> addTo rw (outerT dy x) >> addTo rx (matmul (transpose2D w) dy))

tVdot :: TapeG -> RG -> RG -> IO RG             -- 0-dim result
tVdot tp (RG a ra) (RG b rb) =
  node tp (F.dot a b) (\d -> addTo ra (F.mul d b) >> addTo rb (F.mul d a))

tScaleC :: TapeG -> Double -> RG -> IO RG
tScaleC tp c (RG x rx) = node tp (F.mulScalar c x) (\d -> addTo rx (F.mulScalar c d))

tAddC :: TapeG -> Double -> RG -> IO RG
tAddC tp c (RG x rx) = node tp (F.addScalar c x) (\d -> addTo rx d)

tExp, tRecip :: TapeG -> RG -> IO RG
tExp   tp (RG x rx) = let e = F.exp x in node tp e (\d -> addTo rx (F.mul d e))
tRecip tp (RG x rx) = let r = F.pow (-1 :: Double) x
                      in node tp r (\d -> addTo rx (F.mulScalar (-1 :: Double) (F.mul d (F.mul r r))))

tAdd, tMul, tSub :: TapeG -> RG -> RG -> IO RG
tAdd tp (RG a ra) (RG b rb) = node tp (F.add a b) (\d -> addTo ra d >> addTo rb d)
tSub tp (RG a ra) (RG b rb) =
  node tp (F.sub a b) (\d -> addTo ra d >> addTo rb (F.mulScalar (-1 :: Double) d))
tMul tp (RG a ra) (RG b rb) =
  node tp (F.mul a b) (\d -> addTo ra (F.mul d b) >> addTo rb (F.mul d a))

tScaleV :: TapeG -> RG -> RG -> IO RG           -- 0-dim s · vector v
tScaleV tp (RG s rs) (RG v rv) =
  node tp (F.mul s v) (\dy -> addTo rs (F.dot dy v) >> addTo rv (F.mul s dy))

tExpV :: TapeG -> RG -> IO RG
tExpV tp (RG x rx) = let e = F.exp x in node tp e (\dy -> addTo rx (F.mul e dy))

tSquareV :: TapeG -> RG -> IO RG
tSquareV tp (RG x rx) =
  node tp (F.mul x x) (\dy -> addTo rx (F.mulScalar (2 :: Double) (F.mul x dy)))

tReluV :: TapeG -> RG -> IO RG
tReluV tp (RG x rx) =
  let mask = F.toDType Double (F.gt x (scalarT 0))
  in node tp (F.relu x) (\dy -> addTo rx (F.mul mask dy))

tVsum :: TapeG -> RG -> IO RG
tVsum tp (RG x rx) = node tp (F.sumAll x) (\d -> addTo rx (F.mul d (onesLike x)))

tMeanV :: TapeG -> RG -> IO RG
tMeanV tp (RG x rx) =
  let n = fromIntegral (tensorLen x) :: Double
  in node tp (F.divScalar n (F.sumAll x))
             (\d -> addTo rx (F.mulScalar (1 / n) (F.mul d (onesLike x))))

tCenter :: TapeG -> RG -> IO RG                 -- x − mean(x); self-adjoint
tCenter tp (RG x rx) =
  let center v = F.sub v (F.divScalar (fromIntegral (tensorLen v) :: Double) (F.sumAll v))
  in node tp (center x) (\dy -> addTo rx (center dy))

tRsqrt :: TapeG -> RG -> IO RG                  -- 0-dim 1/√x
tRsqrt tp (RG x rx) =
  let r = F.pow (-0.5 :: Double) x
  in node tp r (\d -> addTo rx (F.mulScalar (-0.5 :: Double) (F.div (F.mul d r) x)))

tHadamard :: TapeG -> RG -> RG -> IO RG
tHadamard tp (RG a ra) (RG b rb) =
  node tp (F.mul a b) (\dy -> addTo ra (F.mul b dy) >> addTo rb (F.mul a dy))

tSelect :: TapeG -> Int -> RG -> IO RG          -- coordinate t; kernel = dot with e_t
tSelect tp i (RG x rx) =
  let oh = oneHotT (tensorLen x) i
  in node tp (F.dot x oh) (\d -> addTo rx (F.mul d oh))

tDetachMax :: TapeG -> RG -> IO RG              -- subtract detached max; identity adjoint
tDetachMax tp (RG x rx) =
  let m = maximum (flatList x)
  in node tp (F.subScalar m x) (\dy -> addTo rx dy)

tEmbedRow :: TapeG -> Int -> RG -> IO RG        -- row i; adjoint = outer(e_i, dv)
tEmbedRow tp i (RG w rw) =
  let r  = head (tensorShape w)
      oh = oneHotT r i
  in node tp (matmul oh w) (\dv -> addTo rw (outerT oh dv))

tConcatV :: TapeG -> RG -> RG -> IO RG          -- concat; adjoint splits
tConcatV tp (RG x rx) (RG y ry) =
  let k = tensorLen x
  in node tp (F.cat (Dim 0) [x, y])
             (\dz -> do addTo rx (sliceDim 0 0 k 1 dz)
                        addTo ry (sliceDim 0 k (tensorLen x + tensorLen y) 1 dz))

oneHotT :: Int -> Int -> Tensor
oneHotT k i = asTensor [ if j == i then 1 else 0 :: Double | j <- [0 .. k - 1] ]

tensorShape :: Tensor -> [Int]
tensorShape = shape

tensorLen :: Tensor -> Int
tensorLen t = product (tensorShape t)

-- ── the forward (mirrors Transformer.forwardSeqT line by line) ────────────────

affineT :: TapeG -> RG -> RG -> RG -> IO RG
affineT tp rw rb rx = do wx <- tMatvec tp rw rx; tVadd tp wx rb

layerNormT :: TapeG -> RG -> RG -> RG -> IO RG
layerNormT tp rg rb rx = do
  xc   <- tCenter tp rx
  sq   <- tSquareV tp xc
  var  <- tMeanV tp sq
  var' <- tAddC tp 1.0e-5 var
  inv  <- tRsqrt tp var'
  norm <- tScaleV tp inv xc
  gn   <- tHadamard tp rg norm
  tVadd tp gn rb

fold1M :: (a -> a -> IO a) -> [a] -> IO a
fold1M f (x : xs) = foldM f x xs
fold1M _ []       = error "fold1M: empty"

-- leaf indices (see leafShapes)
lTok, lPos, lWoW, lWoB, lG1, lB1, lUpW, lUpB, lDnW, lDnB, lG2, lB2, lUnW, lUnB :: Int
lTok = 0; lPos = 1
lWoW = 14; lWoB = 15; lG1 = 16; lB1 = 17
lUpW = 18; lUpB = 19; lDnW = 20; lDnB = 21
lG2 = 22; lB2 = 23; lUnW = 24; lUnB = 25

headBase :: Int -> Int           -- head h (0|1) → first leaf index
headBase h = 2 + 6 * h

forwardSeqG :: TapeG -> [Tensor] -> [Int] -> IO [RG]
forwardSeqG tp leaves toks = do
  rTok <- tInput tp lTok leaves
  rPos <- tInput tp lPos leaves
  embeds <- forM (zip [0 ..] toks) $ \(i, tk) -> do
    et <- tEmbedRow tp tk rTok
    ep <- tEmbedRow tp i rPos
    tVadd tp et ep
  let sc = 1.0 / sqrt (fromIntegral dK)
      headOuts h = do
        let b = headBase h
        qW <- tInput tp b       leaves ; qB <- tInput tp (b + 1) leaves
        kW <- tInput tp (b + 2) leaves ; kB <- tInput tp (b + 3) leaves
        vW <- tInput tp (b + 4) leaves ; vB <- tInput tp (b + 5) leaves
        qs <- mapM (affineT tp qW qB) embeds
        ks <- mapM (affineT tp kW kB) embeds
        vs <- mapM (affineT tp vW vB) embeds
        forM (zip [0 ..] qs) $ \(i, qi) -> do
          let ksA = take (i + 1) ks
              vsA = take (i + 1) vs
          ds  <- mapM (tVdot tp qi) ksA
          ss  <- mapM (tScaleC tp sc) ds
          let sm = maximum (map (toDouble . primalG) ss)
          ss' <- mapM (tAddC tp (negate sm)) ss
          es  <- mapM (tExp tp) ss'
          z   <- fold1M (tAdd tp) es
          rz  <- tRecip tp z
          ws  <- mapM (\e -> tMul tp e rz) es
          avs <- zipWithM (tScaleV tp) ws vsA
          fold1M (tVadd tp) avs
  h1 <- headOuts 0
  h2 <- headOuts 1
  oW  <- tInput tp lWoW leaves ; oB  <- tInput tp lWoB leaves
  g1  <- tInput tp lG1  leaves ; b1  <- tInput tp lB1  leaves
  upW <- tInput tp lUpW leaves ; upB <- tInput tp lUpB leaves
  dnW <- tInput tp lDnW leaves ; dnB <- tInput tp lDnB leaves
  g2  <- tInput tp lG2  leaves ; b2  <- tInput tp lB2  leaves
  uW  <- tInput tp lUnW leaves ; uB  <- tInput tp lUnB leaves
  forM (zip3 embeds h1 h2) $ \(x, a1, a2) -> do
    cat <- tConcatV tp a1 a2
    ao  <- affineT tp oW oB cat
    r10 <- tVadd tp x ao
    n10 <- layerNormT tp g1 b1 r10
    hh  <- affineT tp upW upB n10
    hr  <- tReluV tp hh
    ff  <- affineT tp dnW dnB hr
    r20 <- tVadd tp n10 ff
    o   <- layerNormT tp g2 b2 r20
    affineT tp uW uB o

-- mean next-token CE (mirrors seqGradLoss's build)
seqLossG :: TapeG -> [Tensor] -> [Int] -> IO ([RG], RG)
seqLossG tp leaves toks = do
  logits <- forwardSeqG tp leaves toks
  ls <- forM (zip (init logits) (drop 1 toks)) $ \(lg, t) -> do
    l2  <- tDetachMax tp lg
    e   <- tExpV tp l2
    s   <- tVsum tp e
    lse <- tLog tp s
    sel <- tSelect tp t l2
    tSub tp lse sel
  tot <- fold1M (tAdd tp) ls
  loss <- tScaleC tp (1 / fromIntegral (length ls)) tot
  pure (logits, loss)
  where tLog tp' (RG x rx) = node tp' (F.log x) (\d -> addTo rx (F.div d x))

-- run forward+backward; return (per-position logits, loss, flat grad)
gradLossG :: [Tensor] -> [Int] -> IO ([[Double]], Double, [Double])
gradLossG leaves toks = do
  acts <- newIORef []
  gref <- newIORef (map zerosLike leaves)
  let tp = TapeG acts gref
  (logits, RG lossT lossAdj) <- seqLossG tp leaves toks
  writeIORef lossAdj (scalarT 1)
  as <- readIORef acts
  sequence_ as
  gs <- readIORef gref
  pure ( map (flatList . primalG) logits
       , toDouble lossT
       , concatMap flatList gs )

main :: IO ()
main = do
  args <- getArgs
  (inP, outP) <- case args of
    [a, b] -> pure (a, b)
    _      -> hPutStrLn stderr "usage: ConformanceG <params.txt> <out.txt>" >> exitFailure
  nums <- map read . lines <$> readFile inP :: IO [Double]
  let sizes = map product leafShapes
  unless (length nums == sum sizes) $ do
    hPutStrLn stderr ("param count " ++ show (length nums) ++ " /= " ++ show (sum sizes))
    exitFailure
  let leaves = go nums leafShapes
        where go _  []        = []
              go xs (sh : ss) = let (h, t) = splitAt (product sh) xs in toT sh h : go t ss
  outs <- forM seqCases $ \toks -> do
    (logits, loss, grad) <- gradLossG leaves toks
    pure (concat logits ++ (loss : grad))
  writeFile outP (unlines (map show (concat outs)))
  putStrLn ("[gpu-tier] wrote " ++ outP)
