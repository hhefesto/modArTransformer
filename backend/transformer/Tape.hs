{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE BangPatterns #-}

-- Reverse-mode AD with a Wengert tape (ST).  Each node owns a mutable cotangent
-- cell; consumers ADD into it.  Backprop runs the recorded nodes once in reverse
-- creation order (a valid reverse-topological order, since a node is created
-- before any consumer), so each node's cotangent is fully accumulated before it
-- propagates to its parents.  This is the correct reverse mode: O(graph) forward
-- and backward, each node touched once — no diamond blowup.
--
-- The local adjoints are exactly the Conal Dual-category adjoints; the tape just
-- sequences their accumulation.  Still no hand-written backward for any composite.
module Tape
  ( R, primalR
  , Tape, tGradLoss, tEval
  , tInput, tConst
  , tAdd, tSub, tMul, tExp, tLog, tRecip, tRsqrt, tScaleC, tAddC
  , tVadd, tMatvec, tHadamard, tVdot, tScaleV, tExpV, tSquareV, tReluV
  , tVsum, tMeanV, tCenter, tSelect, tDetachMax, tEmbedRow
  ) where

import Control.Monad.ST
import Data.STRef
import GHC.TypeNats (KnownNat)
import AD (Lens(..))
import Tensor

-- A node: primal value + mutable cotangent accumulator.
data R s x = R { primalR :: !x, adjR :: !(STRef s x) }

data Tape s p = Tape
  { tapeActs :: !(STRef s [ST s ()])   -- backprop actions, newest first
  , tapeGrad :: !(STRef s p)           -- parameter-gradient accumulator
  }

addTo :: Additive x => STRef s x -> x -> ST s ()
addTo ref d = modifySTRef' ref (`addA` d)
{-# INLINE addTo #-}

-- create a node: fresh zero cotangent cell + a registered backprop action that
-- reads the (by-then fully accumulated) cotangent and propagates it to parents.
node :: Additive x => Tape s p -> x -> (x -> ST s ()) -> ST s (R s x)
node tp !primal propagate = do
  adj <- newSTRef zeroA
  modifySTRef' (tapeActs tp) ((readSTRef adj >>= propagate) :)
  pure (R primal adj)
{-# INLINE node #-}

-- ── parameter leaf / constant ─────────────────────────────────────────────────

tInput :: Additive a => Tape s p -> Lens p a -> p -> ST s (R s a)
tInput tp l p = node tp (lget l p) (\d -> modifySTRef' (tapeGrad tp) (lmod l (`addA` d)))

tConst :: Additive x => Tape s p -> x -> ST s (R s x)
tConst tp x = node tp x (\_ -> pure ())

-- ── scalar ──────────────────────────────────────────────────────────────────

tAdd, tSub, tMul :: Tape s p -> R s Double -> R s Double -> ST s (R s Double)
tAdd tp (R a ra) (R b rb) = node tp (a + b) (\d -> addTo ra d >> addTo rb d)
tSub tp (R a ra) (R b rb) = node tp (a - b) (\d -> addTo ra d >> addTo rb (negate d))
tMul tp (R a ra) (R b rb) = node tp (a * b) (\d -> addTo ra (d * b) >> addTo rb (d * a))

tExp, tLog, tRecip, tRsqrt :: Tape s p -> R s Double -> ST s (R s Double)
tExp   tp (R x rx) = let !e = exp x    in node tp e (\d -> addTo rx (d * e))
tLog   tp (R x rx) =                       node tp (log x) (\d -> addTo rx (d / x))
tRecip tp (R x rx) = let !r = 1 / x    in node tp r (\d -> addTo rx (negate d * r * r))
tRsqrt tp (R x rx) = let !r = 1/sqrt x in node tp r (\d -> addTo rx (d * (-0.5) * r / x))

tScaleC :: Tape s p -> Double -> R s Double -> ST s (R s Double)
tScaleC tp c (R x rx) = node tp (c * x) (\d -> addTo rx (c * d))

tAddC :: Tape s p -> Double -> R s Double -> ST s (R s Double)
tAddC tp c (R x rx) = node tp (x + c) (\d -> addTo rx d)

-- ── vector ────────────────────────────────────────────────────────────────────

tVadd :: KnownNat n => Tape s p -> R s (V n) -> R s (V n) -> ST s (R s (V n))
tVadd tp (R a ra) (R b rb) = node tp (vaddT a b) (\dy -> addTo ra dy >> addTo rb dy)

tMatvec :: (KnownNat m, KnownNat n)
        => Tape s p -> R s (M m n) -> R s (V n) -> ST s (R s (V m))
tMatvec tp (R w rw) (R x rx) =
  node tp (matvec w x) (\dy -> addTo rw (vouter dy x) >> addTo rx (matvec (mtr w) dy))

tHadamard :: KnownNat n => Tape s p -> R s (V n) -> R s (V n) -> ST s (R s (V n))
tHadamard tp (R a ra) (R b rb) =
  node tp (vzipT (*) a b) (\dy -> addTo ra (vzipT (*) b dy) >> addTo rb (vzipT (*) a dy))

tVdot :: KnownNat n => Tape s p -> R s (V n) -> R s (V n) -> ST s (R s Double)
tVdot tp (R a ra) (R b rb) =
  node tp (dotT a b) (\d -> addTo ra (vscaleT d b) >> addTo rb (vscaleT d a))

tScaleV :: KnownNat n => Tape s p -> R s Double -> R s (V n) -> ST s (R s (V n))
tScaleV tp (R s' rs) (R v rv) =
  node tp (vscaleT s' v) (\dy -> addTo rs (dotT dy v) >> addTo rv (vscaleT s' dy))

tExpV :: KnownNat n => Tape s p -> R s (V n) -> ST s (R s (V n))
tExpV tp (R x rx) = let !e = vmapT exp x in node tp e (\dy -> addTo rx (vzipT (*) e dy))

tSquareV :: KnownNat n => Tape s p -> R s (V n) -> ST s (R s (V n))
tSquareV tp (R x rx) =
  node tp (vmapT (\t -> t*t) x) (\dy -> addTo rx (vzipT (\xi d -> 2*xi*d) x dy))

tReluV :: KnownNat n => Tape s p -> R s (V n) -> ST s (R s (V n))
tReluV tp (R x rx) =
  node tp (vmapT (\t -> if t>0 then t else 0) x)
          (\dy -> addTo rx (vzipT (\xi d -> if xi>0 then d else 0) x dy))

tVsum :: KnownNat n => Tape s p -> R s (V n) -> ST s (R s Double)
tVsum tp (R x rx) = node tp (vsumElems x) (\d -> addTo rx (vmapT (const d) x))

tMeanV :: KnownNat n => Tape s p -> R s (V n) -> ST s (R s Double)
tMeanV tp (R x rx) = let !n = fromIntegral (vdim x)
                     in node tp (vsumElems x / n) (\d -> addTo rx (vmapT (const (d/n)) x))

-- centering x ↦ x − mean(x)·1 ; symmetric linear map ⇒ adjoint = same centering
tCenter :: KnownNat n => Tape s p -> R s (V n) -> ST s (R s (V n))
tCenter tp (R x rx) =
  let !n = fromIntegral (vdim x)
      center v = let !m = vsumElems v / n in vmapT (subtract m) v
  in node tp (center x) (\dy -> addTo rx (center dy))

tSelect :: forall s p n. KnownNat n => Tape s p -> Int -> R s (V n) -> ST s (R s Double)
tSelect tp i (R x rx) =
  node tp (vindex x i) (\d -> addTo rx (vfromList @n [ if j==i then d else 0 | j <- [0..nI-1] ]))
  where nI = vdim x

-- detached max subtraction (identity adjoint; softmax/CE are shift-invariant)
tDetachMax :: KnownNat n => Tape s p -> R s (V n) -> ST s (R s (V n))
tDetachMax tp (R x rx) = let !m = vmaxElem x in node tp (vmapT (subtract m) x) (\dy -> addTo rx dy)

tEmbedRow :: (KnownNat r, KnownNat c) => Tape s p -> Int -> R s (M r c) -> ST s (R s (V c))
tEmbedRow tp i (R w rw) = node tp (mrow w i) (\dv -> addTo rw (mScatterRow i dv))

-- ── driver ────────────────────────────────────────────────────────────────────

-- Run a forward computation that returns the scalar loss node, then backprop and
-- read off the parameter gradient.
tGradLoss :: forall p. Additive p
          => (forall s. Tape s p -> ST s (R s Double)) -> (p, Double)
tGradLoss build = runST $ do
  acts <- newSTRef []
  gref <- newSTRef zeroA
  let tp = Tape acts gref
  R l ladj <- build tp
  writeSTRef ladj 1.0                 -- seed: d loss / d loss = 1
  as <- readSTRef acts                -- newest-first == reverse topological
  sequence_ as
  g <- readSTRef gref
  pure (g, l)

-- Forward-only evaluation (no backprop), for periodic accuracy checks.
tEval :: forall p x. Additive p => (forall s. Tape s p -> ST s (R s x)) -> x
tEval build = runST $ do
  acts <- newSTRef []
  gref <- newSTRef zeroA
  R v _ <- build (Tape acts gref)
  pure v
