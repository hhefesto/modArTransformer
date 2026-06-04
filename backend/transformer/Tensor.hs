{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns #-}

-- Shape-indexed tensors backed by hmatrix (BLAS).  This is *one interpretation*
-- of the linear-map category: dense vectors/matrices over Double.  The AD/tape
-- layers are agnostic to this representation; primitive tensor operations here
-- are the only place that touches hmatrix.
module Tensor
  ( V(..)
  , M(..)
  , Additive(..)
  , Scale(..)
  , vzero
  , mzero
  , vfromList
  , mfromRows
  , vtoList
  , mtoRows
  , vdim
  , matvec
  , vouter
  , mtr
  , vsumElems
  , vmapT
  , vzipT
  , vscaleT
  , vaddT
  , dotT
  , vindex
  , vmaxElem
  , mrow
  , mScatterRow
  ) where

import GHC.TypeNats (KnownNat, Nat, natVal)
import Data.Proxy (Proxy(..))
import Control.DeepSeq (NFData(..))
import qualified Numeric.LinearAlgebra as LA
import qualified Data.Vector.Storable as VS
import Numeric.LinearAlgebra (Vector, Matrix, R)

-- Phantom-shaped vector / matrix over hmatrix.
newtype V (n :: Nat)            = V { unV :: Vector R }
newtype M (m :: Nat) (n :: Nat) = M { unM :: Matrix R }

instance Show (V n) where show (V v) = "V " <> show (LA.toList v)
instance Show (M m n) where show (M m) = "M " <> show (LA.toLists m)

instance NFData (V n) where rnf (V v) = rnf v
instance NFData (M m n) where rnf (M m) = rnf m

natI :: forall n. KnownNat n => Int
natI = fromIntegral (natVal (Proxy @n))

-- ─── Additive (zero + ⊕) — the cotangent monoid for reverse AD ────────────────

class Additive a where
  zeroA :: a
  addA  :: a -> a -> a

instance Additive Double where
  zeroA = 0
  addA  = (+)

instance Additive () where
  zeroA = ()
  addA _ _ = ()

instance (Additive a, Additive b) => Additive (a, b) where
  zeroA = (zeroA, zeroA)
  addA (a, b) (c, d) = let !x = addA a c; !y = addA b d in (x, y)

instance KnownNat n => Additive (V n) where
  zeroA = vzero
  addA (V a) (V b) = V (a + b)

instance (KnownNat m, KnownNat n) => Additive (M m n) where
  zeroA = mzero
  addA (M a) (M b) = M (a + b)

-- ─── Scale (scalar multiplication) ────────────────────────────────────────────

class Scale a where
  scaleA :: Double -> a -> a

instance Scale Double where scaleA s x = s * x
instance Scale () where scaleA _ _ = ()
instance (Scale a, Scale b) => Scale (a, b) where
  scaleA s (a, b) = (scaleA s a, scaleA s b)
instance KnownNat n => Scale (V n) where scaleA s (V v) = V (LA.scale s v)
instance (KnownNat m, KnownNat n) => Scale (M m n) where scaleA s (M m) = M (LA.scale s m)

-- ─── constructors / accessors ─────────────────────────────────────────────────

vzero :: forall n. KnownNat n => V n
vzero = V (LA.konst 0 (natI @n))

mzero :: forall m n. (KnownNat m, KnownNat n) => M m n
mzero = M (LA.konst 0 (natI @m, natI @n))

vfromList :: forall n. KnownNat n => [Double] -> V n
vfromList xs
  | length xs == natI @n = V (LA.fromList xs)
  | otherwise = error ("vfromList: expected " <> show (natI @n) <> " got " <> show (length xs))

mfromRows :: forall m n. (KnownNat m, KnownNat n) => [[Double]] -> M m n
mfromRows rows
  | length rows == natI @m && all ((== natI @n) . length) rows = M (LA.fromLists rows)
  | otherwise = error ("mfromRows: expected " <> show (natI @m) <> "x" <> show (natI @n))

vtoList :: V n -> [Double]
vtoList (V v) = LA.toList v

mtoRows :: M m n -> [[Double]]
mtoRows (M m) = LA.toLists m

vdim :: V n -> Int
vdim (V v) = LA.size v

-- ─── core linear-algebra ops ──────────────────────────────────────────────────

matvec :: M m n -> V n -> V m
matvec (M a) (V v) = V (a LA.#> v)

vouter :: V m -> V n -> M m n
vouter (V a) (V b) = M (LA.outer a b)

mtr :: M m n -> M n m
mtr (M a) = M (LA.tr a)

vsumElems :: V n -> Double
vsumElems (V v) = VS.sum v

-- native unboxed map/zip (no boxed-list round trip)
vmapT :: (Double -> Double) -> V n -> V n
vmapT f (V v) = V (VS.map f v)

vzipT :: (Double -> Double -> Double) -> V n -> V n -> V n
vzipT f (V a) (V b) = V (VS.zipWith f a b)

vscaleT :: Double -> V n -> V n
vscaleT s (V v) = V (LA.scale s v)

vaddT :: V n -> V n -> V n
vaddT (V a) (V b) = V (a + b)

dotT :: V n -> V n -> Double
dotT (V a) (V b) = a LA.<.> b

vindex :: V n -> Int -> Double
vindex (V v) i = v VS.! i

vmaxElem :: V n -> Double
vmaxElem (V v) = VS.maximum v

-- extract row i of a matrix as a vector
mrow :: M m n -> Int -> V n
mrow (M m) i = V (LA.flatten (m LA.? [i]))

-- zero matrix with row i set to v (used by the embedding adjoint)
mScatterRow :: forall m n. (KnownNat m, KnownNat n) => Int -> V n -> M m n
mScatterRow i (V v) =
  let r = natI @m
      z = LA.konst 0 (natI @n) :: Vector R
  in M (LA.fromRows [ if j == i then v else z | j <- [0 .. r - 1] ])
