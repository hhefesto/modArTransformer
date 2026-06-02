{-# LANGUAGE DataKinds #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Tensor
  ( Vec(..)
  , Mat(..)
  , vec
  , mat
  , vzipWith
  , vmap
  , vadd
  , vscale
  , vdot
  , matvec
  , transpose
  , outer
  , vecToList
  , matToRows
  ) where

import GHC.TypeNats (KnownNat, Nat, natVal)
import Prelude hiding (transpose)
import Data.Proxy (Proxy(..))

newtype Vec (n :: Nat) = Vec { vecToList :: [Double] }
  deriving (Eq, Show)

newtype Mat (m :: Nat) (n :: Nat) = Mat { matToRows :: [[Double]] }
  deriving (Eq, Show)

expected :: forall n. KnownNat n => Int
expected = fromIntegral (natVal (Proxy @n))

vec :: forall n. KnownNat n => [Double] -> Vec n
vec xs
  | length xs == expected @n = Vec xs
  | otherwise = error $ "Vec length mismatch: expected " <> show (expected @n) <> ", got " <> show (length xs)

mat :: forall m n. (KnownNat m, KnownNat n) => [[Double]] -> Mat m n
mat rows
  | length rows == expected @m && all ((== expected @n) . length) rows = Mat rows
  | otherwise = error $ "Mat shape mismatch: expected " <> show (expected @m) <> "x" <> show (expected @n)

vzipWith :: (Double -> Double -> Double) -> Vec n -> Vec n -> Vec n
vzipWith f (Vec xs) (Vec ys) = Vec (zipWith f xs ys)

vmap :: (Double -> Double) -> Vec n -> Vec n
vmap f (Vec xs) = Vec (map f xs)

vadd :: Vec n -> Vec n -> Vec n
vadd = vzipWith (+)

vscale :: Double -> Vec n -> Vec n
vscale s = vmap (s *)

vdot :: Vec n -> Vec n -> Double
vdot (Vec xs) (Vec ys) = sum (zipWith (*) xs ys)

matvec :: Mat m n -> Vec n -> Vec m
matvec (Mat rows) v = Vec (map (vdot v . Vec) rows)

transpose :: forall m n. Mat m n -> Mat n m
transpose (Mat rows) = Mat (go rows)
  where
    go [] = []
    go ([] : _) = []
    go rs = map head rs : go (map tail rs)

outer :: Vec m -> Vec n -> Mat m n
outer (Vec xs) (Vec ys) = Mat [[x * y | y <- ys] | x <- xs]
