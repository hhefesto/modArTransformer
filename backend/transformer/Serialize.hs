{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}

-- Flatten/unflatten over the parameter product tree, in a fixed leaf order, for
-- finite-difference gradient checking and checkpoint I/O.
module Serialize
  ( Serialize(..)
  , toFloats
  ) where

import GHC.TypeNats (KnownNat)
import Tensor

class Serialize a where
  -- consume a prefix of the list, returning the value and the remainder
  fromFloats :: [Double] -> (a, [Double])
  -- append this value's floats onto the accumulator
  putFloats  :: a -> [Double] -> [Double]

toFloats :: Serialize a => a -> [Double]
toFloats a = putFloats a []

instance Serialize Double where
  fromFloats (x : xs) = (x, xs)
  fromFloats []       = error "fromFloats: ran out of input (Double)"
  putFloats x acc     = x : acc

instance KnownNat n => Serialize (V n) where
  fromFloats xs = let k = vdim (vzero @n)
                      (h, t) = splitAt k xs
                  in (vfromList @n h, t)
  putFloats v acc = vtoList v ++ acc

instance (KnownNat m, KnownNat n) => Serialize (M m n) where
  fromFloats xs = let r = length (mtoRows (mzero @m @n))
                      c = vdim (vzero @n)
                      (h, t) = splitAt (r * c) xs
                  in (mfromRows @m @n (chunk c h), t)
    where chunk _ [] = []
          chunk k ys = let (a, b) = splitAt k ys in a : chunk k b
  putFloats m acc = concat (mtoRows m) ++ acc

instance (Serialize a, Serialize b) => Serialize (a, b) where
  fromFloats xs = let (a, xs')  = fromFloats xs
                      (b, xs'') = fromFloats xs'
                  in ((a, b), xs'')
  putFloats (a, b) acc = putFloats a (putFloats b acc)
