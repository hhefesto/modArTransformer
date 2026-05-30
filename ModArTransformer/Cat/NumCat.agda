-- Numeric primitives as morphisms in D (Dual AddFun): the calculus derivatives
-- of +, ×, −, negate, recip, exp, log, and relu, each defined *once* with its
-- local derivative as a reversed linear map.  This is the analogue of Elliott's
-- `NumCat` instance for the derivative category.  Neural layers are built by
-- composing these; their gradients then follow from the chain rule in D —
-- there is no per-layer backward code anywhere.
{-# OPTIONS --without-K #-}
module ModArTransformer.Cat.NumCat where

open import Data.Product using (_,_; proj₁; proj₂)
open import Data.Bool using (if_then_else_)
open import Agda.Builtin.Float
  using ( Float; primFloatPlus; primFloatMinus; primFloatTimes; primFloatDiv
        ; primFloatNegate; primFloatExp; primFloatLog; primFloatLess )

open import ModArTransformer.Cat.Objects
open import ModArTransformer.Cat.Additive
open import ModArTransformer.Cat.AddFun
open import ModArTransformer.Cat.Dual
open import ModArTransformer.Cat.D

-- addition: linear; pullback duplicates the upstream cotangent.
addD : D (Float × Float) Float
addD = linearD (λ p → primFloatPlus (proj₁ p) (proj₂ p))
               (mkDual (mkAddFun (λ dz → dz , dz)))

-- subtraction: linear; pullback (dz ↦ (dz, −dz)).
subD : D (Float × Float) Float
subD = linearD (λ p → primFloatMinus (proj₁ p) (proj₂ p))
               (mkDual (mkAddFun (λ dz → dz , primFloatNegate dz)))

-- multiplication: NOT linear; derivative depends on the inputs (product rule):
--   d(xy) = y·dx + x·dy ,  transpose: dz ↦ (y·dz, x·dz).
mulD : D (Float × Float) Float
mulD = mkD (λ p →
  let x = proj₁ p ; y = proj₂ p in
  ( primFloatTimes x y
  , mkDual (mkAddFun (λ dz → primFloatTimes y dz , primFloatTimes x dz)) ))

-- negation: linear.
negD : D Float Float
negD = linearD primFloatNegate (mkDual (mkAddFun primFloatNegate))

-- scaling by a constant: linear.
scaleByD : Float → D Float Float
scaleByD c = linearD (primFloatTimes c) (mkDual (mkAddFun (primFloatTimes c)))

-- reciprocal: d(1/x) = −1/x² · dx.
recipD : D Float Float
recipD = mkD (λ x →
  let r = primFloatDiv 1.0 x
      d = primFloatNegate (primFloatTimes r r) in
  ( r , mkDual (mkAddFun (primFloatTimes d)) ))

-- exponential: d(eˣ) = eˣ · dx.
expD : D Float Float
expD = mkD (λ x →
  let e = primFloatExp x in
  ( e , mkDual (mkAddFun (primFloatTimes e)) ))

-- natural log: d(log x) = dx / x.
logD : D Float Float
logD = mkD (λ x →
  ( primFloatLog x
  , mkDual (mkAddFun (λ dz → primFloatDiv dz x)) ))

-- ReLU: derivative is the 0/1 mask of the input sign.
reluD : D Float Float
reluD = mkD (λ x →
  let neg  = primFloatLess x 0.0
      y    = if neg then 0.0 else x
      mask = if neg then 0.0 else 1.0 in
  ( y , mkDual (mkAddFun (primFloatTimes mask)) ))
