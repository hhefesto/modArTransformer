{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE RankNTypes #-}

-- The transformer built on a Wengert tape (Tape.hs), transliterated from
-- ModArTransformer/Layers/Transformer.agda.  Every intermediate is a monadically
-- bound node computed once; multiple uses register multiple cotangent
-- contributions into the same node cell (correct accumulation, no recomputation).
-- Parameter access is via field-local lenses.  The backward pass is the
-- composition of the per-primitive adjoints — no hand-written backward for any
-- composite.  seqLen = 2, single head; only position 0 feeds the readout.
module Transformer
  ( Lin, LN, Attn, FFN, Params, ParamsC
  , transformerLogitsVal
  , transformerGradLoss
  ) where

import Control.Monad.ST (ST)
import GHC.TypeNats (KnownNat, natVal)
import Data.Proxy (Proxy(..))
import AD (Lens, fstL, sndL, (.<))
import Tape
import Tensor

-- ── parameter product tree ──────────────────────────────────────────────────

type Lin o i      = (M o i, V o)                                   -- weight, bias
type LN n         = (V n, V n)                                     -- gamma, beta
type Attn dM dK   = ((Lin dK dM, Lin dK dM), (Lin dK dM, Lin dM dK)) -- ((Wq,Wk),(Wv,Wo))
type FFN dM dF    = (Lin dF dM, Lin dM dF)                         -- up, down

type Params v dM dF dK =
  ( M v dM                                  -- tokEmbed
  , ( M 2 dM                                -- posEmbed
  , ( Attn dM dK
  , ( LN dM                                 -- LN1
  , ( FFN dM dF
  , ( LN dM                                 -- LN2
  ,   Lin v dM ))))))                       -- unembed

type P v dM dF dK = Params v dM dF dK

type ParamsC v dM dF dK =
  ( KnownNat v, KnownNat dM, KnownNat dF, KnownNat dK
  , Additive (M v dM), Additive (M 2 dM), Additive (V v)
  , Additive (M dK dM), Additive (V dK), Additive (M dM dK), Additive (V dM)
  , Additive (M dF dM), Additive (V dF), Additive (M dM dF)
  , Additive (Params v dM dF dK) )

-- ── layer helpers (on the tape) ───────────────────────────────────────────────

affineT :: (KnownNat o, KnownNat n)
        => Tape s p -> R s (M o n) -> R s (V o) -> R s (V n) -> ST s (R s (V o))
affineT tp rw rb rx = do
  wx <- tMatvec tp rw rx
  tVadd tp wx rb

layerNormT :: KnownNat n
           => Tape s p -> R s (V n) -> R s (V n) -> R s (V n) -> ST s (R s (V n))
layerNormT tp rg rb rx = do
  xc   <- tCenter tp rx
  sq   <- tSquareV tp xc
  var  <- tMeanV tp sq
  var' <- tAddC tp 1.0e-5 var
  inv  <- tRsqrt tp var'
  norm <- tScaleV tp inv xc
  gn   <- tHadamard tp rg norm
  tVadd tp gn rb

-- ── full forward ──────────────────────────────────────────────────────────────

forwardT :: forall s v dM dF dK. ParamsC v dM dF dK
         => Tape s (P v dM dF dK) -> Int -> Int -> P v dM dF dK -> ST s (R s (V v))
forwardT tp tokA tokB p = do
  let lTok  = fstL
      lPos  = sndL .< fstL
      lAttn = sndL .< sndL .< fstL
      lLn1  = sndL .< sndL .< sndL .< fstL
      lFfn  = sndL .< sndL .< sndL .< sndL .< fstL
      lLn2  = sndL .< sndL .< sndL .< sndL .< sndL .< fstL
      lUn   = sndL .< sndL .< sndL .< sndL .< sndL .< sndL
  rTok <- tInput tp lTok p
  rPos <- tInput tp lPos p
  let embedAt tk ps = do
        et <- tEmbedRow tp tk rTok
        ep <- tEmbedRow tp ps rPos
        tVadd tp et ep
  e0 <- embedAt tokA 0
  e1 <- embedAt tokB 1

  qW <- tInput tp (lAttn .< fstL .< fstL .< fstL) p
  qB <- tInput tp (lAttn .< fstL .< fstL .< sndL) p
  kW <- tInput tp (lAttn .< fstL .< sndL .< fstL) p
  kB <- tInput tp (lAttn .< fstL .< sndL .< sndL) p
  vW <- tInput tp (lAttn .< sndL .< fstL .< fstL) p
  vB <- tInput tp (lAttn .< sndL .< fstL .< sndL) p
  oW <- tInput tp (lAttn .< sndL .< sndL .< fstL) p
  oB <- tInput tp (lAttn .< sndL .< sndL .< sndL) p
  q0 <- affineT tp qW qB e0
  k0 <- affineT tp kW kB e0
  k1 <- affineT tp kW kB e1
  v0 <- affineT tp vW vB e0
  v1 <- affineT tp vW vB e1
  let sc = 1.0 / sqrt (fromIntegral (natVal (Proxy @dK)))
  d0 <- tVdot tp q0 k0
  d1 <- tVdot tp q0 k1
  score0 <- tScaleC tp sc d0
  score1 <- tScaleC tp sc d1
  es0 <- tExp tp score0
  es1 <- tExp tp score1
  z   <- tAdd tp es0 es1
  rz  <- tRecip tp z
  w0  <- tMul tp es0 rz
  w1  <- tMul tp es1 rz
  a0v <- tScaleV tp w0 v0
  a1v <- tScaleV tp w1 v1
  attended <- tVadd tp a0v a1v
  a0  <- affineT tp oW oB attended

  r10 <- tVadd tp e0 a0
  g1 <- tInput tp (lLn1 .< fstL) p
  b1 <- tInput tp (lLn1 .< sndL) p
  n10 <- layerNormT tp g1 b1 r10

  upW <- tInput tp (lFfn .< fstL .< fstL) p
  upB <- tInput tp (lFfn .< fstL .< sndL) p
  dnW <- tInput tp (lFfn .< sndL .< fstL) p
  dnB <- tInput tp (lFfn .< sndL .< sndL) p
  h0 <- affineT tp upW upB n10
  hr <- tReluV tp h0
  f0 <- affineT tp dnW dnB hr

  r20 <- tVadd tp n10 f0
  g2 <- tInput tp (lLn2 .< fstL) p
  b2 <- tInput tp (lLn2 .< sndL) p
  o0 <- layerNormT tp g2 b2 r20

  uW <- tInput tp (lUn .< fstL) p
  uB <- tInput tp (lUn .< sndL) p
  affineT tp uW uB o0

-- ── exports ─────────────────────────────────────────────────────────────────

transformerLogitsVal :: forall v dM dF dK. ParamsC v dM dF dK
                     => Int -> Int -> P v dM dF dK -> V v
transformerLogitsVal a b p = tEval (\tp -> forwardT tp a b p)

transformerGradLoss :: forall v dM dF dK. ParamsC v dM dF dK
                    => Int -> Int -> Int -> P v dM dF dK -> (P v dM dF dK, Double)
transformerGradLoss a b t p = tGradLoss build
  where
    build :: forall s. Tape s (P v dM dF dK) -> ST s (R s Double)
    build tp = do
      logits <- forwardT tp a b p
      l2  <- tDetachMax tp logits
      e   <- tExpV tp l2
      s   <- tVsum tp e
      lse <- tLog tp s
      sel <- tSelect @s @_ @v tp t l2
      tSub tp lse sel
