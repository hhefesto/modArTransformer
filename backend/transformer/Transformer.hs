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
  -- two-layer variant (two stacked blocks; both token positions flow through
  -- each block, so layer 2 attends over layer 1's outputs — like the paper's
  -- 2-layer model).
  , Block, Params2, ParamsC2
  , transformerLogitsVal2
  , transformerGradLoss2
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

-- ── two-layer transformer ──────────────────────────────────────────────────────
--
-- One block = (Attn, LN1, FFN, LN2); a two-layer model stacks two of them between
-- the embedding and the unembed.  Unlike the single-layer forward (which only
-- fully processes position 0), `blockT` processes BOTH token positions through the
-- block — necessary so the second layer's attention can read the first layer's
-- output at both positions.  Wk/Wv/Wo/FFN/LN are shared across the two positions
-- within a block (the usual transformer weight sharing); only position 0 of the
-- final layer feeds the readout.

type Block dM dF dK = (Attn dM dK, (LN dM, (FFN dM dF, LN dM)))

type Params2 v dM dF dK =
  ( M v dM                                  -- tokEmbed
  , ( M 2 dM                                -- posEmbed
  , ( Block dM dF dK                        -- layer 1
  , ( Block dM dF dK                        -- layer 2
  ,   Lin v dM ))))                         -- unembed

type P2 v dM dF dK = Params2 v dM dF dK

type ParamsC2 v dM dF dK =
  ( KnownNat v, KnownNat dM, KnownNat dF, KnownNat dK
  , Additive (M v dM), Additive (M 2 dM), Additive (V v)
  , Additive (M dK dM), Additive (V dK), Additive (M dM dK), Additive (V dM)
  , Additive (M dF dM), Additive (V dF), Additive (M dM dF)
  , Additive (Params2 v dM dF dK) )

-- one transformer block applied to both positions (x0,x1) -> (y0,y1).
blockT :: forall s p dM dF dK. (KnownNat dM, KnownNat dF, KnownNat dK)
       => Tape s p -> p
       -> Lens p (Attn dM dK) -> Lens p (LN dM) -> Lens p (FFN dM dF) -> Lens p (LN dM)
       -> Double -> R s (V dM) -> R s (V dM)
       -> ST s (R s (V dM), R s (V dM))
blockT tp p lAttn lLn1 lFfn lLn2 sc x0 x1 = do
  qW <- tInput tp (lAttn .< fstL .< fstL .< fstL) p
  qB <- tInput tp (lAttn .< fstL .< fstL .< sndL) p
  kW <- tInput tp (lAttn .< fstL .< sndL .< fstL) p
  kB <- tInput tp (lAttn .< fstL .< sndL .< sndL) p
  vW <- tInput tp (lAttn .< sndL .< fstL .< fstL) p
  vB <- tInput tp (lAttn .< sndL .< fstL .< sndL) p
  oW <- tInput tp (lAttn .< sndL .< sndL .< fstL) p
  oB <- tInput tp (lAttn .< sndL .< sndL .< sndL) p
  g1 <- tInput tp (lLn1 .< fstL) p
  b1 <- tInput tp (lLn1 .< sndL) p
  upW <- tInput tp (lFfn .< fstL .< fstL) p
  upB <- tInput tp (lFfn .< fstL .< sndL) p
  dnW <- tInput tp (lFfn .< sndL .< fstL) p
  dnB <- tInput tp (lFfn .< sndL .< sndL) p
  g2 <- tInput tp (lLn2 .< fstL) p
  b2 <- tInput tp (lLn2 .< sndL) p
  k0 <- affineT tp kW kB x0
  k1 <- affineT tp kW kB x1
  v0 <- affineT tp vW vB x0
  v1 <- affineT tp vW vB x1
  let finish q xres = do
        d0 <- tVdot tp q k0
        d1 <- tVdot tp q k1
        s0 <- tScaleC tp sc d0
        s1 <- tScaleC tp sc d1
        e0' <- tExp tp s0
        e1' <- tExp tp s1
        z   <- tAdd tp e0' e1'
        rz  <- tRecip tp z
        w0  <- tMul tp e0' rz
        w1  <- tMul tp e1' rz
        a0v <- tScaleV tp w0 v0
        a1v <- tScaleV tp w1 v1
        att <- tVadd tp a0v a1v
        ao  <- affineT tp oW oB att
        r1  <- tVadd tp xres ao
        n1  <- layerNormT tp g1 b1 r1
        hh  <- affineT tp upW upB n1
        hr  <- tReluV tp hh
        ff  <- affineT tp dnW dnB hr
        r2  <- tVadd tp n1 ff
        layerNormT tp g2 b2 r2
  q0 <- affineT tp qW qB x0
  q1 <- affineT tp qW qB x1
  y0 <- finish q0 x0
  y1 <- finish q1 x1
  pure (y0, y1)

forwardT2 :: forall s v dM dF dK. ParamsC2 v dM dF dK
          => Tape s (P2 v dM dF dK) -> Int -> Int -> P2 v dM dF dK -> ST s (R s (V v))
forwardT2 tp tokA tokB p = do
  let lTok = fstL
      lPos = sndL .< fstL
      lB1  = sndL .< sndL .< fstL
      lB2  = sndL .< sndL .< sndL .< fstL
      lUn  = sndL .< sndL .< sndL .< sndL
  rTok <- tInput tp lTok p
  rPos <- tInput tp lPos p
  let embedAt tk ps = do
        et <- tEmbedRow tp tk rTok
        ep <- tEmbedRow tp ps rPos
        tVadd tp et ep
  e0 <- embedAt tokA 0
  e1 <- embedAt tokB 1
  let sc = 1.0 / sqrt (fromIntegral (natVal (Proxy @dK)))
  (h0, h1) <- blockT tp p (lB1 .< fstL) (lB1 .< sndL .< fstL)
                          (lB1 .< sndL .< sndL .< fstL) (lB1 .< sndL .< sndL .< sndL) sc e0 e1
  (o0, _o1) <- blockT tp p (lB2 .< fstL) (lB2 .< sndL .< fstL)
                          (lB2 .< sndL .< sndL .< fstL) (lB2 .< sndL .< sndL .< sndL) sc h0 h1
  uW <- tInput tp (lUn .< fstL) p
  uB <- tInput tp (lUn .< sndL) p
  affineT tp uW uB o0

transformerLogitsVal2 :: forall v dM dF dK. ParamsC2 v dM dF dK
                      => Int -> Int -> P2 v dM dF dK -> V v
transformerLogitsVal2 a b p = tEval (\tp -> forwardT2 tp a b p)

transformerGradLoss2 :: forall v dM dF dK. ParamsC2 v dM dF dK
                     => Int -> Int -> Int -> P2 v dM dF dK -> (P2 v dM dF dK, Double)
transformerGradLoss2 a b t p = tGradLoss build
  where
    build :: forall s. Tape s (P2 v dM dF dK) -> ST s (R s Double)
    build tp = do
      logits <- forwardT2 tp a b p
      l2  <- tDetachMax tp logits
      e   <- tExpV tp l2
      s   <- tVsum tp e
      lse <- tLog tp s
      sel <- tSelect @s @_ @v tp t l2
      tSub tp lse sel
