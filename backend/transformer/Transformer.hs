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
  -- pre-unembed residual ("meaning" representation, position 0 of the final
  -- block) — used by the enriched-semantics probe.
  , transformerReprVal2
  -- generalized sequence model (n positions, causal, two heads) — mirrors
  -- ModArTransformer/Layers/SeqTransformer.agda with the same parameter
  -- nesting, so Serialize stays aligned leaf-for-leaf with the Agda side.
  , HeadP, MH2, SeqBlock, ParamsSeq, ParamsCSeq
  , seqLogitsVal
  , seqGradLoss
  ) where

import Control.Monad (forM, foldM, zipWithM)
import Control.Monad.ST (ST)
import Control.DeepSeq (NFData)
import GHC.TypeNats (KnownNat, natVal, type (+))
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
  , Additive (Params v dM dF dK), NFData (Params v dM dF dK) )

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
  let sm = max (primalR score0) (primalR score1)
  score0' <- tAddC tp (negate sm) score0
  score1' <- tAddC tp (negate sm) score1
  es0 <- tExp tp score0'
  es1 <- tExp tp score1'
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
  , Additive (Params2 v dM dF dK), NFData (Params2 v dM dF dK) )

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
        let sm = max (primalR s0) (primalR s1)
        s0' <- tAddC tp (negate sm) s0
        s1' <- tAddC tp (negate sm) s1
        e0' <- tExp tp s0'
        e1' <- tExp tp s1'
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

-- forward up to (but not including) the unembed: the position-0 residual of the
-- final block.  This is the context's "meaning" vector — what the enriched
-- copresheaf (the softmax) is read off from.  forwardT2 = unembed ∘ this, so the
-- logits/grad path is unchanged.
forwardT2Repr :: forall s v dM dF dK. ParamsC2 v dM dF dK
              => Tape s (P2 v dM dF dK) -> Int -> Int -> P2 v dM dF dK -> ST s (R s (V dM))
forwardT2Repr tp tokA tokB p = do
  let lTok = fstL
      lPos = sndL .< fstL
      lB1  = sndL .< sndL .< fstL
      lB2  = sndL .< sndL .< sndL .< fstL
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
  pure o0

forwardT2 :: forall s v dM dF dK. ParamsC2 v dM dF dK
          => Tape s (P2 v dM dF dK) -> Int -> Int -> P2 v dM dF dK -> ST s (R s (V v))
forwardT2 tp tokA tokB p = do
  o0 <- forwardT2Repr tp tokA tokB p
  let lUn = sndL .< sndL .< sndL .< sndL
  uW <- tInput tp (lUn .< fstL) p
  uB <- tInput tp (lUn .< sndL) p
  affineT tp uW uB o0

transformerLogitsVal2 :: forall v dM dF dK. ParamsC2 v dM dF dK
                      => Int -> Int -> P2 v dM dF dK -> V v
transformerLogitsVal2 a b p = tEval (\tp -> forwardT2 tp a b p)

-- the context's pre-unembed "meaning" vector (position-0 residual of the final block)
transformerReprVal2 :: forall v dM dF dK. ParamsC2 v dM dF dK
                    => Int -> Int -> P2 v dM dF dK -> V dM
transformerReprVal2 a b p = tEval (\tp -> forwardT2Repr tp a b p)

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

-- ── generalized sequence model ────────────────────────────────────────────────
--
-- n positions, CAUSAL attention (position i attends 0..i), TWO heads (concat +
-- joint Wo), one block, next-token logits at every position, loss = mean
-- cross-entropy over the n−1 prediction positions.  This is the Phase-0/1
-- model of the production roadmap (README §10).  The parameter tree nests
-- exactly like SeqTransformer.agda's SeqTransformerParams, so the shared flat
-- [Float] layout (Serialize ↔ Cat/Serialize.agda) is identical and the
-- conformance oracle can feed both sides the same parameters.

type HeadP dM dK    = (Lin dK dM, (Lin dK dM, Lin dK dM))      -- Wq, (Wk, Wv)
type MH2 dM dK      = (HeadP dM dK, (HeadP dM dK, Lin dM (dK + dK)))
type SeqBlock dM dF dK = (MH2 dM dK, (LN dM, (FFN dM dF, LN dM)))

type ParamsSeq v n dM dF dK =
  ( M v dM                                  -- tokEmbed
  , ( M n dM                                -- posEmbed
  , ( SeqBlock dM dF dK
  ,   Lin v dM )))                          -- unembed

type PSeq v n dM dF dK = ParamsSeq v n dM dF dK

type ParamsCSeq v n dM dF dK =
  ( KnownNat v, KnownNat n, KnownNat dM, KnownNat dF, KnownNat dK
  , KnownNat (dK + dK)
  , Additive (M v dM), Additive (M n dM), Additive (V v)
  , Additive (M dK dM), Additive (V dK), Additive (M dM (dK + dK)), Additive (V dM)
  , Additive (V (dK + dK))
  , Additive (M dF dM), Additive (V dF), Additive (M dM dF)
  , Additive (ParamsSeq v n dM dF dK), NFData (ParamsSeq v n dM dF dK) )

fold1M :: Monad m => (a -> a -> m a) -> [a] -> m a
fold1M f (x : xs) = foldM f x xs
fold1M _ []       = error "fold1M: empty"

-- forward over the whole sequence: next-token logits at every position.
forwardSeqT :: forall s v n dM dF dK. ParamsCSeq v n dM dF dK
            => Tape s (PSeq v n dM dF dK) -> [Int] -> PSeq v n dM dF dK
            -> ST s [R s (V v)]
forwardSeqT tp toks p = do
  let lTok = fstL
      lPos = sndL .< fstL
      lBlk = sndL .< sndL .< fstL
      lUn  = sndL .< sndL .< sndL
      lAtt = lBlk .< fstL
      lLn1 = lBlk .< sndL .< fstL
      lFfn = lBlk .< sndL .< sndL .< fstL
      lLn2 = lBlk .< sndL .< sndL .< sndL
      lH1  = lAtt .< fstL
      lH2  = lAtt .< sndL .< fstL
      lWo  = lAtt .< sndL .< sndL
      sc   = 1.0 / sqrt (fromIntegral (natVal (Proxy @dK)))
  rTok <- tInput tp lTok p
  rPos <- tInput tp lPos p
  embeds <- mapM (\(i, tk) -> do
                    et <- tEmbedRow tp tk rTok
                    ep <- tEmbedRow tp i rPos
                    tVadd tp et ep)
                 (zip [0 ..] toks)
  -- one head: causal per-position outputs (mirrors SeqTransformer.headOuts;
  -- the mask is realized by attending only the prefix — masked weights are 0
  -- on the Agda side, absent here: same weights, same parameter cotangents).
  let headOuts lH = do
        let lWq = lH .< fstL
            lWk = lH .< sndL .< fstL
            lWv = lH .< sndL .< sndL
        qW <- tInput tp (lWq .< fstL) p ; qB <- tInput tp (lWq .< sndL) p
        kW <- tInput tp (lWk .< fstL) p ; kB <- tInput tp (lWk .< sndL) p
        vW <- tInput tp (lWv .< fstL) p ; vB <- tInput tp (lWv .< sndL) p
        qs <- mapM (affineT tp qW qB) embeds
        ks <- mapM (affineT tp kW kB) embeds
        vs <- mapM (affineT tp vW vB) embeds
        forM (zip [0 ..] qs) $ \(i, qi) -> do
          let ksA = take (i + 1) ks
              vsA = take (i + 1) vs
          ds  <- mapM (tVdot tp qi) ksA
          ss  <- mapM (tScaleC tp sc) ds
          let sm = maximum (map primalR ss)
          ss' <- mapM (tAddC tp (negate sm)) ss
          es  <- mapM (tExp tp) ss'
          z   <- fold1M (tAdd tp) es
          rz  <- tRecip tp z
          ws  <- mapM (\e -> tMul tp e rz) es
          avs <- zipWithM (tScaleV tp) ws vsA
          fold1M (tVadd tp) avs
  h1 <- headOuts lH1
  h2 <- headOuts lH2
  oW  <- tInput tp (lWo .< fstL) p  ; oB  <- tInput tp (lWo .< sndL) p
  g1  <- tInput tp (lLn1 .< fstL) p ; b1  <- tInput tp (lLn1 .< sndL) p
  upW <- tInput tp (lFfn .< fstL .< fstL) p ; upB <- tInput tp (lFfn .< fstL .< sndL) p
  dnW <- tInput tp (lFfn .< sndL .< fstL) p ; dnB <- tInput tp (lFfn .< sndL .< sndL) p
  g2  <- tInput tp (lLn2 .< fstL) p ; b2  <- tInput tp (lLn2 .< sndL) p
  uW  <- tInput tp (lUn .< fstL) p  ; uB  <- tInput tp (lUn .< sndL) p
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

-- next-token logits at every position (forward only).
seqLogitsVal :: forall v n dM dF dK. ParamsCSeq v n dM dF dK
             => [Int] -> PSeq v n dM dF dK -> [V v]
seqLogitsVal toks p = tEvalMany (\tp -> forwardSeqT tp toks p)

-- mean next-token cross-entropy over the n−1 prediction positions, with its
-- gradient (position i predicts token i+1) — mirrors seqTransformerLoss.
seqGradLoss :: forall v n dM dF dK. ParamsCSeq v n dM dF dK
            => [Int] -> PSeq v n dM dF dK -> (PSeq v n dM dF dK, Double)
seqGradLoss toks p = tGradLoss build
  where
    build :: forall s. Tape s (PSeq v n dM dF dK) -> ST s (R s Double)
    build tp = do
      logits <- forwardSeqT tp toks p
      ls <- forM (zip (init logits) (tail toks)) $ \(lg, t) -> do
        l2  <- tDetachMax tp lg
        e   <- tExpV tp l2
        s   <- tVsum tp e
        lse <- tLog tp s
        sel <- tSelect @s @_ @v tp t l2
        tSub tp lse sel
      tot <- fold1M (tAdd tp) ls
      tScaleC tp (1 / fromIntegral (length ls)) tot
