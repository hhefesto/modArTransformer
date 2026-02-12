{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

{- |
Module      : ModularTransformer
Description : A 1-layer transformer trained on (a + b) mod p.

Applied changes (from the comment you posted, but keeping what compiles cleanly in hmatrix):
  • AdamW (decoupled weight decay; no WD on bias / LayerNorm params).
  • Learning rate schedule: linear warmup + cosine decay (per batch step).
  • ST-based in-place accumulation for token/pos embedding gradients (avoids giant scatter lists).
  • Checkpoint save/load:
      - Text format (fast enough for this small model).
      - Stores params + Adam state + timestep so schedule resumes correctly.

Dependencies:
  cabal/stack packages: hmatrix, vector, random
-}

module Main where

import qualified Numeric.LinearAlgebra as LA
import           Numeric.LinearAlgebra (Matrix, Vector, R)

import qualified Data.Vector as V
import qualified Data.Vector.Mutable as MV

import qualified Data.Vector.Storable as VS
import qualified Data.Vector.Storable.Mutable as VSM

import           Control.Monad (forM_, when)
import           Control.Monad.ST (ST, runST)
import           Data.List (foldl')
import           Data.STRef (newSTRef, readSTRef, writeSTRef, modifySTRef')
import           System.Directory (doesFileExist)
import           System.IO (hFlush, stdout)
import           System.Random (StdGen, mkStdGen, randomR, random)

import           Text.Printf (printf)

--------------------------------------------------------------------------------
-- Types / helpers
--------------------------------------------------------------------------------

type Vec = Vector R
type Mat = Matrix R

zerosV :: Int -> Vec
zerosV n = LA.konst 0 n

zerosM :: Int -> Int -> Mat
zerosM r c = LA.konst 0 (r, c)

vlen :: Vec -> Int
vlen = LA.size

relu :: Vec -> Vec
relu = LA.cmap (max 0)

relu' :: Vec -> Vec
relu' = LA.cmap (\x -> if x > 0 then 1 else 0)

softmax :: Vec -> Vec
softmax xs =
  let !mx   = LA.maxElement xs
      !exps = LA.cmap (\x -> exp (x - mx)) xs
      !s    = max 1e-12 (LA.sumElements exps)
  in LA.scale (1 / s) exps

-- Given y = softmax(x), and upstream dy, compute dx.
softmaxBackward :: Vec -> Vec -> Vec
softmaxBackward y dy =
  let n      = vlen y
      !dotYD = y LA.<.> dy
  in y * (dy - LA.konst dotYD n)

--------------------------------------------------------------------------------
-- Linear layer
--------------------------------------------------------------------------------

data LinearParams = LinearParams
  { linW :: !Mat   -- out × in
  , linB :: !Vec   -- out
  }

zeroLinearGrads :: Int -> Int -> LinearParams
zeroLinearGrads outDim inDim = LinearParams (zerosM outDim inDim) (zerosV outDim)

addLinear :: LinearParams -> LinearParams -> LinearParams
addLinear (LinearParams w1 b1) (LinearParams w2 b2) =
  LinearParams (w1 + w2) (b1 + b2)

scaleLinear :: R -> LinearParams -> LinearParams
scaleLinear s (LinearParams w b) = LinearParams (LA.scale s w) (LA.scale s b)

linearForward :: LinearParams -> Vec -> Vec
linearForward LinearParams{..} x = (linW LA.#> x) + linB

-- Returns (grads, dx)
linearBackward :: LinearParams -> Vec -> Vec -> (LinearParams, Vec)
linearBackward LinearParams{..} x dY =
  let dW = LA.outer dY x
      dB = dY
      dX = (LA.tr linW) LA.#> dY
  in (LinearParams dW dB, dX)

--------------------------------------------------------------------------------
-- LayerNorm (per-token)
--------------------------------------------------------------------------------

data LayerNormParams = LayerNormParams
  { lnGamma :: !Vec  -- d_model
  , lnBeta  :: !Vec  -- d_model
  }

data LayerNormCache = LayerNormCache
  { lnXHat   :: !Vec
  , lnInvStd :: !R
  }

initLayerNorm :: Int -> LayerNormParams
initLayerNorm d = LayerNormParams
  { lnGamma = LA.konst 1 d
  , lnBeta  = LA.konst 0 d
  }

layerNormForward :: LayerNormParams -> Vec -> (Vec, LayerNormCache)
layerNormForward LayerNormParams{..} x =
  let !d     = fromIntegral (vlen x)
      !mu    = LA.sumElements x / d
      !diff  = x - LA.konst mu (vlen x)
      !var   = LA.sumElements (diff * diff) / d
      !eps   = 1e-5
      !invS  = 1 / sqrt (var + eps)
      !xHat  = LA.scale invS diff
      !out   = lnGamma * xHat + lnBeta
  in (out, LayerNormCache xHat invS)

-- Correct LN backward (per token).
layerNormBackward :: LayerNormParams -> LayerNormCache -> Vec -> (Vec, Vec, Vec)
layerNormBackward LayerNormParams{..} LayerNormCache{..} dOut =
  let !n      = vlen dOut
      !d      = fromIntegral n

      !dBeta  = dOut
      !dGamma = dOut * lnXHat

      !dXHat  = dOut * lnGamma

      !sumDXHat     = LA.sumElements dXHat
      !sumDXHatXHat = LA.sumElements (dXHat * lnXHat)

      !term1 = LA.scale d dXHat
      !term2 = LA.konst sumDXHat n
      !term3 = lnXHat * LA.konst sumDXHatXHat n

      !dx = LA.scale (lnInvStd / d) (term1 - term2 - term3)
  in (dx, dGamma, dBeta)

--------------------------------------------------------------------------------
-- FFN
--------------------------------------------------------------------------------

data FFNParams = FFNParams
  { ffnLinear1 :: !LinearParams  -- d_model -> d_ff
  , ffnLinear2 :: !LinearParams  -- d_ff    -> d_model
  }

data FFNCache = FFNCache
  { ffnHidden    :: !Vec  -- pre-ReLU
  , ffnActivated :: !Vec  -- post-ReLU
  }

ffnForward :: FFNParams -> Vec -> (Vec, FFNCache)
ffnForward FFNParams{..} x =
  let !h   = linearForward ffnLinear1 x
      !act = relu h
      !out = linearForward ffnLinear2 act
  in (out, FFNCache h act)

ffnBackward :: FFNParams -> Vec -> FFNCache -> Vec -> (LinearParams, LinearParams, Vec)
ffnBackward FFNParams{..} x FFNCache{..} dOut =
  let (g2, dAct) = linearBackward ffnLinear2 ffnActivated dOut
      !dH        = dAct * relu' ffnHidden
      (g1, dX)   = linearBackward ffnLinear1 x dH
  in (g1, g2, dX)

--------------------------------------------------------------------------------
-- Self-attention (single head)
--------------------------------------------------------------------------------

data AttentionParams = AttentionParams
  { attnWq :: !LinearParams
  , attnWk :: !LinearParams
  , attnWv :: !LinearParams
  , attnWo :: !LinearParams
  }

data AttentionCache = AttentionCache
  { acQ        :: ![Vec]
  , acK        :: ![Vec]
  , acV        :: ![Vec]
  , acWeights  :: ![Vec]
  , acAttended :: ![Vec]
  }

attentionForward :: AttentionParams -> [Vec] -> ([Vec], AttentionCache)
attentionForward AttentionParams{..} xs =
  let !qs = map (linearForward attnWq) xs
      !ks = map (linearForward attnWk) xs
      !vs = map (linearForward attnWv) xs
      !n  = length xs
      !dk = fromIntegral (vlen (head qs))
      !sf = 1 / sqrt dk

      !weights =
        [ softmax $ LA.scale sf $ LA.fromList [ qi LA.<.> kj | kj <- ks ]
        | qi <- qs
        ]

      !attended =
        [ foldl' (+) (zerosV (vlen (head vs)))
            [ LA.scale (w_i `LA.atIndex` j) (vs !! j)
            | j <- [0 .. n - 1]
            ]
        | w_i <- weights
        ]

      !outs = map (linearForward attnWo) attended
      !cache = AttentionCache qs ks vs weights attended
  in (outs, cache)

data AttentionGrads = AttentionGrads
  { gWqParams :: !LinearParams
  , gWkParams :: !LinearParams
  , gWvParams :: !LinearParams
  , gWoParams :: !LinearParams
  }

attentionBackward
  :: AttentionParams
  -> [Vec]
  -> AttentionCache
  -> [Vec]
  -> (AttentionGrads, [Vec])
attentionBackward AttentionParams{..} xs AttentionCache{..} dOuts =
  let !n  = length xs
      !dK = vlen (head acQ)
      !dk = fromIntegral dK
      !sf = 1 / sqrt dk

      perTokWo :: [(LinearParams, Vec)]
      perTokWo =
        [ linearBackward attnWo (acAttended !! i) (dOuts !! i)
        | i <- [0 .. n - 1]
        ]

      !gWoTotal = foldl' addLinear (zeroLinearGrads (vlen (head dOuts)) dK) (map fst perTokWo)
      !dAttended = map snd perTokWo

      !dVs =
        [ foldl' (+) (zerosV dK)
            [ LA.scale ((acWeights !! i) `LA.atIndex` j) (dAttended !! i)
            | i <- [0 .. n - 1]
            ]
        | j <- [0 .. n - 1]
        ]

      !dAlphas =
        [ LA.fromList
            [ (dAttended !! i) LA.<.> (acV !! j)
            | j <- [0 .. n - 1]
            ]
        | i <- [0 .. n - 1]
        ]

      !dDots =
        [ LA.scale sf (softmaxBackward (acWeights !! i) (dAlphas !! i))
        | i <- [0 .. n - 1]
        ]

      !dQs =
        [ foldl' (+) (zerosV dK)
            [ LA.scale ((dDots !! i) `LA.atIndex` j) (acK !! j)
            | j <- [0 .. n - 1]
            ]
        | i <- [0 .. n - 1]
        ]

      !dKs =
        [ foldl' (+) (zerosV dK)
            [ LA.scale ((dDots !! i) `LA.atIndex` j) (acQ !! i)
            | i <- [0 .. n - 1]
            ]
        | j <- [0 .. n - 1]
        ]

      perTokQ = [ linearBackward attnWq (xs !! i) (dQs !! i) | i <- [0 .. n - 1] ]
      perTokK = [ linearBackward attnWk (xs !! i) (dKs !! i) | i <- [0 .. n - 1] ]
      perTokV = [ linearBackward attnWv (xs !! i) (dVs !! i) | i <- [0 .. n - 1] ]

      !gWqTotal = foldl' addLinear (zeroLinearGrads dK (vlen (head xs))) (map fst perTokQ)
      !gWkTotal = foldl' addLinear (zeroLinearGrads dK (vlen (head xs))) (map fst perTokK)
      !gWvTotal = foldl' addLinear (zeroLinearGrads dK (vlen (head xs))) (map fst perTokV)

      !dX_from_q = map snd perTokQ
      !dX_from_k = map snd perTokK
      !dX_from_v = map snd perTokV

      !dXs = zipWith3 (\a b c -> a + b + c) dX_from_q dX_from_k dX_from_v

      !grads = AttentionGrads
        { gWqParams = gWqTotal
        , gWkParams = gWkTotal
        , gWvParams = gWvTotal
        , gWoParams = gWoTotal
        }
  in (grads, dXs)

--------------------------------------------------------------------------------
-- Full transformer
--------------------------------------------------------------------------------

data TransformerParams = TransformerParams
  { tokEmbed   :: !Mat
  , posEmbed   :: !Mat
  , attnParams :: !AttentionParams
  , ln1        :: !LayerNormParams
  , ffnParams  :: !FFNParams
  , ln2        :: !LayerNormParams
  , unembed    :: !LinearParams
  }

data ForwardCache = ForwardCache
  { cEmbedded   :: ![Vec]
  , cAttnCache  :: !AttentionCache
  , cLn1Out     :: ![Vec]
  , cLn1Cache   :: ![LayerNormCache]
  , cFfnCache   :: ![FFNCache]
  , cLn2Out     :: ![Vec]
  , cLn2Cache   :: ![LayerNormCache]
  , cLogits     :: !Vec
  , cProbs      :: !Vec
  }

forwardInfer :: TransformerParams -> Int -> Int -> Vec
forwardInfer TransformerParams{..} tokA tokB =
  let embA = tokEmbed LA.! tokA
      embB = tokEmbed LA.! tokB
      x0   = embA + (posEmbed LA.! 0)
      x1   = embB + (posEmbed LA.! 1)
      xs   = [x0, x1]

      (attnOut, _) = attentionForward attnParams xs
      residual1    = zipWith (+) xs attnOut
      ln1Out       = map (fst . layerNormForward ln1) residual1

      ffnOut       = map (fst . ffnForward ffnParams) ln1Out
      residual2    = zipWith (+) ln1Out ffnOut
      ln2Out       = map (fst . layerNormForward ln2) residual2

      readout = head ln2Out
      logits  = linearForward unembed readout
  in softmax logits

forwardTrain :: TransformerParams -> Int -> Int -> (Vec, ForwardCache)
forwardTrain TransformerParams{..} tokA tokB =
  let embA = tokEmbed LA.! tokA
      embB = tokEmbed LA.! tokB
      x0   = embA + (posEmbed LA.! 0)
      x1   = embB + (posEmbed LA.! 1)
      embedded = [x0, x1]

      (attnOut, attnCache) = attentionForward attnParams embedded
      residual1 = zipWith (+) embedded attnOut

      ln1Pairs   = map (layerNormForward ln1) residual1
      ln1Out     = map fst ln1Pairs
      ln1Cache   = map snd ln1Pairs

      ffnPairs   = map (ffnForward ffnParams) ln1Out
      ffnCache   = map snd ffnPairs

      ffnOut     = map fst ffnPairs
      residual2  = zipWith (+) ln1Out ffnOut
      ln2Pairs   = map (layerNormForward ln2) residual2
      ln2Out     = map fst ln2Pairs
      ln2Cache   = map snd ln2Pairs

      readout = head ln2Out
      logits  = linearForward unembed readout
      probs   = softmax logits

      cache = ForwardCache
        { cEmbedded  = embedded
        , cAttnCache = attnCache
        , cLn1Out    = ln1Out
        , cLn1Cache  = ln1Cache
        , cFfnCache  = ffnCache
        , cLn2Out    = ln2Out
        , cLn2Cache  = ln2Cache
        , cLogits    = logits
        , cProbs     = probs
        }
  in (probs, cache)

--------------------------------------------------------------------------------
-- Loss & Per-example grads (embedding grads returned separately for ST accumulation)
--------------------------------------------------------------------------------

crossEntropyLoss :: Vec -> Int -> R
crossEntropyLoss probs target =
  let p = max 1e-12 (probs `LA.atIndex` target)
  in negate (log p)

crossEntropyGradLogits :: Vec -> Int -> Vec
crossEntropyGradLogits probs target =
  let n = vlen probs
      oneHot = LA.assoc n 0 [(target, 1)]
  in probs - oneHot

data ExampleGrads = ExampleGrads
  { egDX0      :: !Vec
  , egDX1      :: !Vec
  , egUnembed  :: !LinearParams
  , egFFN1     :: !LinearParams
  , egFFN2     :: !LinearParams
  , egAttnWq   :: !LinearParams
  , egAttnWk   :: !LinearParams
  , egAttnWv   :: !LinearParams
  , egAttnWo   :: !LinearParams
  , egLn1Gamma :: !Vec
  , egLn1Beta  :: !Vec
  , egLn2Gamma :: !Vec
  , egLn2Beta  :: !Vec
  }

zeroExampleGrads :: Int -> Int -> Int -> Int -> ExampleGrads
zeroExampleGrads vocab dModel dFF dK =
  ExampleGrads
    { egDX0      = zerosV dModel
    , egDX1      = zerosV dModel
    , egUnembed  = zeroLinearGrads vocab dModel
    , egFFN1     = zeroLinearGrads dFF dModel
    , egFFN2     = zeroLinearGrads dModel dFF
    , egAttnWq   = zeroLinearGrads dK dModel
    , egAttnWk   = zeroLinearGrads dK dModel
    , egAttnWv   = zeroLinearGrads dK dModel
    , egAttnWo   = zeroLinearGrads dModel dK
    , egLn1Gamma = zerosV dModel
    , egLn1Beta  = zerosV dModel
    , egLn2Gamma = zerosV dModel
    , egLn2Beta  = zerosV dModel
    }

addExampleGrads :: ExampleGrads -> ExampleGrads -> ExampleGrads
addExampleGrads a b =
  ExampleGrads
    { egDX0      = egDX0 a + egDX0 b
    , egDX1      = egDX1 a + egDX1 b
    , egUnembed  = addLinear (egUnembed a) (egUnembed b)
    , egFFN1     = addLinear (egFFN1 a) (egFFN1 b)
    , egFFN2     = addLinear (egFFN2 a) (egFFN2 b)
    , egAttnWq   = addLinear (egAttnWq a) (egAttnWq b)
    , egAttnWk   = addLinear (egAttnWk a) (egAttnWk b)
    , egAttnWv   = addLinear (egAttnWv a) (egAttnWv b)
    , egAttnWo   = addLinear (egAttnWo a) (egAttnWo b)
    , egLn1Gamma = egLn1Gamma a + egLn1Gamma b
    , egLn1Beta  = egLn1Beta a + egLn1Beta b
    , egLn2Gamma = egLn2Gamma a + egLn2Gamma b
    , egLn2Beta  = egLn2Beta a + egLn2Beta b
    }

scaleExampleGrads :: R -> ExampleGrads -> ExampleGrads
scaleExampleGrads s g =
  g { egDX0      = LA.scale s (egDX0 g)
    , egDX1      = LA.scale s (egDX1 g)
    , egUnembed  = scaleLinear s (egUnembed g)
    , egFFN1     = scaleLinear s (egFFN1 g)
    , egFFN2     = scaleLinear s (egFFN2 g)
    , egAttnWq   = scaleLinear s (egAttnWq g)
    , egAttnWk   = scaleLinear s (egAttnWk g)
    , egAttnWv   = scaleLinear s (egAttnWv g)
    , egAttnWo   = scaleLinear s (egAttnWo g)
    , egLn1Gamma = LA.scale s (egLn1Gamma g)
    , egLn1Beta  = LA.scale s (egLn1Beta g)
    , egLn2Gamma = LA.scale s (egLn2Gamma g)
    , egLn2Beta  = LA.scale s (egLn2Beta g)
    }

backward
  :: TransformerParams
  -> ForwardCache
  -> Int -> Int -> Int
  -> ExampleGrads
backward TransformerParams{..} cache _tokA _tokB target =
  let
    !dModel = vlen (lnGamma ln1)
    !vocab  = LA.rows tokEmbed
    !n      = length (cEmbedded cache)

    !dLogits = crossEntropyGradLogits (cProbs cache) target
    !readout = head (cLn2Out cache)
    (gUnemb, dReadout) = linearBackward unembed readout dLogits

    !dLn2Outs = dReadout : replicate (n - 1) (zerosV dModel)

    ln2Backs :: [(Vec, Vec, Vec)]
    ln2Backs =
      [ layerNormBackward ln2 (cLn2Cache cache !! i) (dLn2Outs !! i)
      | i <- [0 .. n - 1]
      ]

    !dResidual2 = map (\(dx,_,_) -> dx) ln2Backs
    !gLn2G      = foldl' (+) (zerosV dModel) (map (\(_,dg,_) -> dg) ln2Backs)
    !gLn2B      = foldl' (+) (zerosV dModel) (map (\(_,_,db) -> db) ln2Backs)

    !dLn1Out_from_res2 = dResidual2
    !dFfnOut           = dResidual2

    ffnBacks :: [(LinearParams, LinearParams, Vec)]
    ffnBacks =
      [ ffnBackward ffnParams (cLn1Out cache !! i) (cFfnCache cache !! i) (dFfnOut !! i)
      | i <- [0 .. n - 1]
      ]

    !gFFN1Total = foldl' addLinear (zeroLinearGrads (vlen (linB (ffnLinear1 ffnParams))) dModel) (map (\(g1,_,_) -> g1) ffnBacks)
    !gFFN2Total = foldl' addLinear (zeroLinearGrads dModel (vlen (linB (ffnLinear1 ffnParams)))) (map (\(_,g2,_) -> g2) ffnBacks)
    !dLn1Out_from_ffn = map (\(_,_,dx) -> dx) ffnBacks

    !dLn1Out = zipWith (+) dLn1Out_from_res2 dLn1Out_from_ffn

    ln1Backs :: [(Vec, Vec, Vec)]
    ln1Backs =
      [ layerNormBackward ln1 (cLn1Cache cache !! i) (dLn1Out !! i)
      | i <- [0 .. n - 1]
      ]

    !dResidual1 = map (\(dx,_,_) -> dx) ln1Backs
    !gLn1G      = foldl' (+) (zerosV dModel) (map (\(_,dg,_) -> dg) ln1Backs)
    !gLn1B      = foldl' (+) (zerosV dModel) (map (\(_,_,db) -> db) ln1Backs)

    !dAttnOut            = dResidual1

    attnC = cAttnCache cache
    xs    = cEmbedded cache
    (attnGrads, dEmbedded_from_attn) =
      attentionBackward attnParams xs attnC dAttnOut

    !dEmbedded = zipWith (+) dResidual1 dEmbedded_from_attn
    !dX0 = dEmbedded !! 0
    !dX1 = dEmbedded !! 1

    !dK = LA.rows (linW (attnWq attnParams)) -- out dim of Wq
  in ExampleGrads
      { egDX0      = dX0
      , egDX1      = dX1
      , egUnembed  = gUnemb
      , egFFN1     = gFFN1Total
      , egFFN2     = gFFN2Total
      , egAttnWq   = gWqParams attnGrads
      , egAttnWk   = gWkParams attnGrads
      , egAttnWv   = gWvParams attnGrads
      , egAttnWo   = gWoParams attnGrads
      , egLn1Gamma = gLn1G
      , egLn1Beta  = gLn1B
      , egLn2Gamma = gLn2G
      , egLn2Beta  = gLn2B
      }

--------------------------------------------------------------------------------
-- AdamW + schedule
--------------------------------------------------------------------------------

data MatMoments = MatMoments { mmM :: !Mat, mmV :: !Mat }
data VecMoments = VecMoments { vmM :: !Vec, vmV :: !Vec }

data LinearMoments = LinearMoments
  { lmW :: !MatMoments
  , lmB :: !VecMoments
  }

data AdamState = AdamState
  { asTimestep :: !Int
  , asB1Pow    :: !R
  , asB2Pow    :: !R
  , asTokMom   :: !MatMoments
  , asPosMom   :: !MatMoments
  , asWqMom    :: !LinearMoments
  , asWkMom    :: !LinearMoments
  , asWvMom    :: !LinearMoments
  , asWoMom    :: !LinearMoments
  , asLn1GMom  :: !VecMoments
  , asLn1BMom  :: !VecMoments
  , asFF1Mom   :: !LinearMoments
  , asFF2Mom   :: !LinearMoments
  , asLn2GMom  :: !VecMoments
  , asLn2BMom  :: !VecMoments
  , asUnMom    :: !LinearMoments
  }

data AdamConfig = AdamConfig
  { adamB1  :: !R
  , adamB2  :: !R
  , adamEps :: !R
  , adamWD  :: !R
  }

zeroMatMomentsLike :: Mat -> MatMoments
zeroMatMomentsLike w = MatMoments (LA.scale 0 w) (LA.scale 0 w)

zeroVecMomentsLike :: Vec -> VecMoments
zeroVecMomentsLike v = VecMoments (LA.scale 0 v) (LA.scale 0 v)

zeroLinearMomentsLike :: LinearParams -> LinearMoments
zeroLinearMomentsLike LinearParams{..} =
  LinearMoments (zeroMatMomentsLike linW) (zeroVecMomentsLike linB)

initAdamState :: TransformerParams -> AdamState
initAdamState p =
  AdamState
    { asTimestep = 0
    , asB1Pow    = 1
    , asB2Pow    = 1
    , asTokMom   = zeroMatMomentsLike (tokEmbed p)
    , asPosMom   = zeroMatMomentsLike (posEmbed p)
    , asWqMom    = zeroLinearMomentsLike (attnWq (attnParams p))
    , asWkMom    = zeroLinearMomentsLike (attnWk (attnParams p))
    , asWvMom    = zeroLinearMomentsLike (attnWv (attnParams p))
    , asWoMom    = zeroLinearMomentsLike (attnWo (attnParams p))
    , asLn1GMom  = zeroVecMomentsLike (lnGamma (ln1 p))
    , asLn1BMom  = zeroVecMomentsLike (lnBeta  (ln1 p))
    , asFF1Mom   = zeroLinearMomentsLike (ffnLinear1 (ffnParams p))
    , asFF2Mom   = zeroLinearMomentsLike (ffnLinear2 (ffnParams p))
    , asLn2GMom  = zeroVecMomentsLike (lnGamma (ln2 p))
    , asLn2BMom  = zeroVecMomentsLike (lnBeta  (ln2 p))
    , asUnMom    = zeroLinearMomentsLike (unembed p)
    }

-- LR schedule: warmup + cosine decay
lrWarmupCosine
  :: Int  -- warmup steps
  -> Int  -- total steps
  -> R    -- peak lr
  -> R    -- min lr
  -> Int  -- step (1-based)
  -> R
lrWarmupCosine warmup total base minLR step
  | total <= 1 = base
  | step <= 0  = 0
  | step <= warmup =
      base * (fromIntegral step / fromIntegral (max 1 warmup))
  | otherwise =
      let !t  = fromIntegral (min step total)
          !w  = fromIntegral warmup
          !tt = fromIntegral total
          !progress = (t - w) / max 1 (tt - w)
          !cosine   = 0.5 * (1 + cos (pi * progress))
      in minLR + (base - minLR) * cosine

-- AdamW update helpers (elementwise ops via (*) and (/))
adamUpdateMat :: R -> AdamConfig -> R -> R -> Mat -> MatMoments -> Mat -> (Mat, MatMoments)
adamUpdateMat lr AdamConfig{..} b1Pow' b2Pow' w (MatMoments m v) g =
  let !m'   = LA.scale adamB1 m + LA.scale (1 - adamB1) g
      !g2   = g * g
      !v'   = LA.scale adamB2 v + LA.scale (1 - adamB2) g2
      !mHat = LA.scale (1 / (1 - b1Pow')) m'
      !vHat = LA.scale (1 / (1 - b2Pow')) v'
      !den  = LA.cmap (+ adamEps) (LA.cmap sqrt vHat)
      !step = mHat / den
      -- decoupled weight decay:
      !w'   = w - LA.scale lr step - LA.scale (lr * adamWD) w
  in (w', MatMoments m' v')

adamUpdateVecNoWD :: R -> AdamConfig -> R -> R -> Vec -> VecMoments -> Vec -> (Vec, VecMoments)
adamUpdateVecNoWD lr AdamConfig{..} b1Pow' b2Pow' w (VecMoments m v) g =
  let !m'   = LA.scale adamB1 m + LA.scale (1 - adamB1) g
      !g2   = g * g
      !v'   = LA.scale adamB2 v + LA.scale (1 - adamB2) g2
      !mHat = LA.scale (1 / (1 - b1Pow')) m'
      !vHat = LA.scale (1 / (1 - b2Pow')) v'
      !den  = LA.cmap (+ adamEps) (LA.cmap sqrt vHat)
      !step = mHat / den
      !w'   = w - LA.scale lr step
  in (w', VecMoments m' v')

adamUpdateLinear :: R -> AdamConfig -> R -> R -> LinearParams -> LinearMoments -> LinearParams -> (LinearParams, LinearMoments)
adamUpdateLinear lr cfg b1Pow' b2Pow' p (LinearMoments wm bm) g =
  let (w', wm') = adamUpdateMat lr cfg b1Pow' b2Pow' (linW p) wm (linW g)
      -- No WD on bias
      (b', bm') = adamUpdateVecNoWD lr cfg b1Pow' b2Pow' (linB p) bm (linB g)
  in (LinearParams w' b', LinearMoments wm' bm')

adamStep
  :: R                -- current lr
  -> AdamConfig
  -> TransformerParams
  -> Mat -> Mat        -- gTok, gPos
  -> ExampleGrads
  -> AdamState
  -> (TransformerParams, AdamState)
adamStep lr cfg p gTok gPos eg st =
  let !t'     = asTimestep st + 1
      !b1Pow' = asB1Pow st * adamB1 cfg
      !b2Pow' = asB2Pow st * adamB2 cfg

      (tok', tokMom') = adamUpdateMat lr cfg b1Pow' b2Pow' (tokEmbed p) (asTokMom st) gTok
      (pos', posMom') = adamUpdateMat lr cfg b1Pow' b2Pow' (posEmbed p) (asPosMom st) gPos

      AttentionParams wq wk wv wo = attnParams p
      (wq', wqMom') = adamUpdateLinear lr cfg b1Pow' b2Pow' wq (asWqMom st) (egAttnWq eg)
      (wk', wkMom') = adamUpdateLinear lr cfg b1Pow' b2Pow' wk (asWkMom st) (egAttnWk eg)
      (wv', wvMom') = adamUpdateLinear lr cfg b1Pow' b2Pow' wv (asWvMom st) (egAttnWv eg)
      (wo', woMom') = adamUpdateLinear lr cfg b1Pow' b2Pow' wo (asWoMom st) (egAttnWo eg)

      -- LN params: no WD
      (ln1g', ln1gMom') = adamUpdateVecNoWD lr cfg b1Pow' b2Pow' (lnGamma (ln1 p)) (asLn1GMom st) (egLn1Gamma eg)
      (ln1b', ln1bMom') = adamUpdateVecNoWD lr cfg b1Pow' b2Pow' (lnBeta  (ln1 p)) (asLn1BMom st) (egLn1Beta  eg)

      FFNParams ff1 ff2 = ffnParams p
      (ff1', ff1Mom') = adamUpdateLinear lr cfg b1Pow' b2Pow' ff1 (asFF1Mom st) (egFFN1 eg)
      (ff2', ff2Mom') = adamUpdateLinear lr cfg b1Pow' b2Pow' ff2 (asFF2Mom st) (egFFN2 eg)

      (ln2g', ln2gMom') = adamUpdateVecNoWD lr cfg b1Pow' b2Pow' (lnGamma (ln2 p)) (asLn2GMom st) (egLn2Gamma eg)
      (ln2b', ln2bMom') = adamUpdateVecNoWD lr cfg b1Pow' b2Pow' (lnBeta  (ln2 p)) (asLn2BMom st) (egLn2Beta  eg)

      (un', unMom') = adamUpdateLinear lr cfg b1Pow' b2Pow' (unembed p) (asUnMom st) (egUnembed eg)

      p' = TransformerParams
        { tokEmbed   = tok'
        , posEmbed   = pos'
        , attnParams = AttentionParams wq' wk' wv' wo'
        , ln1        = LayerNormParams ln1g' ln1b'
        , ffnParams  = FFNParams ff1' ff2'
        , ln2        = LayerNormParams ln2g' ln2b'
        , unembed    = un'
        }

      st' = st
        { asTimestep = t'
        , asB1Pow    = b1Pow'
        , asB2Pow    = b2Pow'
        , asTokMom   = tokMom'
        , asPosMom   = posMom'
        , asWqMom    = wqMom'
        , asWkMom    = wkMom'
        , asWvMom    = wvMom'
        , asWoMom    = woMom'
        , asLn1GMom  = ln1gMom'
        , asLn1BMom  = ln1bMom'
        , asFF1Mom   = ff1Mom'
        , asFF2Mom   = ff2Mom'
        , asLn2GMom  = ln2gMom'
        , asLn2BMom  = ln2bMom'
        , asUnMom    = unMom'
        }
  in (p', st')

--------------------------------------------------------------------------------
-- Xavier init (no LA.uniformSample dependency)
--------------------------------------------------------------------------------

randomListN :: Int -> (R, R) -> StdGen -> ([R], StdGen)
randomListN n range0 g0 = go n g0 []
  where
    go 0 !g acc = (reverse acc, g)
    go k !g acc =
      let (x, g') = randomR range0 g
      in go (k - 1) g' (x : acc)

xavierInit :: Int -> Int -> Int -> (Mat, Int)
xavierInit r c seed0 =
  let bound = sqrt (6 / fromIntegral (r + c))
      g0 = mkStdGen seed0
      (vals, g1) = randomListN (r * c) (-bound, bound) g0
      (seed1, _) = (random g1 :: (Int, StdGen))
  in ((r LA.>< c) vals, seed1)

initLinear :: Int -> Int -> Int -> (LinearParams, Int)
initLinear outDim inDim seed0 =
  let (w, seed1) = xavierInit outDim inDim seed0
  in (LinearParams w (zerosV outDim), seed1)

initTransformer :: Int -> Int -> Int -> Int -> Int -> (TransformerParams, Int)
initTransformer vocab dModel dFF dK seed0 =
  let (tokE,  s1) = xavierInit vocab dModel seed0
      (posE,  s2) = xavierInit 2     dModel s1
      (wq,    s3) = initLinear dK     dModel s2
      (wk,    s4) = initLinear dK     dModel s3
      (wv,    s5) = initLinear dK     dModel s4
      (wo,    s6) = initLinear dModel dK     s5
      (ff1,   s7) = initLinear dFF    dModel s6
      (ff2,   s8) = initLinear dModel dFF    s7
      (unemb, s9) = initLinear vocab  dModel s8
      params = TransformerParams
        { tokEmbed   = tokE
        , posEmbed   = posE
        , attnParams = AttentionParams wq wk wv wo
        , ln1        = initLayerNorm dModel
        , ffnParams  = FFNParams ff1 ff2
        , ln2        = initLayerNorm dModel
        , unembed    = unemb
        }
  in (params, s9)

--------------------------------------------------------------------------------
-- ST embedding accumulation (column-major buffer to match (><) / flatten
--------------------------------------------------------------------------------

-- Column-major index: ix = row + col*rows
addRowCM :: VSM.MVector s R -> Int -> Int -> Vec -> ST s ()
addRowCM buf !rows !row !v = do
  let !cols = vlen v
  forM_ [0 .. cols - 1] $ \j -> do
    let !ix = row + j * rows
    old <- VSM.unsafeRead buf ix
    let !new = old + (v `LA.atIndex` j)
    VSM.unsafeWrite buf ix new

--------------------------------------------------------------------------------
-- Data & evaluation
--------------------------------------------------------------------------------

type Example = (Int, Int, Int)

generateData :: Int -> [Example]
generateData p = [ (a, b, (a + b) `mod` p) | a <- [0 .. p - 1], b <- [0 .. p - 1] ]

shuffle :: Int -> [a] -> ([a], Int)
shuffle seed xs = runST $ do
  let v0 = V.fromList xs
      n  = V.length v0
  mv <- V.thaw v0
  gRef <- newSTRef (mkStdGen seed)
  forM_ [n-1, n-2 .. 1] $ \i -> do
    g <- readSTRef gRef
    let (j, g') = randomR (0, i) g
    writeSTRef gRef g'
    MV.swap mv i j
  v1 <- V.freeze mv
  gFinal <- readSTRef gRef
  let (seed', _) = (random gFinal :: (Int, StdGen))
  pure (V.toList v1, seed')

splitData :: Double -> Int -> [a] -> ([a], [a], Int)
splitData frac seed xs =
  let (shuf, seed1) = shuffle seed xs
      n = length shuf
      nTrain = round (frac * fromIntegral n)
  in (take nTrain shuf, drop nTrain shuf, seed1)

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = let (h, t) = splitAt n xs in h : chunksOf n t

evaluate :: TransformerParams -> [Example] -> Double
evaluate params examples =
  let correct = length $ filter (\(a,b,t) -> LA.maxIndex (forwardInfer params a b) == t) examples
  in fromIntegral correct / fromIntegral (length examples)

--------------------------------------------------------------------------------
-- Training (batched)
--------------------------------------------------------------------------------

trainBatch
  :: TransformerParams
  -> AdamState
  -> AdamConfig
  -> (Int -> R)             -- lr(step)
  -> Int -> Int -> Int -> Int -- vocab dModel dFF dK
  -> [Example]
  -> (TransformerParams, AdamState, R)
trainBatch params st0 cfg lrOfStep vocab dModel dFF dK batch =
  let (!gTok, !gPos, !egSum, !lossSum) = runST $ do
        tokBuf <- VSM.replicate (vocab * dModel) 0
        posBuf <- VSM.replicate (2 * dModel) 0
        gRef   <- newSTRef (zeroExampleGrads vocab dModel dFF dK)
        lRef   <- newSTRef 0

        forM_ batch $ \(a,b,tgt) -> do
          let (probs, cache) = forwardTrain params a b
              !l = crossEntropyLoss probs tgt
              eg = backward params cache a b tgt

          addRowCM tokBuf vocab a (egDX0 eg)
          addRowCM tokBuf vocab b (egDX1 eg)
          addRowCM posBuf 2     0 (egDX0 eg)
          addRowCM posBuf 2     1 (egDX1 eg)

          modifySTRef' gRef (`addExampleGrads` eg)
          modifySTRef' lRef (+ l)

        tokV <- VS.unsafeFreeze tokBuf
        posV <- VS.unsafeFreeze posBuf
        egAcc <- readSTRef gRef
        lAcc  <- readSTRef lRef

        let !gTokM = (vocab LA.>< dModel) (VS.toList tokV)
            !gPosM = (2     LA.>< dModel) (VS.toList posV)
        pure (gTokM, gPosM, egAcc, lAcc)

      !bs = max 1 (length batch)
      !invBS = 1 / fromIntegral bs
      !gTokAvg = LA.scale invBS gTok
      !gPosAvg = LA.scale invBS gPos
      !egAvg   = scaleExampleGrads invBS egSum

      !stepNow = asTimestep st0 + 1
      !lrNow   = lrOfStep stepNow

      (params', st') = adamStep lrNow cfg params gTokAvg gPosAvg egAvg st0
  in (params', st', lossSum)

trainEpoch
  :: TransformerParams
  -> AdamState
  -> AdamConfig
  -> (Int -> R)
  -> Int -> Int -> Int -> Int
  -> Int
  -> [Example]
  -> Int
  -> (TransformerParams, AdamState, R, Int)
trainEpoch params st cfg lrSched vocab dModel dFF dK batchSize examples seed =
  let (shuf, seed1) = shuffle seed examples
      batches = chunksOf batchSize shuf

      step (!p, !as, !lossAcc) batch =
        let (p', as', l) = trainBatch p as cfg lrSched vocab dModel dFF dK batch
        in (p', as', lossAcc + l)

      (!finalP, !finalS, !totalLoss) = foldl' step (params, st, 0) batches
      avgLoss = totalLoss / fromIntegral (length examples)
  in (finalP, finalS, avgLoss, seed1)

--------------------------------------------------------------------------------
-- Text checkpoint: params + adam moments + timestep
--------------------------------------------------------------------------------

matToList :: Mat -> [R]
matToList m = LA.toList (LA.flatten m)

listToMat :: Int -> Int -> [R] -> Mat
listToMat r c xs = (r LA.>< c) xs

vecToList :: Vec -> [R]
vecToList = LA.toList

listToVec :: [R] -> Vec
listToVec = LA.fromList

takeN :: Int -> [R] -> ([R], [R])
takeN n xs = splitAt n xs

serializeLinear :: LinearParams -> [R]
serializeLinear LinearParams{..} = matToList linW ++ vecToList linB

deserializeLinear :: Int -> Int -> [R] -> (LinearParams, [R])
deserializeLinear outDim inDim xs =
  let (wData, xs1) = takeN (outDim * inDim) xs
      (bData, xs2) = takeN outDim xs1
  in (LinearParams (listToMat outDim inDim wData) (listToVec bData), xs2)

serializeLN :: LayerNormParams -> [R]
serializeLN LayerNormParams{..} = vecToList lnGamma ++ vecToList lnBeta

deserializeLN :: Int -> [R] -> (LayerNormParams, [R])
deserializeLN d xs =
  let (gData, xs1) = takeN d xs
      (bData, xs2) = takeN d xs1
  in (LayerNormParams (listToVec gData) (listToVec bData), xs2)

serializeParams :: TransformerParams -> [R]
serializeParams TransformerParams{..} = concat
  [ matToList tokEmbed, matToList posEmbed
  , serializeLinear (attnWq attnParams), serializeLinear (attnWk attnParams)
  , serializeLinear (attnWv attnParams), serializeLinear (attnWo attnParams)
  , serializeLN ln1
  , serializeLinear (ffnLinear1 ffnParams), serializeLinear (ffnLinear2 ffnParams)
  , serializeLN ln2
  , serializeLinear unembed
  ]

deserializeParams :: Int -> Int -> Int -> Int -> [R] -> (TransformerParams, [R])
deserializeParams vocab dModel dFF dK xs0 =
  let (tokData, xs1)  = takeN (vocab * dModel) xs0
      (posData, xs2)  = takeN (2 * dModel)     xs1
      (wq,      xs3)  = deserializeLinear dK     dModel xs2
      (wk,      xs4)  = deserializeLinear dK     dModel xs3
      (wv,      xs5)  = deserializeLinear dK     dModel xs4
      (wo,      xs6)  = deserializeLinear dModel  dK    xs5
      (ln1p,    xs7)  = deserializeLN dModel xs6
      (ff1,     xs8)  = deserializeLinear dFF    dModel xs7
      (ff2,     xs9)  = deserializeLinear dModel  dFF   xs8
      (ln2p,    xs10) = deserializeLN dModel xs9
      (unemb,   xs11) = deserializeLinear vocab  dModel xs10
      p = TransformerParams
          { tokEmbed   = listToMat vocab dModel tokData
          , posEmbed   = listToMat 2 dModel posData
          , attnParams = AttentionParams wq wk wv wo
          , ln1        = ln1p
          , ffnParams  = FFNParams ff1 ff2
          , ln2        = ln2p
          , unembed    = unemb
          }
  in (p, xs11)

serializeMatMom :: MatMoments -> [R]
serializeMatMom (MatMoments m v) = matToList m ++ matToList v

deserializeMatMom :: Int -> Int -> [R] -> (MatMoments, [R])
deserializeMatMom r c xs =
  let (mData, xs1) = takeN (r*c) xs
      (vData, xs2) = takeN (r*c) xs1
  in (MatMoments (listToMat r c mData) (listToMat r c vData), xs2)

serializeVecMom :: VecMoments -> [R]
serializeVecMom (VecMoments m v) = vecToList m ++ vecToList v

deserializeVecMom :: Int -> [R] -> (VecMoments, [R])
deserializeVecMom n xs =
  let (mData, xs1) = takeN n xs
      (vData, xs2) = takeN n xs1
  in (VecMoments (listToVec mData) (listToVec vData), xs2)

serializeLinearMom :: LinearMoments -> [R]
serializeLinearMom (LinearMoments wm bm) = serializeMatMom wm ++ serializeVecMom bm

deserializeLinearMom :: Int -> Int -> [R] -> (LinearMoments, [R])
deserializeLinearMom outDim inDim xs =
  let (wm, xs1) = deserializeMatMom outDim inDim xs
      (bm, xs2) = deserializeVecMom outDim xs1
  in (LinearMoments wm bm, xs2)

serializeAdam :: AdamState -> [R]
serializeAdam AdamState{..} =
  concat
    [ serializeMatMom asTokMom
    , serializeMatMom asPosMom
    , serializeLinearMom asWqMom
    , serializeLinearMom asWkMom
    , serializeLinearMom asWvMom
    , serializeLinearMom asWoMom
    , serializeVecMom asLn1GMom
    , serializeVecMom asLn1BMom
    , serializeLinearMom asFF1Mom
    , serializeLinearMom asFF2Mom
    , serializeVecMom asLn2GMom
    , serializeVecMom asLn2BMom
    , serializeLinearMom asUnMom
    ]

deserializeAdam
  :: Int -> Int -> Int -> Int
  -> [R]
  -> (AdamState, [R])
deserializeAdam vocab dModel dFF dK xs0 =
  let (tokMom, xs1) = deserializeMatMom vocab dModel xs0
      (posMom, xs2) = deserializeMatMom 2 dModel xs1
      (wqMom,  xs3) = deserializeLinearMom dK dModel xs2
      (wkMom,  xs4) = deserializeLinearMom dK dModel xs3
      (wvMom,  xs5) = deserializeLinearMom dK dModel xs4
      (woMom,  xs6) = deserializeLinearMom dModel dK xs5
      (ln1GM,  xs7) = deserializeVecMom dModel xs6
      (ln1BM,  xs8) = deserializeVecMom dModel xs7
      (ff1M,   xs9) = deserializeLinearMom dFF dModel xs8
      (ff2M,   xs10)= deserializeLinearMom dModel dFF xs9
      (ln2GM,  xs11)= deserializeVecMom dModel xs10
      (ln2BM,  xs12)= deserializeVecMom dModel xs11
      (unM,    xs13)= deserializeLinearMom vocab dModel xs12
      dummy = AdamState
        { asTimestep = 0, asB1Pow = 1, asB2Pow = 1
        , asTokMom = tokMom, asPosMom = posMom
        , asWqMom = wqMom, asWkMom = wkMom, asWvMom = wvMom, asWoMom = woMom
        , asLn1GMom = ln1GM, asLn1BMom = ln1BM
        , asFF1Mom = ff1M, asFF2Mom = ff2M
        , asLn2GMom = ln2GM, asLn2BMom = ln2BM
        , asUnMom = unM
        }
  in (dummy, xs13)

saveCheckpoint :: FilePath -> TransformerParams -> AdamState -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
saveCheckpoint path params st epoch seed vocab dModel dFF dK = do
  let header = unwords
        [ "CHECKPOINT_ADAMW"
        , show epoch, show seed
        , show (asTimestep st)
        , show (asB1Pow st), show (asB2Pow st)
        , show vocab, show dModel, show dFF, show dK
        ]
      doubles = serializeParams params ++ serializeAdam st
      contents = unlines (header : map show doubles)
  writeFile path contents
  printf "  [checkpoint] Saved epoch %d (step=%d) to %s\n" epoch (asTimestep st) path

loadCheckpoint :: FilePath -> IO (Maybe (TransformerParams, AdamState, Int, Int))
loadCheckpoint path = do
  exists <- doesFileExist path
  if not exists then pure Nothing else do
    contents <- readFile path
    length contents `seq` pure ()
    case lines contents of
      (header:rest) ->
        case words header of
          ("CHECKPOINT_ADAMW":e:s:ts:b1p:b2p:v:dm:df:dk:_) -> do
            let epoch  = read e
                seed   = read s
                tstep  = read ts
                b1pow  = read b1p
                b2pow  = read b2p
                vocab  = read v
                dModel = read dm
                dFF    = read df
                dK     = read dk
                nums   = map read rest
                (params, xs1) = deserializeParams vocab dModel dFF dK nums
                (adam0, _xs2) = deserializeAdam vocab dModel dFF dK xs1
                adam = adam0 { asTimestep = tstep, asB1Pow = b1pow, asB2Pow = b2pow }
            pure (Just (params, adam, epoch, seed))

          -- Back-compat: old params-only checkpoints
          ("CHECKPOINT":e:s:v:dm:df:dk:_) -> do
            let epoch  = read e
                seed   = read s
                vocab  = read v
                dModel = read dm
                dFF    = read df
                dK     = read dk
                nums   = map read rest
                (params, _rest2) = deserializeParams vocab dModel dFF dK nums
                adam = initAdamState params
            pure (Just (params, adam, epoch, seed))

          _ -> pure Nothing
      _ -> pure Nothing

--------------------------------------------------------------------------------
-- Main
--------------------------------------------------------------------------------

main :: IO ()
main = do
  putStrLn "╔══════════════════════════════════════════════════════════════╗"
  putStrLn "║  Modular Transformer (AdamW + Warmup/Cosine + ST Embedding)  ║"
  putStrLn "╚══════════════════════════════════════════════════════════════╝"

  let p         = 53
      dModel    = 64
      dFF       = 256
      dK        = 64
      batchSize = 32

      baseLR    = 1e-3
      minLR     = 1e-5
      wd        = 1e-2

      epochs    = 20000
      trainFrac = 0.5
      seed0     = 42
      ckptPath  = "checkpoint.ckpt"

      cfg = AdamConfig
        { adamB1  = 0.9
        , adamB2  = 0.999
        , adamEps = 1e-8
        , adamWD  = wd
        }

  let allData = generateData p
  let (trainData, testData, seed1) = splitData trainFrac seed0 allData

  let batchesPerEpoch = max 1 ((length trainData + batchSize - 1) `div` batchSize)
      totalSteps      = epochs * batchesPerEpoch
      warmupSteps     = min 2000 (max 100 (totalSteps `div` 20))
      lrSched step    = lrWarmupCosine warmupSteps totalSteps baseLR minLR step

  mCkpt <- loadCheckpoint ckptPath
  let (initParams, initState, startEpoch, seed2) =
        case mCkpt of
          Just (cp, st, ce, cs) -> (cp, st, ce + 1, cs)
          Nothing ->
            let (fp, s) = initTransformer p dModel dFF dK seed1
                st = initAdamState fp
            in (fp, st, 1, s)

  printf "Params: p=%d dModel=%d dFF=%d dK=%d Batch=%d\n" p dModel dFF dK batchSize
  printf "AdamW: baseLR=%.6f minLR=%.6f warmup=%d steps totalSteps=%d wd=%.4f\n"
    baseLR minLR warmupSteps totalSteps wd
  printf "Data: %d train, %d test\n\n" (length trainData) (length testData)

  let loop !epoch !params !st !seed
        | epoch > epochs = putStrLn "Done."
        | otherwise = do
            let (!params', !st', !loss, !seed') =
                  trainEpoch params st cfg lrSched p dModel dFF dK batchSize trainData seed

            when (epoch `mod` 100 == 0) $ do
              let trAcc = evaluate params' trainData
                  teAcc = evaluate params' testData
                  lrNow = lrSched (asTimestep st')
              printf "%5d | step=%7d | lr=%.6f | loss=%.4f | train=%.1f%% | test=%.1f%%\n"
                epoch (asTimestep st') lrNow loss (trAcc*100) (teAcc*100)
              hFlush stdout

            when (epoch `mod` 1000 == 0) $
              saveCheckpoint ckptPath params' st' epoch seed' p dModel dFF dK

            loop (epoch + 1) params' st' seed'

  loop startEpoch initParams initState seed2
