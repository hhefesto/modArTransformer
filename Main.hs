{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

{- |
Module      : ModularTransformer
Description : A small, *correct* 1-layer transformer trained on (a + b) mod p.

Key goals vs the earlier version:
  • Correct LayerNorm backward (and we train gamma/beta).
  • Correct attention backward (single-head, row-wise softmax).
  • Proper Fisher–Yates shuffle (uniform, O(n)).
  • Separate inference forward path (no training cache).
  • Much cleaner gradients accumulation and parameter updates.
  • Checkpoint save/load for pausing and resuming training.

Dependencies (common):
  • hmatrix
  • vector
  • random

Build (example):
  stack ghci --package hmatrix --package vector --package random

Notes:
  • Sequence length is 2 (tokens a,b). We read out from position 0.
  • Vocab size is p (0..p-1).
-}

module Main where

import qualified Numeric.LinearAlgebra as LA
import           Numeric.LinearAlgebra (Matrix, Vector, R)

import qualified Data.Vector as V
import qualified Data.Vector.Mutable as MV

import           Control.Monad (forM_, when)
import           Control.Monad.ST (runST)
import           Data.List (foldl')
import           Data.STRef (newSTRef, readSTRef, writeSTRef)
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
addLinear (LinearParams w1 b1) (LinearParams w2 b2) = LinearParams (w1 + w2) (b1 + b2)

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
-- Given upstream dout, returns (dx, dGamma, dBeta).
layerNormBackward :: LayerNormParams -> LayerNormCache -> Vec -> (Vec, Vec, Vec)
layerNormBackward LayerNormParams{..} LayerNormCache{..} dOut =
  let !n      = vlen dOut
      !d      = fromIntegral n

      -- grads for affine
      !dBeta  = dOut
      !dGamma = dOut * lnXHat

      -- back through xHat
      !dXHat  = dOut * lnGamma

      !sumDXHat      = LA.sumElements dXHat
      !sumDXHatXHat  = LA.sumElements (dXHat * lnXHat)

      -- dx = (invStd / d) * (d*dXHat - sum(dXHat) - xHat*sum(dXHat*xHat))
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
  let (g2, dAct)   = linearBackward ffnLinear2 ffnActivated dOut
      !dH          = dAct * relu' ffnHidden
      (g1, dX)     = linearBackward ffnLinear1 x dH
  in (g1, g2, dX)

--------------------------------------------------------------------------------
-- Self-attention (single head, general n)
--------------------------------------------------------------------------------

data AttentionParams = AttentionParams
  { attnWq :: !LinearParams -- d_model -> d_k
  , attnWk :: !LinearParams -- d_model -> d_k
  , attnWv :: !LinearParams -- d_model -> d_k
  , attnWo :: !LinearParams -- d_k     -> d_model
  }

data AttentionCache = AttentionCache
  { acQ        :: ![Vec] -- n × d_k
  , acK        :: ![Vec] -- n × d_k
  , acV        :: ![Vec] -- n × d_k
  , acWeights  :: ![Vec] -- n × n (row-wise softmax)
  , acAttended :: ![Vec] -- n × d_k (before Wo)
  }

attentionForward :: AttentionParams -> [Vec] -> ([Vec], AttentionCache)
attentionForward AttentionParams{..} xs =
  let !qs = map (linearForward attnWq) xs
      !ks = map (linearForward attnWk) xs
      !vs = map (linearForward attnWv) xs
      !n  = length xs
      !dk = fromIntegral (vlen (head qs))
      !sf = 1 / sqrt dk

      -- weights[i] = softmax( [q_i·k_0, ...] * sf )
      !weights =
        [ softmax $ LA.scale sf $ LA.fromList [ qi LA.<.> kj | kj <- ks ]
        | qi <- qs
        ]

      -- attended[i] = Σ_j weights[i][j] * v_j
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

zeroAttentionGrads :: Int -> Int -> AttentionGrads
zeroAttentionGrads dK dModel = AttentionGrads
  { gWqParams = zeroLinearGrads dK dModel
  , gWkParams = zeroLinearGrads dK dModel
  , gWvParams = zeroLinearGrads dK dModel
  , gWoParams = zeroLinearGrads dModel dK
  }

addAttentionGrads :: AttentionGrads -> AttentionGrads -> AttentionGrads
addAttentionGrads a b = AttentionGrads
  { gWqParams = addLinear (gWqParams a) (gWqParams b)
  , gWkParams = addLinear (gWkParams a) (gWkParams b)
  , gWvParams = addLinear (gWvParams a) (gWvParams b)
  , gWoParams = addLinear (gWoParams a) (gWoParams b)
  }

attentionBackward
  :: AttentionParams
  -> [Vec]             -- inputs xs
  -> AttentionCache
  -> [Vec]             -- upstream gradients dOuts (wrt attention outputs)
  -> (AttentionGrads, [Vec])  -- (grads, dXs)
attentionBackward AttentionParams{..} xs AttentionCache{..} dOuts =
  let !n  = length xs
      !dK = vlen (head acQ)
      !dk = fromIntegral dK
      !sf = 1 / sqrt dk

      -- Back through Wo for each token: out_i = Wo(attended_i)
      perTokWo :: [(LinearParams, Vec)] -- (gWo_i, dAttended_i)
      perTokWo =
        [ linearBackward attnWo (acAttended !! i) (dOuts !! i)
        | i <- [0 .. n - 1]
        ]

      !gWoTotal = foldl' addLinear (zeroLinearGrads (vlen (head dOuts)) dK) (map fst perTokWo)
      !dAttended = map snd perTokWo  -- n × dK

      -- attended_i = Σ_j alpha_i[j] * v_j
      -- dV_j = Σ_i alpha_i[j] * dAttended_i
      !dVs =
        [ foldl' (+) (zerosV dK)
            [ LA.scale ((acWeights !! i) `LA.atIndex` j) (dAttended !! i)
            | i <- [0 .. n - 1]
            ]
        | j <- [0 .. n - 1]
        ]

      -- dAlpha_i[j] = dAttended_i · v_j
      !dAlphas =
        [ LA.fromList
            [ (dAttended !! i) LA.<.> (acV !! j)
            | j <- [0 .. n - 1]
            ]
        | i <- [0 .. n - 1]
        ]

      -- alpha_i = softmax(scores_i), scores_i = sf * [q_i·k_j]
      -- dScores_i = softmaxBackward(alpha_i, dAlpha_i)
      -- dDot_i[j] = sf * dScores_i[j]
      !dDots =
        [ LA.scale sf (softmaxBackward (acWeights !! i) (dAlphas !! i))
        | i <- [0 .. n - 1]
        ] -- each Vec length n

      -- dot_ij = q_i · k_j
      -- dQ_i = Σ_j dDot_ij * k_j
      -- dK_j = Σ_i dDot_ij * q_i
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

      -- Back through Q,K,V projections per token
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
-- Full transformer (1 layer), seq length = 2
--------------------------------------------------------------------------------

data TransformerParams = TransformerParams
  { tokEmbed   :: !Mat             -- p × d_model
  , posEmbed   :: !Mat             -- 2 × d_model
  , attnParams :: !AttentionParams
  , ln1        :: !LayerNormParams
  , ffnParams  :: !FFNParams
  , ln2        :: !LayerNormParams
  , unembed    :: !LinearParams    -- d_model → p
  }

data ForwardCache = ForwardCache
  { cEmbedded   :: ![Vec]              -- n × d_model
  , cAttnCache  :: !AttentionCache
  , cAttnOut    :: ![Vec]              -- n × d_model
  , cResidual1  :: ![Vec]
  , cLn1Out     :: ![Vec]
  , cLn1Cache   :: ![LayerNormCache]
  , cFfnOut     :: ![Vec]
  , cFfnCache   :: ![FFNCache]
  , cResidual2  :: ![Vec]
  , cLn2Out     :: ![Vec]
  , cLn2Cache   :: ![LayerNormCache]
  , cLogits     :: !Vec               -- p
  , cProbs      :: !Vec               -- p
  }

-- Inference-only forward (no cache)
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

-- Training forward (cache everything needed for exact backprop)
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
      ffnOut     = map fst ffnPairs
      ffnCache   = map snd ffnPairs

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
        , cAttnOut   = attnOut
        , cResidual1 = residual1
        , cLn1Out    = ln1Out
        , cLn1Cache  = ln1Cache
        , cFfnOut    = ffnOut
        , cFfnCache  = ffnCache
        , cResidual2 = residual2
        , cLn2Out    = ln2Out
        , cLn2Cache  = ln2Cache
        , cLogits    = logits
        , cProbs     = probs
        }
  in (probs, cache)

--------------------------------------------------------------------------------
-- Loss
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

--------------------------------------------------------------------------------
-- Backprop + gradients
--------------------------------------------------------------------------------

data Gradients = Gradients
  { gTokEmbed :: !Mat
  , gPosEmbed :: !Mat
  , gUnembed  :: !LinearParams
  , gFFN1     :: !LinearParams
  , gFFN2     :: !LinearParams
  , gAttnWq   :: !LinearParams
  , gAttnWk   :: !LinearParams
  , gAttnWv   :: !LinearParams
  , gAttnWo   :: !LinearParams
  , gLn1Gamma :: !Vec
  , gLn1Beta  :: !Vec
  , gLn2Gamma :: !Vec
  , gLn2Beta  :: !Vec
  }

zeroGradients :: Int -> Int -> Int -> Int -> Gradients
zeroGradients vocab dModel dFF dK = Gradients
  { gTokEmbed = zerosM vocab dModel
  , gPosEmbed = zerosM 2 dModel
  , gUnembed  = zeroLinearGrads vocab dModel
  , gFFN1     = zeroLinearGrads dFF dModel
  , gFFN2     = zeroLinearGrads dModel dFF
  , gAttnWq   = zeroLinearGrads dK dModel
  , gAttnWk   = zeroLinearGrads dK dModel
  , gAttnWv   = zeroLinearGrads dK dModel
  , gAttnWo   = zeroLinearGrads dModel dK
  , gLn1Gamma = zerosV dModel
  , gLn1Beta  = zerosV dModel
  , gLn2Gamma = zerosV dModel
  , gLn2Beta  = zerosV dModel
  }

backward
  :: TransformerParams
  -> ForwardCache
  -> Int -> Int -> Int
  -> Gradients
backward TransformerParams{..} cache tokA tokB target =
  let
    !dModel = vlen (lnGamma ln1)
    !vocab  = LA.rows tokEmbed
    !dFF    = vlen (linB (ffnLinear1 ffnParams))
    -- !dK     = vlen (linB (attnWq attnParams))
    !n      = length (cEmbedded cache)

    -- Output: CE loss on probs
    !dLogits = crossEntropyGradLogits (cProbs cache) target

    -- Unembed backward: logits = W_u * readout + b_u
    !readout = head (cLn2Out cache)
    (gUnemb, dReadout) = linearBackward unembed readout dLogits

    -- LN2 backward (per token). Only token 0 receives upstream gradient.
    !dLn2Outs = dReadout : replicate (n - 1) (zerosV dModel)

    ln2Backs :: [(Vec, Vec, Vec)] -- (dResidual2_i, dGamma_i, dBeta_i)
    ln2Backs =
      [ layerNormBackward ln2 (cLn2Cache cache !! i) (dLn2Outs !! i)
      | i <- [0 .. n - 1]
      ]

    !dResidual2 = map (\(dx,_,_) -> dx) ln2Backs
    !gLn2G      = foldl' (+) (zerosV dModel) (map (\(_,dg,_) -> dg) ln2Backs)
    !gLn2B      = foldl' (+) (zerosV dModel) (map (\(_,_,db) -> db) ln2Backs)

    -- residual2 = ln1Out + ffnOut
    !dLn1Out_from_res2 = dResidual2
    !dFfnOut           = dResidual2

    -- FFN backward per token
    ffnBacks :: [(LinearParams, LinearParams, Vec)] -- (g1_i, g2_i, dLn1Out_from_ffn_i)
    ffnBacks =
      [ ffnBackward ffnParams (cLn1Out cache !! i) (cFfnCache cache !! i) (dFfnOut !! i)
      | i <- [0 .. n - 1]
      ]

    !gFFN1Total = foldl' addLinear (zeroLinearGrads dFF dModel) (map (\(g1,_,_) -> g1) ffnBacks)
    !gFFN2Total = foldl' addLinear (zeroLinearGrads dModel dFF) (map (\(_,g2,_) -> g2) ffnBacks)
    !dLn1Out_from_ffn = map (\(_,_,dx) -> dx) ffnBacks

    -- Combine gradients into LN1 output
    !dLn1Out =
      zipWith (+) dLn1Out_from_res2 dLn1Out_from_ffn

    -- LN1 backward per token
    ln1Backs :: [(Vec, Vec, Vec)] -- (dResidual1_i, dGamma_i, dBeta_i)
    ln1Backs =
      [ layerNormBackward ln1 (cLn1Cache cache !! i) (dLn1Out !! i)
      | i <- [0 .. n - 1]
      ]

    !dResidual1 = map (\(dx,_,_) -> dx) ln1Backs
    !gLn1G      = foldl' (+) (zerosV dModel) (map (\(_,dg,_) -> dg) ln1Backs)
    !gLn1B      = foldl' (+) (zerosV dModel) (map (\(_,_,db) -> db) ln1Backs)

    -- residual1 = embedded + attnOut
    !dEmbedded_from_res1 = dResidual1
    !dAttnOut            = dResidual1

    -- Attention backward
    attnC = cAttnCache cache
    xs    = cEmbedded cache
    (attnGrads, dEmbedded_from_attn) =
      attentionBackward attnParams xs attnC dAttnOut

    !dEmbedded = zipWith (+) dEmbedded_from_res1 dEmbedded_from_attn

    -- Scatter gradients into token/pos embeddings
    -- embedded[0] = tokEmbed[tokA] + posEmbed[0]
    -- embedded[1] = tokEmbed[tokB] + posEmbed[1]
    mkScatter :: Int -> Vec -> [((Int, Int), R)]
    mkScatter row v =
      [ ((row, j), v `LA.atIndex` j) | j <- [0 .. vlen v - 1] ]

    !gTok =
      LA.accum (zerosM vocab dModel) (+) (mkScatter tokA (dEmbedded !! 0))
      + LA.accum (zerosM vocab dModel) (+) (mkScatter tokB (dEmbedded !! 1))

    !gPos =
      LA.accum (zerosM 2 dModel) (+)
        ( mkScatter 0 (dEmbedded !! 0)
       ++ mkScatter 1 (dEmbedded !! 1)
        )

  in Gradients
      { gTokEmbed = gTok
      , gPosEmbed = gPos
      , gUnembed  = gUnemb
      , gFFN1     = gFFN1Total
      , gFFN2     = gFFN2Total
      , gAttnWq   = gWqParams attnGrads
      , gAttnWk   = gWkParams attnGrads
      , gAttnWv   = gWvParams attnGrads
      , gAttnWo   = gWoParams attnGrads
      , gLn1Gamma = gLn1G
      , gLn1Beta  = gLn1B
      , gLn2Gamma = gLn2G
      , gLn2Beta  = gLn2B
      }

--------------------------------------------------------------------------------
-- Updates (SGD + decoupled weight decay on weights/embeddings)
--------------------------------------------------------------------------------

updateMat :: R -> R -> Mat -> Mat -> Mat
updateMat lr wd p g = p - LA.scale lr (g + LA.scale wd p)

-- Apply weight decay to W only; no decay on bias.
updateLinear :: R -> R -> LinearParams -> LinearParams -> LinearParams
updateLinear lr wd p g = LinearParams
  { linW = linW p - LA.scale lr (linW g + LA.scale wd (linW p))
  , linB = linB p - LA.scale lr (linB g)
  }

updateLayerNorm :: R -> LayerNormParams -> (Vec, Vec) -> LayerNormParams
updateLayerNorm lr LayerNormParams{..} (dG, dB) = LayerNormParams
  { lnGamma = lnGamma - LA.scale lr dG
  , lnBeta  = lnBeta  - LA.scale lr dB
  }

applyGradients :: R -> R -> TransformerParams -> Gradients -> TransformerParams
applyGradients lr wd p g = p
  { tokEmbed = updateMat lr wd (tokEmbed p) (gTokEmbed g)
  , posEmbed = updateMat lr wd (posEmbed p) (gPosEmbed g)

  , unembed  = updateLinear lr wd (unembed p) (gUnembed g)

  , ln1      = updateLayerNorm lr (ln1 p) (gLn1Gamma g, gLn1Beta g)
  , ln2      = updateLayerNorm lr (ln2 p) (gLn2Gamma g, gLn2Beta g)

  , ffnParams = (ffnParams p)
      { ffnLinear1 = updateLinear lr wd (ffnLinear1 (ffnParams p)) (gFFN1 g)
      , ffnLinear2 = updateLinear lr wd (ffnLinear2 (ffnParams p)) (gFFN2 g)
      }

  , attnParams = (attnParams p)
      { attnWq = updateLinear lr wd (attnWq (attnParams p)) (gAttnWq g)
      , attnWk = updateLinear lr wd (attnWk (attnParams p)) (gAttnWk g)
      , attnWv = updateLinear lr wd (attnWv (attnParams p)) (gAttnWv g)
      , attnWo = updateLinear lr wd (attnWo (attnParams p)) (gAttnWo g)
      }
  }

--------------------------------------------------------------------------------
-- Init (Xavier) using hmatrix uniformSample
--------------------------------------------------------------------------------

xavierInit :: Int -> Int -> Int -> (Mat, Int)
xavierInit r c seed =
  let bound = sqrt (6 / fromIntegral (r + c))
      m     = LA.uniformSample seed r (replicate c (-bound, bound))
  in (m, seed + 1)

initLinear :: Int -> Int -> Int -> (LinearParams, Int)
initLinear outDim inDim seed0 =
  let (w, seed1) = xavierInit outDim inDim seed0
  in (LinearParams w (zerosV outDim), seed1)

initTransformer :: Int -> Int -> Int -> Int -> Int -> (TransformerParams, Int)
initTransformer vocab dModel dFF dK seed0 =
  let (tokE,  s1) = xavierInit vocab dModel seed0
      (posE,  s2) = xavierInit 2     dModel s1

      (wq,    s3) = initLinear dK    dModel s2
      (wk,    s4) = initLinear dK    dModel s3
      (wv,    s5) = initLinear dK    dModel s4
      (wo,    s6) = initLinear dModel dK   s5

      (ff1,   s7) = initLinear dFF   dModel s6
      (ff2,   s8) = initLinear dModel dFF  s7

      (unemb, s9) = initLinear vocab dModel s8

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
-- Checkpoint save / load
--
-- Format: a single text file with all parameters serialized as flat doubles.
-- Header line: "CHECKPOINT <epoch> <seed> <vocab> <dModel> <dFF> <dK>"
-- Remaining lines: one double per line.
--
-- The order of serialization must match exactly between save and load.
--------------------------------------------------------------------------------

-- | Flatten a matrix row-major into a list of doubles.
matToList :: Mat -> [R]
matToList m = LA.toList (LA.flatten m)

-- | Rebuild a matrix from a flat list given (rows, cols).
listToMat :: Int -> Int -> [R] -> Mat
listToMat r c xs = (r LA.>< c) xs

-- | Flatten a vector into a list of doubles.
vecToList :: Vec -> [R]
vecToList = LA.toList

-- | Rebuild a vector from a list of doubles.
listToVec :: [R] -> Vec
listToVec = LA.fromList

-- | Take n elements from a list, returning (taken, rest).
takeN :: Int -> [R] -> ([R], [R])
takeN n xs = splitAt n xs

-- | Serialize LinearParams to a list of doubles.
serializeLinear :: LinearParams -> [R]
serializeLinear LinearParams{..} = matToList linW ++ vecToList linB

-- | Deserialize LinearParams given (outDim, inDim).
deserializeLinear :: Int -> Int -> [R] -> (LinearParams, [R])
deserializeLinear outDim inDim xs =
  let (wData, xs1) = takeN (outDim * inDim) xs
      (bData, xs2) = takeN outDim xs1
  in (LinearParams (listToMat outDim inDim wData) (listToVec bData), xs2)

-- | Serialize LayerNormParams.
serializeLN :: LayerNormParams -> [R]
serializeLN LayerNormParams{..} = vecToList lnGamma ++ vecToList lnBeta

-- | Deserialize LayerNormParams given dim.
deserializeLN :: Int -> [R] -> (LayerNormParams, [R])
deserializeLN d xs =
  let (gData, xs1) = takeN d xs
      (bData, xs2) = takeN d xs1
  in (LayerNormParams (listToVec gData) (listToVec bData), xs2)

-- | Serialize all TransformerParams to a flat list.
serializeParams :: TransformerParams -> [R]
serializeParams TransformerParams{..} = concat
  [ matToList tokEmbed
  , matToList posEmbed
  , serializeLinear (attnWq attnParams)
  , serializeLinear (attnWk attnParams)
  , serializeLinear (attnWv attnParams)
  , serializeLinear (attnWo attnParams)
  , serializeLN ln1
  , serializeLinear (ffnLinear1 ffnParams)
  , serializeLinear (ffnLinear2 ffnParams)
  , serializeLN ln2
  , serializeLinear unembed
  ]

-- | Deserialize TransformerParams from a flat list.
deserializeParams :: Int -> Int -> Int -> Int -> [R] -> TransformerParams
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
      (unemb,   _)    = deserializeLinear vocab  dModel xs10
  in TransformerParams
      { tokEmbed   = listToMat vocab dModel tokData
      , posEmbed   = listToMat 2 dModel posData
      , attnParams = AttentionParams wq wk wv wo
      , ln1        = ln1p
      , ffnParams  = FFNParams ff1 ff2
      , ln2        = ln2p
      , unembed    = unemb
      }

-- | Save a checkpoint to a file.
-- Stores epoch, RNG seed, architecture dims, and all weights.
saveCheckpoint :: FilePath -> TransformerParams -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
saveCheckpoint path params epoch seed vocab dModel dFF dK = do
  let header = unwords
        [ "CHECKPOINT", show epoch, show seed
        , show vocab, show dModel, show dFF, show dK
        ]
      doubles = serializeParams params
      contents = unlines (header : map show doubles)
  writeFile path contents
  printf "  [checkpoint] Saved epoch %d to %s (%d parameters)\n" epoch path (length doubles)

-- | Load a checkpoint from a file.
-- Returns (params, epoch, seed) or Nothing if the file doesn't exist.
loadCheckpoint :: FilePath -> IO (Maybe (TransformerParams, Int, Int))
loadCheckpoint path = do
  exists <- doesFileExist path
  if not exists
    then return Nothing
    else do
      contents <- readFile path
      length contents `seq` return ()  -- force full read to release handle
      let ls = lines contents
      case ls of
        [] -> return Nothing
        (headerLine : rest) ->
          case words headerLine of
            ["CHECKPOINT", epochS, seedS, vocabS, dModelS, dFFS, dKS] ->
              let epoch  = read epochS  :: Int
                  seed   = read seedS   :: Int
                  vocab  = read vocabS  :: Int
                  dModel = read dModelS :: Int
                  dFF    = read dFFS    :: Int
                  dK     = read dKS     :: Int
                  doubles = map read rest :: [R]
                  params = deserializeParams vocab dModel dFF dK doubles
              in return $ Just (params, epoch, seed)
            _ -> do
              putStrLn "  [checkpoint] Warning: invalid checkpoint header, ignoring."
              return Nothing

--------------------------------------------------------------------------------
-- Data + shuffle (proper Fisher–Yates)
--------------------------------------------------------------------------------

type Example = (Int, Int, Int)  -- (a, b, (a+b) mod p)

generateData :: Int -> [Example]
generateData p = [ (a, b, (a + b) `mod` p) | a <- [0 .. p - 1], b <- [0 .. p - 1] ]

-- Proper Fisher–Yates shuffle on a vector, returns a new seed.
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

--------------------------------------------------------------------------------
-- Train / eval
--------------------------------------------------------------------------------

predict :: TransformerParams -> Int -> Int -> Int
predict params a b =
  let probs = forwardInfer params a b
  in LA.maxIndex probs

evaluate :: TransformerParams -> [Example] -> Double
evaluate params examples =
  let correct = length $ filter (\(a,b,t) -> predict params a b == t) examples
  in fromIntegral correct / fromIntegral (length examples)

trainStep :: TransformerParams -> R -> R -> Example -> (TransformerParams, R)
trainStep !params lr wd (a, b, target) =
  let (!probs, !cache) = forwardTrain params a b
      !loss  = crossEntropyLoss probs target
      !grads = backward params cache a b target
      !params' = applyGradients lr wd params grads
  in (params', loss)

trainEpoch :: TransformerParams -> R -> R -> [Example] -> Int -> (TransformerParams, R, Int)
trainEpoch params lr wd examples seed =
  let (shuf, seed1) = shuffle seed examples
      (!params', !lossSum) = foldl' step (params, 0) shuf
      step (!p, !acc) ex =
        let (!p', !l) = trainStep p lr wd ex
        in (p', acc + l)
      avgLoss = lossSum / fromIntegral (length examples)
  in (params', avgLoss, seed1)

--------------------------------------------------------------------------------
-- Main
--------------------------------------------------------------------------------

main :: IO ()
main = do
  putStrLn "╔══════════════════════════════════════════════════════════════╗"
  putStrLn "║  Modular Arithmetic Transformer — Haskell + hmatrix          ║"
  putStrLn "║  Task: Learn (a + b) mod p                                   ║"
  putStrLn "╚══════════════════════════════════════════════════════════════╝"
  putStrLn ""

  -- Hyperparameters
  let p         = 53
      dModel    = 64
      dFF       = 128
      dK        = 32
      lr        = 0.005
      wd        = 0.05
      epochs    = 20000000
      trainFrac = 0.3
      seed0     = 42
      ckptPath  = "checkpoint.ckpt"
      ckptEvery = 1000  -- save every N epochs

  printf "Modulus p = %d\n" p
  printf "Model dim = %d, FFN dim = %d, Key dim = %d\n" dModel dFF dK
  printf "Learning rate = %.4f, Weight decay = %.4f\n" lr wd
  printf "Train fraction = %.0f%%\n\n" (trainFrac * 100 :: Double)

  let allData = generateData p
  printf "Total examples: %d (= %d × %d)\n" (length allData) p p

  let (trainData, testData, seed1) = splitData trainFrac seed0 allData
  printf "Training: %d examples, Test: %d examples\n\n" (length trainData) (length testData)

  -- Try to load an existing checkpoint
  mCkpt <- loadCheckpoint ckptPath
  let (initParams, startEpoch, seed2) = case mCkpt of
        Just (ckptParams, ckptEpoch, ckptSeed) ->
          (ckptParams, ckptEpoch + 1, ckptSeed)
        Nothing ->
          let (freshParams, s) = initTransformer p dModel dFF dK seed1
          in (freshParams, 1, s)

  case mCkpt of
    Just (_, ckptEpoch, _) ->
      printf "Resumed from checkpoint at epoch %d.\n\n" ckptEpoch
    Nothing ->
      putStrLn "No checkpoint found. Starting fresh (Xavier init).\n"

  putStrLn "Epoch | Train Loss | Train Acc | Test Acc"
  putStrLn "------+------------+-----------+---------"
  hFlush stdout

  let loop !epoch !params !seed
        | epoch > epochs = do
            putStrLn ""
            putStrLn "Training complete!"
            let finalTrainAcc = evaluate params trainData
                finalTestAcc  = evaluate params testData
            printf "Final — Train: %.1f%%, Test: %.1f%%\n"
              (finalTrainAcc * 100) (finalTestAcc * 100)
            -- Save final checkpoint (only if we actually trained)
            when (epoch > startEpoch) $
              saveCheckpoint ckptPath params (epoch - 1) seed p dModel dFF dK

        | otherwise = do
            let (!params', !avgLoss, !seed') = trainEpoch params lr wd trainData seed

            when (epoch == startEpoch || epoch `mod` 10 == 0) $ do
              let trainAcc = evaluate params' trainData
                  testAcc  = evaluate params' testData
              printf "%5d | %10.4f | %8.1f%% | %7.1f%%\n"
                epoch avgLoss (trainAcc * 100) (testAcc * 100)
              hFlush stdout

            -- Periodic checkpoint save
            when (epoch `mod` ckptEvery == 0) $
              saveCheckpoint ckptPath params' epoch seed' p dModel dFF dK

            loop (epoch + 1) params' seed'

  loop startEpoch initParams seed2
