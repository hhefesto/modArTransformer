{-# OPTIONS --guardedness #-}
module ModArTransformer.Optimizer.Adam where

open import Agda.Builtin.Float using (Float; primNatToFloat)
open import Data.Nat     using (ℕ; suc)
open import Data.Product using (_×_; _,_)
open import ModArTransformer.Tensor
open import ModArTransformer.Layers.Linear
open import ModArTransformer.Layers.LayerNorm
open import ModArTransformer.Layers.FFN
open import ModArTransformer.Layers.Attention
open import ModArTransformer.Layers.Transformer

-- ─── Adam moments ─────────────────────────────────────────────────────────────
-- Matches Main.hs:576-601.

record MatMoments (m n : ℕ) : Set where
  constructor mkMatM
  field mmM mmV : ℝMat m n

record VecMoments (n : ℕ) : Set where
  constructor mkVecM
  field vmM vmV : ℝVec n

record LinearMoments (out inp : ℕ) : Set where
  constructor mkLinM
  field lmW : MatMoments out inp
        lmB : VecMoments out

record LNMoments (n : ℕ) : Set where
  constructor mkLNM
  field lnmGamma lnmBeta : VecMoments n

record FFNMoments (dModel dFF : ℕ) : Set where
  constructor mkFFNM
  field ffnM1 : LinearMoments dFF   dModel   -- linear1: dModel → dFF
        ffnM2 : LinearMoments dModel dFF     -- linear2: dFF    → dModel

record AttnMoments (dModel dK : ℕ) : Set where
  constructor mkAttnM
  field aqM akM avM : LinearMoments dK    dModel   -- project dModel → dK
        aoM         : LinearMoments dModel dK       -- project dK    → dModel

record AdamState (p dModel dFF dK : ℕ) : Set where
  constructor mkAdam
  field
    adamT    : ℕ         -- timestep
    adamB1t  : Float     -- β1^t (accumulated)
    adamB2t  : Float     -- β2^t
    -- moments for every parameter group
    mTokEmb  : MatMoments (suc p) dModel
    mPosEmb  : MatMoments 2       dModel
    mAttn    : AttnMoments dModel dK
    mLn1     : LNMoments dModel
    mFFN1    : LinearMoments dFF  dModel
    mFFN2    : LinearMoments dModel dFF
    mLn2     : LNMoments dModel
    mUnembed : LinearMoments (suc p) dModel

-- ─── AdamConfig ───────────────────────────────────────────────────────────────
record AdamConfig : Set where
  constructor mkAdamCfg
  field
    adamB1  : Float   -- 0.9
    adamB2  : Float   -- 0.999
    adamEps : Float   -- 1e-8
    adamWD  : Float   -- weight decay

-- ─── Per-tensor Adam update ───────────────────────────────────────────────────
-- Matches Main.hs:663-686.

-- Matrix update with weight decay
adamMat : {m n : ℕ} → AdamConfig → Float → ℝMat m n
        → MatMoments m n → ℝMat m n
        → ℝMat m n × MatMoments m n
adamMat cfg lr param (mkMatM m v) g =
  let b1    = AdamConfig.adamB1 cfg
      b2    = AdamConfig.adamB2 cfg
      eps   = AdamConfig.adamEps cfg
      wd    = AdamConfig.adamWD  cfg
      m'    = mscale b1 m m+ mscale (fone f- b1) g
      -- elementwise square of g
      g2    = mapMat (λ x → x f* x) g
      v'    = mscale b2 v m+ mscale (fone f- b2) g2
      -- bias-corrected (done at call site via b1t/b2t)
      step  = mapMat2 (λ mi vi → mi f/ (fsqrt vi f+ eps)) m' v'
      param' = mapMat2 (λ p s → p f- lr f* s f- lr f* wd f* p) param step
  in  (param' , mkMatM m' v')
  where
    mapMat  : {m n : ℕ} → (Float → Float) → ℝMat m n → ℝMat m n
    mapMat f = Data.Vec.Base.map (Data.Vec.Base.map f)
      where open import Data.Vec.Base
    mapMat2 : {m n : ℕ} → (Float → Float → Float)
            → ℝMat m n → ℝMat m n → ℝMat m n
    mapMat2 f = Data.Vec.Base.zipWith (Data.Vec.Base.zipWith f)
      where open import Data.Vec.Base

-- Vector update without weight decay (biases, LN params)
adamVec : {n : ℕ} → AdamConfig → Float
        → ℝVec n → VecMoments n → ℝVec n
        → ℝVec n × VecMoments n
adamVec cfg lr param (mkVecM m v) g =
  let b1    = AdamConfig.adamB1 cfg
      b2    = AdamConfig.adamB2 cfg
      eps   = AdamConfig.adamEps cfg
      m'    = vscale b1 m v+ vscale (fone f- b1) g
      g2    = zipWith _f*_ g g
      v'    = vscale b2 v v+ vscale (fone f- b2) g2
      step  = zipWith (λ mi vi → mi f/ (fsqrt vi f+ eps)) m' v'
      param' = zipWith (λ p s → p f- lr f* s) param step
  in  (param' , mkVecM m' v')
  where open import Data.Vec.Base using (zipWith)

-- Linear layer update
adamLinear : {out inp : ℕ} → AdamConfig → Float
           → LinearParams out inp → LinearMoments out inp → LinearParams out inp
           → LinearParams out inp × LinearMoments out inp
adamLinear cfg lr p (mkLinM mW mB) g =
  let (w' , mW') = adamMat cfg lr (linW p) mW (linW g)
      (b' , mB') = adamVec cfg lr (linB p) mB (linB g)
  in  (mkLinear w' b' , mkLinM mW' mB')

-- Zero-initialized moments
zeroMatMoments : {m n : ℕ} → MatMoments m n
zeroMatMoments = mkMatM mzero mzero

zeroVecMoments : {n : ℕ} → VecMoments n
zeroVecMoments = mkVecM vzero vzero

zeroLinMoments : {out inp : ℕ} → LinearMoments out inp
zeroLinMoments = mkLinM zeroMatMoments zeroVecMoments

zeroLNMoments : {n : ℕ} → LNMoments n
zeroLNMoments = mkLNM zeroVecMoments zeroVecMoments

initAdamState : {p dModel dFF dK : ℕ} → Float → Float → AdamState p dModel dFF dK
initAdamState b1 b2 = mkAdam 0 b1 b2
  zeroMatMoments zeroMatMoments
  (mkAttnM zeroLinMoments zeroLinMoments zeroLinMoments zeroLinMoments)
  zeroLNMoments
  zeroLinMoments zeroLinMoments
  zeroLNMoments
  zeroLinMoments

-- ─── Full parameter update step (Main.hs:695-758) ─────────────────────────────
adamStep : {p dModel dFF dK : ℕ}
         → Float → AdamConfig → Float → Float
         → TransformerParams (suc p) dModel dFF dK
         → TransformerParams (suc p) dModel dFF dK  -- gradients
         → AdamState p dModel dFF dK
         → TransformerParams (suc p) dModel dFF dK × AdamState p dModel dFF dK
adamStep lr cfg b1t b2t params grads (mkAdam t _ _ mTE mPE mAt mL1 mF1 mF2 mL2 mUn) =
  let -- bias-correction factors baked into lr
      lrHat = lr f* fsqrt (fone f- b2t) f/ (fone f- b1t)

      -- token embed (matrix, with WD)
      (te' , mTE') = adamMat cfg lrHat (tokEmbed params) mTE (tokEmbed grads)
      -- pos embed (matrix, with WD)
      (pe' , mPE') = adamMat cfg lrHat (posEmbed params) mPE (posEmbed grads)

      -- attention weights
      (wq' , mWq') = adamLinear cfg lrHat (attnWq (attnP params)) (AttnMoments.aqM mAt) (attnWq (attnP grads))
      (wk' , mWk') = adamLinear cfg lrHat (attnWk (attnP params)) (AttnMoments.akM mAt) (attnWk (attnP grads))
      (wv' , mWv') = adamLinear cfg lrHat (attnWv (attnP params)) (AttnMoments.avM mAt) (attnWv (attnP grads))
      (wo' , mWo') = adamLinear cfg lrHat (attnWo (attnP params)) (AttnMoments.aoM mAt) (attnWo (attnP grads))

      -- LN1
      (g1' , mg1') = adamVec cfg lrHat (lnGamma (ln1P params)) (LNMoments.lnmGamma mL1) (lnGamma (ln1P grads))
      (b1'x , mb1') = adamVec cfg lrHat (lnBeta  (ln1P params)) (LNMoments.lnmBeta mL1)  (lnBeta  (ln1P grads))

      -- FFN
      (f1'  , mF1') = adamLinear cfg lrHat (ffnLinear1 (ffnP params)) mF1 (ffnLinear1 (ffnP grads))
      (f2'  , mF2') = adamLinear cfg lrHat (ffnLinear2 (ffnP params)) mF2 (ffnLinear2 (ffnP grads))

      -- LN2
      (g2' , mg2') = adamVec cfg lrHat (lnGamma (ln2P params)) (LNMoments.lnmGamma mL2) (lnGamma (ln2P grads))
      (b2'x , mb2') = adamVec cfg lrHat (lnBeta  (ln2P params)) (LNMoments.lnmBeta mL2)  (lnBeta  (ln2P grads))

      -- unembed
      (un' , mUn') = adamLinear cfg lrHat (unembed params) mUn (unembed grads)

      params' = mkTransformer te' pe'
                  (mkAttn wq' wk' wv' wo')
                  (mkLN g1' b1'x)
                  (mkFFN f1' f2')
                  (mkLN g2' b2'x)
                  un'

      b1t' = b1t f* AdamConfig.adamB1 cfg
      b2t' = b2t f* AdamConfig.adamB2 cfg
      adam' = mkAdam (suc t) b1t' b2t' mTE' mPE'
                (mkAttnM mWq' mWk' mWv' mWo')
                (mkLNM mg1' mb1')
                mF1' mF2'
                (mkLNM mg2' mb2')
                mUn'

  in  (params' , adam')
