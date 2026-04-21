{-# OPTIONS --guardedness #-}
module ModArTransformer.Layers.Attention where

open import Agda.Builtin.Float using (Float; primNatToFloat)
open import Data.Nat     using (ℕ; suc)
open import Data.Fin     using (Fin; zero; suc)
open import Data.Vec.Base using (Vec; _∷_; []; lookup; map; zipWith; replicate; tabulate)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.Bool    using (Bool; true; false; if_then_else_)
open import ModArTransformer.Tensor
open import ModArTransformer.Additive
open import ModArTransformer.AD.AddFun
open import ModArTransformer.AD.Dual
open import ModArTransformer.AD.Core
open import ModArTransformer.Layers.Linear

-- ─── Single-head self-attention for seqLen = 2 (Main.hs:193-319) ──────────────
-- Params: Wq, Wk, Wv, Wo each in ℝMat dK dModel.
-- Forward: Q,K,V = linear projections; scores = scaled dot-product; out = Wo(sum)

record AttentionParams (dModel dK : ℕ) : Set where
  constructor mkAttn
  field
    attnWq attnWk attnWv : LinearParams dK dModel   -- project dModel → dK
    attnWo               : LinearParams dModel dK   -- project dK → dModel

open AttentionParams public

zeroAttn : {dModel dK : ℕ} → AttentionParams dModel dK
zeroAttn = mkAttn zeroLinear zeroLinear zeroLinear zeroLinear

addAttn : {dModel dK : ℕ}
        → AttentionParams dModel dK → AttentionParams dModel dK → AttentionParams dModel dK
addAttn p q = mkAttn
  (addLinear (attnWq p) (attnWq q))
  (addLinear (attnWk p) (attnWk q))
  (addLinear (attnWv p) (attnWv q))
  (addLinear (attnWo p) (attnWo q))

scaleAttn : {dModel dK : ℕ} → Float → AttentionParams dModel dK → AttentionParams dModel dK
scaleAttn s p = mkAttn
  (scaleLinear s (attnWq p)) (scaleLinear s (attnWk p))
  (scaleLinear s (attnWv p)) (scaleLinear s (attnWo p))

instance
  additiveAttn : {dModel dK : ℕ} → Additive (AttentionParams dModel dK)
  additiveAttn = record { zero = zeroAttn ; _⊕_ = addAttn }

-- ─── Softmax (single row) ────────────────────────────────────────────────────
softmax2 : Float × Float → Float × Float
softmax2 (a , b) =
  let m = if a f< b then b else a
      ea = fexp (a f- m)
      eb = fexp (b f- m)
      s  = ea f+ eb
  in  (ea f/ s , eb f/ s)

-- ─── Differentiable attention for seqLen = 2 ────────────────────────────────
-- Input:  (params, (x0, x1)) : AttentionParams × (ℝVec dModel × ℝVec dModel)
-- Output: (y0, y1) : ℝVec dModel × ℝVec dModel
-- Pullback matches Main.hs:241-319 for n=2.

private
  k = Dual AddFun

attnD : {dModel dK : ℕ}
      → D k (AttentionParams dModel dK × (ℝVec dModel × ℝVec dModel))
            (ℝVec dModel × ℝVec dModel)
attnD {dModel} {dK} = mkD λ (p , (x0 , x1)) →
  let scale_dk = fone f/ fsqrt (primNatToFloat dK)

      -- Q, K, V projections (Main.hs:208-232)
      q0 = linW (attnWq p) #> x0 v+ linB (attnWq p)
      q1 = linW (attnWq p) #> x1 v+ linB (attnWq p)
      k0 = linW (attnWk p) #> x0 v+ linB (attnWk p)
      k1 = linW (attnWk p) #> x1 v+ linB (attnWk p)
      v0 = linW (attnWv p) #> x0 v+ linB (attnWv p)
      v1 = linW (attnWv p) #> x1 v+ linB (attnWv p)

      -- Attention scores (scaled dot-product)
      dot00 = vdot q0 k0 f* scale_dk
      dot01 = vdot q0 k1 f* scale_dk
      dot10 = vdot q1 k0 f* scale_dk
      dot11 = vdot q1 k1 f* scale_dk

      -- Softmax per row
      (w00 , w01) = softmax2 (dot00 , dot01)
      (w10 , w11) = softmax2 (dot10 , dot11)

      -- Weighted sum of values
      a0 = vscale w00 v0 v+ vscale w01 v1
      a1 = vscale w10 v0 v+ vscale w11 v1

      -- Output projection via Wo
      y0 = linW (attnWo p) #> a0 v+ linB (attnWo p)
      y1 = linW (attnWo p) #> a1 v+ linB (attnWo p)

      pb (dY0 , dY1) =
        -- backward through Wo
        let dA0    = mtr (linW (attnWo p)) #> dY0
            dA1    = mtr (linW (attnWo p)) #> dY1
            dWo    = outer dY0 a0 m+ outer dY1 a1
            dBo    = dY0 v+ dY1

            -- backward through weighted sum of V
            dW00   = vdot dA0 v0
            dW01   = vdot dA0 v1
            dW10   = vdot dA1 v0
            dW11   = vdot dA1 v1
            dV0    = vscale w00 dA0 v+ vscale w10 dA1
            dV1    = vscale w01 dA0 v+ vscale w11 dA1

            -- backward through softmax (Main.hs:278-281)
            smBwd : Float × Float → Float × Float → Float × Float
            smBwd (w0 , w1) (da0 , da1) =
              let dot_w_da = w0 f* da0 f+ w1 f* da1
              in  (w0 f* (da0 f- dot_w_da) f* scale_dk ,
                   w1 f* (da1 f- dot_w_da) f* scale_dk)
            (dDot00 , dDot01) = smBwd (w00 , w01) (dW00 , dW01)
            (dDot10 , dDot11) = smBwd (w10 , w11) (dW10 , dW11)

            -- backward through scaled dot-product (Main.hs:283-297)
            dQ0  = vscale dDot00 k0 v+ vscale dDot10 k0
            dQ1  = vscale dDot01 k1 v+ vscale dDot11 k1
            dK0  = vscale dDot00 q0 v+ vscale dDot10 q1
            dK1  = vscale dDot01 q0 v+ vscale dDot11 q1

            -- backward through Q, K, V projections (Main.hs:299-311)
            dWq  = outer dQ0 x0 m+ outer dQ1 x1
            dBq  = dQ0 v+ dQ1
            dWk  = outer dK0 x0 m+ outer dK1 x1
            dBk  = dK0 v+ dK1
            dWv  = outer dV0 x0 m+ outer dV1 x1
            dBv  = dV0 v+ dV1

            -- backward to inputs (Main.hs:307-311)
            dX0q = mtr (linW (attnWq p)) #> dQ0
            dX0k = mtr (linW (attnWk p)) #> dK0
            dX0v = mtr (linW (attnWv p)) #> dV0
            dX1q = mtr (linW (attnWq p)) #> dQ1
            dX1k = mtr (linW (attnWk p)) #> dK1
            dX1v = mtr (linW (attnWv p)) #> dV1
            dX0  = dX0q v+ dX0k v+ dX0v
            dX1  = dX1q v+ dX1k v+ dX1v

            dP   = mkAttn (mkLinear dWq dBq)
                          (mkLinear dWk dBk)
                          (mkLinear dWv dBv)
                          (mkLinear dWo dBo)
        in  (dP , (dX0 , dX1))

  in  ((y0 , y1) , mkDual (mkAddFun pb))
