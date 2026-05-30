-- Vector/matrix primitives as morphisms in D (Dual AddFun): each carries the
-- closed-form derivative (the transpose) of a reusable linear-algebra operation.
-- Neural layers (Linear, FFN, attention, …) are *compositions* of these — they
-- contain no backward code of their own; gradients come from the chain rule in D.
{-# OPTIONS --guardedness #-}
module ModArTransformer.Cat.VecPrim where

open import Data.Nat using (ℕ; suc)
open import Data.Fin using (Fin)
open import Data.Product using (_,_; proj₁; proj₂)
open import Data.Bool using (if_then_else_)
open import Data.Vec.Base using (zipWith; map; lookup)

open import ModArTransformer.Tensor
open import ModArTransformer.Cat.Objects
open import ModArTransformer.Cat.Additive
open import ModArTransformer.Cat.AddFun
open import ModArTransformer.Cat.Dual
open import ModArTransformer.Cat.D
open import ModArTransformer.Cat.NumCat using (logD; subD)
open import ModArTransformer.Cat.AdditiveTensor

private variable n m : ℕ

-- elementwise vector addition (linear): pullback copies the cotangent.
vaddD : D (ℝVec n × ℝVec n) (ℝVec n)
vaddD = linearD (λ p → proj₁ p v+ proj₂ p) (mkDual (mkAddFun (λ dy → dy , dy)))
  where open import Data.Product using (proj₁; proj₂)

-- matrix·vector (bilinear): forward M #> v ; transpose
--   d(Mv) ↦ (dM = dy ⊗ v ,  dv = Mᵀ #> dy).
matvecD : D (ℝMat m n × ℝVec n) (ℝVec m)
matvecD = mkD (λ p →
  let M = proj₁ p ; v = proj₂ p in
  ( M #> v
  , mkDual (mkAddFun (λ dy → outer dy v , mtr M #> dy)) ))
  where open import Data.Product using (proj₁; proj₂)

-- ReLU over a vector: derivative is the diagonal 0/1 mask of the input.
reluVecD : D (ℝVec n) (ℝVec n)
reluVecD = mkD (λ x →
  ( map (λ xi → if xi f< fzero then fzero else xi) x
  , mkDual (mkAddFun (λ dy →
      zipWith (λ xi dyi → if xi f< fzero then fzero else dyi) x dy)) ))

-- sum of a vector (linear): pullback broadcasts the scalar cotangent.
vsumD : D (ℝVec n) Float
vsumD = linearD vsum (mkDual (mkAddFun (λ dz → vkonst dz)))

-- ─── Cross-entropy loss, built COMPOSITIONALLY (gradient fully derived) ─────────
-- Instead of one fused softmax+CE primitive with a hand-written pullback, we
-- assemble the loss from elementary pieces so its gradient (softmax − eₜ) is
-- *derived* by the chain rule in D — the last hand-coded gradient in the loss
-- path is gone.  forward:  log Σⱼ exp(logitsⱼ) − logitsₜ  =  −log softmax(logits)ₜ.

-- elementwise exp (an activation primitive, like reluVecD): derivative is exp.
expVecD : D (ℝVec n) (ℝVec n)
expVecD = mkD (λ x →
  ( vmap fexp x
  , mkDual (mkAddFun (λ dy → zipWith (λ xi dyi → fexp xi f* dyi) x dy)) ))

-- project onto coordinate t (linear): pullback dz ↦ dz · eₜ.
selectD : Fin n → D (ℝVec n) Float
selectD t = linearD (λ x → lookup x t)
                    (mkDual (mkAddFun (λ dz → vscale dz (oneHot t))))

-- log-sum-exp, composed from exp / sum / log.  Its gradient w.r.t. the logits is
-- exactly softmax, obtained by the chain rule (logD·vsumD·expVecD pullbacks).
-- (Numerically naive — no max-subtraction; fine for this task's small logits.
--  A stable variant would subtract a detached max.)
logSumExpD : D (ℝVec (suc n)) Float
logSumExpD = logD ∘D (vsumD ∘D expVecD)

-- Cross-entropy of softmax(logits) against the one-hot target t.
crossEntropyAtD : Fin (suc n) → D (ℝVec (suc n)) Float
crossEntropyAtD t = subD ∘D (logSumExpD ▵D selectD t)

-- ─── Primitives for compositional attention ────────────────────────────────────

-- dot product (bilinear): forward u·v ; transpose dz ↦ (dz·v , dz·u).
vdotD : D (ℝVec n × ℝVec n) Float
vdotD = mkD (λ p →
  let u = proj₁ p ; v = proj₂ p in
  ( vdot u v
  , mkDual (mkAddFun (λ dz → vscale dz v , vscale dz u)) ))

-- scalar·vector (bilinear): forward s·v ; transpose dy ↦ (ds = dy·v , dv = s·dy).
scaleVecD : D (Float × ℝVec n) (ℝVec n)
scaleVecD = mkD (λ p →
  let s = proj₁ p ; v = proj₂ p in
  ( vscale s v
  , mkDual (mkAddFun (λ dy → vdot dy v , vscale s dy)) ))

-- 2-way softmax with its Jacobian: dᵢ = wᵢ·(dwᵢ − Σⱼ wⱼ·dwⱼ).
-- (The 1/√dK score scaling lives in the score computation, not here.)
softmax2D : D (Float × Float) (Float × Float)
softmax2D = mkD (λ p →
  let a  = proj₁ p ; b = proj₂ p
      mx = if a f< b then b else a
      ea = fexp (a f- mx) ; eb = fexp (b f- mx)
      s  = ea f+ eb
      w0 = ea f/ s ; w1 = eb f/ s
  in  ( (w0 , w1)
      , mkDual (mkAddFun (λ dw →
          let dw0 = proj₁ dw ; dw1 = proj₂ dw
              dot = w0 f* dw0 f+ w1 f* dw1
          in  ( w0 f* (dw0 f- dot) , w1 f* (dw1 f- dot) ))) ))
