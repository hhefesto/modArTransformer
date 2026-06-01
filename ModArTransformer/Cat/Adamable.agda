-- Generic AdamW over a parameter bundle, by structural recursion on the product
-- of tensor leaves.  The matrix-vs-vector split *is* the weight-decay policy:
-- `ℝMat` leaves get decoupled weight decay, `ℝVec` leaves (biases, LN γ/β) do
-- not — exactly matching the old hand-written `adamStep` (Adam.agda). Moments
-- (m,v) have the same shape as the parameters, so `AdamState A` just carries two
-- copies of `A` plus the timestep and bias-correction accumulators.
-- Per-tensor update math transcribes old Adam.agda:73-112.
{-# OPTIONS --guardedness #-}
module ModArTransformer.Cat.Adamable where

open import Data.Nat using (ℕ; suc)
open import Data.Product using (_,_; proj₁; proj₂)
open import Data.Vec.Base using (map; zipWith)

open import ModArTransformer.Tensor
open import ModArTransformer.Cat.Objects using (_×_)
open import ModArTransformer.Cat.Additive

private variable A B : Set

record AdamConfig : Set where
  constructor mkAdamCfg
  field adamB1 adamB2 adamEps adamWD : Float

-- adamStep1 cfg lrHat params grads (m , v) = (params' , (m' , v'))
record Adamable (A : Set) : Set where
  field adamStep1 : AdamConfig → Float → A → A → (A × A) → A × (A × A)
open Adamable ⦃ … ⦄ public

private
  mapMat  : {m n : ℕ} → (Float → Float) → ℝMat m n → ℝMat m n
  mapMat f = map (map f)
  mapMat2 : {m n : ℕ} → (Float → Float → Float) → ℝMat m n → ℝMat m n → ℝMat m n
  mapMat2 f = zipWith (zipWith f)

instance
  -- matrices: decoupled weight decay applied (param ← param − lr·step − lr·wd·param)
  Adamable-ℝMat : {m n : ℕ} → Adamable (ℝMat m n)
  Adamable-ℝMat = record { adamStep1 = λ cfg lr param g mv →
    let b1 = AdamConfig.adamB1 cfg ; b2 = AdamConfig.adamB2 cfg
        eps = AdamConfig.adamEps cfg ; wd = AdamConfig.adamWD cfg
        m = proj₁ mv ; v = proj₂ mv
        m'  = mscale b1 m m+ mscale (fone f- b1) g
        v'  = mscale b2 v m+ mscale (fone f- b2) (mapMat (λ x → x f* x) g)
        step = mapMat2 (λ mi vi → mi f/ (fsqrt vi f+ eps)) m' v'
        param' = mapMat2 (λ p s → p f- lr f* s f- lr f* wd f* p) param step
    in (param' , (m' , v')) }

  -- vectors: no weight decay (biases / LayerNorm parameters)
  Adamable-ℝVec : {n : ℕ} → Adamable (ℝVec n)
  Adamable-ℝVec = record { adamStep1 = λ cfg lr param g mv →
    let b1 = AdamConfig.adamB1 cfg ; b2 = AdamConfig.adamB2 cfg
        eps = AdamConfig.adamEps cfg
        m = proj₁ mv ; v = proj₂ mv
        m'  = vscale b1 m v+ vscale (fone f- b1) g
        v'  = vscale b2 v v+ vscale (fone f- b2) (zipWith _f*_ g g)
        step = zipWith (λ mi vi → mi f/ (fsqrt vi f+ eps)) m' v'
        param' = zipWith (λ p s → p f- lr f* s) param step
    in (param' , (m' , v')) }

  Adamable-× : ⦃ Adamable A ⦄ → ⦃ Adamable B ⦄ → Adamable (A × B)
  Adamable-× = record { adamStep1 = λ cfg lr param g mv →
    let (pa' , mva') = adamStep1 cfg lr (proj₁ param) (proj₁ g) (proj₁ (proj₁ mv) , proj₁ (proj₂ mv))
        (pb' , mvb') = adamStep1 cfg lr (proj₂ param) (proj₂ g) (proj₂ (proj₁ mv) , proj₂ (proj₂ mv))
    in ((pa' , pb') , ((proj₁ mva' , proj₁ mvb') , (proj₂ mva' , proj₂ mvb'))) }

-- ─── Optimizer state and the bias-corrected step (Adam.agda:136-198) ───────────

record AdamState (A : Set) : Set where
  constructor mkAdamSt
  field aT : ℕ ; aB1t aB2t : Float ; aM aV : A
open AdamState public

initAdam : ⦃ Additive A ⦄ → Float → Float → AdamState A
initAdam b1 b2 = mkAdamSt 0 b1 b2 zeroA zeroA

adamStep : ⦃ Adamable A ⦄ → Float → AdamConfig → A → A → AdamState A → A × AdamState A
adamStep lr cfg params grads st =
  let lrHat = lr f* fsqrt (fone f- aB2t st) f/ (fone f- aB1t st)
      (params' , mv') = adamStep1 cfg lrHat params grads (aM st , aV st)
      b1t' = aB1t st f* AdamConfig.adamB1 cfg
      b2t' = aB2t st f* AdamConfig.adamB2 cfg
  in (params' , mkAdamSt (suc (aT st)) b1t' b2t' (proj₁ mv') (proj₂ mv'))
