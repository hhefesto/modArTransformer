-- Primitives and combinators for SEQUENCE models in D (Dual AddFun).
--
-- The structural primitives (empty/zero/cons/append) are linear maps — their
-- pullbacks are the transposed re-arrangements.  The one new nonlinear
-- primitive is `softmaxMaskedD`, the n-ary, mask-parameterized generalization
-- of `softmax2D` (same Jacobian formula).  Everything else a sequence model
-- needs is Agda-level recursion over families of morphisms (`packD`,
-- `sumVecsD`, `sumFloatsD`) — pure composition, no new pullbacks.
{-# OPTIONS --guardedness #-}
module ModArTransformer.Cat.SeqPrim where

open import Data.Nat using (ℕ; suc; _+_)
open import Data.Fin using (Fin)
open import Data.Bool using (Bool; if_then_else_)
open import Data.Product using (_,_; proj₁; proj₂)
open import Data.List using (foldl)
open import Data.Vec.Base
  using (Vec; []; _∷_; head; tail; _++_; take; drop; tabulate; lookup;
         zipWith; allFin; toList)

open import ModArTransformer.Tensor
open import ModArTransformer.Cat.Objects
open import ModArTransformer.Cat.Additive
open import ModArTransformer.Cat.AddFun
open import ModArTransformer.Cat.Dual
open import ModArTransformer.Cat.D
open import ModArTransformer.Cat.NumCat using (addD)
open import ModArTransformer.Cat.VecPrim using (vaddD)
open import ModArTransformer.Cat.AdditiveTensor

private variable
  A : Set
  n m k a b : ℕ

-- ─── structural (linear) primitives ──────────────────────────────────────────

-- the empty vector (constant): pullback is zero.
emptyVecD : ⦃ Additive A ⦄ → D A (ℝVec 0)
emptyVecD = linearD (λ _ → []) (mkDual (mkAddFun (λ _ → zeroA)))

-- the constant-zero vector (seed for empty sums): pullback is zero.
zeroVecD : ⦃ Additive A ⦄ → D A (ℝVec m)
zeroVecD = linearD (λ _ → vzero) (mkDual (mkAddFun (λ _ → zeroA)))

-- the constant-zero scalar (seed for empty sums): pullback is zero.
zeroFloatD : ⦃ Additive A ⦄ → D A Float
zeroFloatD = linearD (λ _ → fzero) (mkDual (mkAddFun (λ _ → zeroA)))

-- cons (linear): pullback splits the cotangent into head and tail.
consD : D (Float × ℝVec n) (ℝVec (suc n))
consD = linearD (λ p → proj₁ p ∷ proj₂ p)
                (mkDual (mkAddFun (λ dy → head dy , tail dy)))

-- append (linear): pullback splits at the boundary.  This is the head-concat
-- of multi-head attention.
appendD : D (ℝVec a × ℝVec b) (ℝVec (a + b))
appendD {a = a} = linearD (λ p → proj₁ p ++ proj₂ p)
                          (mkDual (mkAddFun (λ dy → take a dy , drop a dy)))

-- ─── masked n-ary softmax ─────────────────────────────────────────────────────

-- Generalizes softmax2D to n coordinates with a participation mask: masked
-- outputs are exactly 0 and neither receive nor propagate cotangent (wᵢ = 0 in
-- the Jacobian dᵢ = wᵢ·(dwᵢ − Σⱼ wⱼ·dwⱼ)).  Max-stabilized over the allowed
-- coordinates (shift-invariant, like softmax2D).  With allow j = (j ≤ i) this
-- is row i of a CAUSAL attention softmax.  Precondition: at least one
-- coordinate is allowed (causal rows always allow j = 0).
softmaxMaskedD : (Fin n → Bool) → D (ℝVec n) (ℝVec n)
softmaxMaskedD {n} allow = mkD (λ x →
  let mx = foldl (λ acc j →
                    if allow j
                      then (let xj = lookup x j in if acc f< xj then xj else acc)
                      else acc)
                 (fneg 1.0e308) (toList (allFin n))
      es = tabulate (λ j → if allow j then fexp (lookup x j f- mx) else fzero)
      s  = vsum es
      w  = vmap (λ e → e f/ s) es
  in ( w
     , mkDual (mkAddFun (λ dw →
         let dot = vdot w dw
         in  zipWith (λ wi dwi → wi f* (dwi f- dot)) w dw)) ))

-- ─── combinators (recursion over morphism families; pure composition) ─────────

-- pack k scalar morphisms into one vector morphism.
packD : ⦃ Additive A ⦄ → Vec (D A Float) k → D A (ℝVec k)
packD []       = emptyVecD
packD (f ∷ fs) = consD ∘D (f ▵D packD fs)

-- sum a family of vector morphisms.
sumVecsD : ⦃ Additive A ⦄ → Vec (D A (ℝVec m)) k → D A (ℝVec m)
sumVecsD []           = zeroVecD
sumVecsD (f ∷ [])     = f
sumVecsD (f ∷ g ∷ gs) = vaddD ∘D (f ▵D sumVecsD (g ∷ gs))

-- sum a family of scalar morphisms.
sumFloatsD : ⦃ Additive A ⦄ → Vec (D A Float) k → D A Float
sumFloatsD []           = zeroFloatD
sumFloatsD (f ∷ [])     = f
sumFloatsD (f ∷ g ∷ gs) = addD ∘D (f ▵D sumFloatsD (g ∷ gs))
