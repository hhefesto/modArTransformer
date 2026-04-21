{-# OPTIONS --guardedness #-}
module ModArTransformer.Tensor where

open import Agda.Builtin.Float public
  using (Float; primFloatPlus; primFloatMinus; primFloatTimes; primFloatDiv;
         primFloatNegate; primFloatSqrt; primFloatExp; primFloatLog;
         primFloatLess; primFloatEquality; primShowFloat)

open import Data.Vec.Base as Vec using (Vec; []; _∷_; zipWith; map; foldr; replicate; tabulate; lookup)
open import Data.Fin      using (Fin; zero; suc)
open import Data.Nat      using (ℕ; zero; suc)
open import Data.Bool     using (Bool; true; false; if_then_else_)
open import Data.List     using (List; []; _∷_; foldl)
open import Data.Product  using (_×_; _,_; proj₁; proj₂)

-- ─── Base float operations ────────────────────────────────────────────────────

infixl 6 _f+_ _f-_
infixl 7 _f*_ _f/_

_f+_ : Float → Float → Float
_f+_ = primFloatPlus

_f-_ : Float → Float → Float
_f-_ = primFloatMinus

_f*_ : Float → Float → Float
_f*_ = primFloatTimes

_f/_ : Float → Float → Float
_f/_ = primFloatDiv

fneg : Float → Float
fneg = primFloatNegate

fsqrt : Float → Float
fsqrt = primFloatSqrt

fexp : Float → Float
fexp = primFloatExp

flog : Float → Float
flog = primFloatLog

_f<_ : Float → Float → Bool
_f<_ = primFloatLess

fzero : Float
fzero = 0.0

fone : Float
fone = 1.0

-- ─── Vec operations ───────────────────────────────────────────────────────────

ℝVec : ℕ → Set
ℝVec n = Vec Float n

vzero : {n : ℕ} → ℝVec n
vzero = replicate _ fzero

infixl 6 _v+_ _v-_

_v+_ : {n : ℕ} → ℝVec n → ℝVec n → ℝVec n
_v+_ = zipWith _f+_

_v-_ : {n : ℕ} → ℝVec n → ℝVec n → ℝVec n
_v-_ = zipWith _f-_

vscale : {n : ℕ} → Float → ℝVec n → ℝVec n
vscale s = map (s f*_)

vmap : {n : ℕ} → (Float → Float) → ℝVec n → ℝVec n
vmap = map

vsum : {n : ℕ} → ℝVec n → Float
vsum {zero}  []       = fzero
vsum {suc _} (x ∷ xs) = x f+ vsum xs

vdot : {n : ℕ} → ℝVec n → ℝVec n → Float
vdot u v = vsum (zipWith _f*_ u v)

vkonst : {n : ℕ} → Float → ℝVec n
vkonst c = replicate _ c

-- argmax: index of maximum element
vmaxIndex : {n : ℕ} → ℝVec (suc n) → Fin (suc n)
vmaxIndex {zero}  (x ∷ []) = zero
vmaxIndex {suc n} (x ∷ xs) =
  let imax = vmaxIndex xs
  in  if primFloatLess (lookup xs imax) x
        then zero
        else suc imax

vmaxElement : {n : ℕ} → ℝVec (suc n) → Float
vmaxElement {zero}  (x ∷ []) = x
vmaxElement {suc n} (x ∷ xs) with vmaxElement xs
... | m = if primFloatLess m x then x else m

-- one-hot vector: 1 at position i, 0 elsewhere
private
  finEq : {n : ℕ} → Fin n → Fin n → Bool
  finEq zero    zero    = true
  finEq zero    (suc _) = false
  finEq (suc _) zero    = false
  finEq (suc i) (suc j) = finEq i j

oneHot : {n : ℕ} → Fin n → ℝVec n
oneHot i = tabulate (λ j → if finEq i j then fone else fzero)

-- ─── Mat operations ───────────────────────────────────────────────────────────

-- Stored as rows: Mat m n has m rows, each of length n
ℝMat : ℕ → ℕ → Set
ℝMat m n = Vec (ℝVec n) m

mzero : {m n : ℕ} → ℝMat m n
mzero = replicate _ vzero

-- matrix-vector multiply: (m×n) ·ᵥ n → m
infixl 7 _#>_
_#>_ : {m n : ℕ} → ℝMat m n → ℝVec n → ℝVec m
M #> v = map (vdot v) M

-- outer product: (m-vec) ⊗ (n-vec) → m×n mat
outer : {m n : ℕ} → ℝVec m → ℝVec n → ℝMat m n
outer u v = map (λ ui → vscale ui v) u

-- matrix transpose: ℝMat m n → ℝMat n m
mtr : {m n : ℕ} → ℝMat m n → ℝMat n m
mtr {m} {zero}  _        = []
mtr {m} {suc n} rows     = map Vec.head rows ∷ mtr (map Vec.tail rows)
  where open import Data.Vec.Base as Vec using (head; tail)

-- elementwise matrix add
_m+_ : {m n : ℕ} → ℝMat m n → ℝMat m n → ℝMat m n
_m+_ = zipWith _v+_

-- scalar-matrix multiply
mscale : {m n : ℕ} → Float → ℝMat m n → ℝMat m n
mscale s = map (vscale s)

-- matrix row lookup
mrow : {m n : ℕ} → ℝMat m n → Fin m → ℝVec n
mrow M i = lookup M i

-- update row i of matrix M by adding delta
mAddRow : {m n : ℕ} → ℝMat m n → Fin m → ℝVec n → ℝMat m n
mAddRow (r ∷ rs) zero    d = (r v+ d) ∷ rs
mAddRow (r ∷ rs) (suc i) d = r ∷ mAddRow rs i d
