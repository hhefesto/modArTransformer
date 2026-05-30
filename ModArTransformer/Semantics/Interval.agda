-- The unit interval I = [0,1] as Tai-Danae Bradley's enriching object.
--
-- Bradley, "An Enriched Category Theory of Language" (2021), Lemma 1:
--   The unit interval is a *closed commutative monoidal preorder*
--     (I, ≤, ⊗, 1)
--   with monoidal product ⊗ = ordinary multiplication, unit 1, and internal
--   hom [a,b] = min(b/a, 1) (truncated division). Its categorical product is
--   min (logical AND) and coproduct is max (logical OR).
--
-- We represent I by Float.  The intended invariant 0 ≤ x ≤ 1 is *documented*,
-- not enforced at the type level (felix-style "Raw": structure now, laws/bounds
-- as future hardening).  Probabilities produced by a softmax do live in [0,1].
{-# OPTIONS --without-K #-}
module ModArTransformer.Semantics.Interval where

open import Agda.Builtin.Float
  using ( Float; primFloatTimes; primFloatDiv; primFloatLess )
open import Data.Bool using (Bool; true; false; if_then_else_; not; T)

-- | A point of the unit interval (intended: 0 ≤ x ≤ 1).
I : Set
I = Float

0ᴵ : I
0ᴵ = 0.0

1ᴵ : I
1ᴵ = 1.0

-- ─── The preorder (I, ≤) ──────────────────────────────────────────────────────

-- a ≤ b  ⟺  ¬ (b < a).  A genuine proposition (via T), usable in law statements.
infix 4 _≤ᴵ_
_≤ᴵ_ : I → I → Set
a ≤ᴵ b = T (not (primFloatLess b a))

-- ─── The commutative monoid (I, ⊗, 1) ─────────────────────────────────────────

-- Monoidal product = multiplication; unit = 1.  This is the composition law of
-- the language category: π(z|y) ⊗ π(y|x) = π(z|x).
infixl 7 _⊗ᴵ_
_⊗ᴵ_ : I → I → I
_⊗ᴵ_ = primFloatTimes

-- ─── Categorical product / coproduct (AND / OR) ────────────────────────────────

fmin : Float → Float → Float
fmin a b = if primFloatLess a b then a else b

fmax : Float → Float → Float
fmax a b = if primFloatLess a b then b else a

-- Product of copresheaves (logical AND).
infixl 6 _⊓ᴵ_
_⊓ᴵ_ : I → I → I
_⊓ᴵ_ = fmin

-- Coproduct of copresheaves (logical OR).
infixl 5 _⊔ᴵ_
_⊔ᴵ_ : I → I → I
_⊔ᴵ_ = fmax

-- ─── Internal hom (IMPLIES) ────────────────────────────────────────────────────

-- [a,b] = min(b/a, 1), truncated division (Bradley, Lemma 1).  Models
-- context-sensitive implication on meanings.  Closure: a·b ≤ c ⟺ a ≤ [b,c].
infixr 4 ⟦_,_⟧ᴵ
⟦_,_⟧ᴵ : I → I → I
⟦ a , b ⟧ᴵ = fmin (primFloatDiv b a) 1ᴵ
