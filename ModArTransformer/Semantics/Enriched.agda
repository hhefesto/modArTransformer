-- A category enriched over the unit interval I  (Tai-Danae Bradley).
--
-- Bradley, "An Enriched Category Theory of Language" (2021), Definition 4:
--   The syntax category L is enriched over [0,1]; objects are expressions and
--   hom-objects are conditional probabilities
--       L(x, y) := π(y | x)
--   (the probability that y extends x).  It is a category because
--       π(x|x) = 1                       (identity)
--       π(z|y) · π(y|x) ≤ π(z|x)         (composition / chain rule)
--
-- Following felix's Raw/Laws split, `EnrichedCat` is the *raw* structure
-- (objects + hom).  The two enrichment laws are recorded as predicates
-- (`IdLaw`, `CompLaw`) to be discharged later — they are not yet proved for the
-- Float-valued model (future hardening), but the ground-truth category below
-- satisfies them by construction.
{-# OPTIONS --without-K #-}
module ModArTransformer.Semantics.Enriched where

open import ModArTransformer.Semantics.Interval

-- | A raw [0,1]-enriched category: a set of objects and a hom assigning, to
--   each ordered pair, a point of the unit interval (a conditional probability).
record EnrichedCat : Set₁ where
  field
    Obj : Set
    hom : Obj → Obj → I

open EnrichedCat public

-- ─── Enrichment laws (Bradley Def. 4), as obligations ──────────────────────────

-- Identity: 1 ≤ L(x,x), i.e. π(x|x) = 1.
IdLaw : EnrichedCat → Set
IdLaw L = (x : Obj L) → 1ᴵ ≤ᴵ hom L x x

-- Composition: L(y,z) ⊗ L(x,y) ≤ L(x,z), i.e. π(z|y)·π(y|x) ≤ π(z|x).
CompLaw : EnrichedCat → Set
CompLaw L = (x y z : Obj L) → (hom L y z ⊗ᴵ hom L x y) ≤ᴵ hom L x z
