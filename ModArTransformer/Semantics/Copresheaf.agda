-- Copresheaves on a [0,1]-enriched category: the *semantic* category, where
-- meaning lives (Tai-Danae Bradley).
--
-- Bradley, "An Enriched Category Theory of Language" (2021), Def. 7/8:
--   The semantic category L̂ = [0,1]^L consists of [0,1]-copresheaves f : L → I.
--   For each object x, the representable copresheaf
--       hˣ := L(x, −)        hˣ(c) = π(c | x)
--   is the *meaning of the expression x*: the varying potential of all contexts
--   in which x is used.  The enriched Yoneda lemma states  L̂(hˣ, f) = f(x).
--
-- Logical structure on meanings (Bradley Thms 2,3 & Def. 12): products = AND,
-- coproducts = OR, internal hom = IMPLIES — all computed pointwise in I.
{-# OPTIONS --without-K #-}
module ModArTransformer.Semantics.Copresheaf where

open import ModArTransformer.Semantics.Interval
open import ModArTransformer.Semantics.Enriched

-- | A [0,1]-copresheaf on L: an I-valued function on objects.
Copresheaf : EnrichedCat → Set
Copresheaf L = Obj L → I

module _ {L : EnrichedCat} where

  -- | The representable copresheaf hˣ = L(x, −): the meaning of x.
  --   (Notation: よ is the Yoneda embedding.)
  よ : Obj L → Copresheaf L
  よ x = λ c → hom L x c

  -- Logical operations on meanings, pointwise.

  -- Conjunction (categorical product / weighted limit, unweighted): AND.
  infixl 6 _∧ᶜ_
  _∧ᶜ_ : Copresheaf L → Copresheaf L → Copresheaf L
  (f ∧ᶜ g) c = f c ⊓ᴵ g c

  -- Disjunction (categorical coproduct / weighted colimit, unweighted): OR.
  infixl 5 _∨ᶜ_
  _∨ᶜ_ : Copresheaf L → Copresheaf L → Copresheaf L
  (f ∨ᶜ g) c = f c ⊔ᴵ g c

  -- Implication (internal hom on copresheaves), pointwise truncated division.
  infixr 4 _⇒ᶜ_
  _⇒ᶜ_ : Copresheaf L → Copresheaf L → Copresheaf L
  (f ⇒ᶜ g) c = ⟦ f c , g c ⟧ᴵ
