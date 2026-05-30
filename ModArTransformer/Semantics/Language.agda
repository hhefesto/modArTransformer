-- The modular-arithmetic language as a [0,1]-enriched category, and its
-- ground-truth (deterministic) meaning.
--
-- The task: given a two-token context [a, b], the only correct continuation is
-- the token (a + b) mod n.  The ground truth is therefore a *deterministic*
-- language: π( (a+b) mod n | [a,b] ) = 1 and 0 on every other token.  In
-- Bradley's framework this is a [0,1]-enriched category whose representable
-- copresheaf at [a,b] is a Dirac copresheaf — and it is exactly the target the
-- transformer's learned copresheaf ⟦θ⟧ is trained to approximate.
--
-- Conventions match ModArTransformer.Data: tokens are `Fin (suc p)`, vocab size
-- is `suc p`, and the target is `fromℕ< (m%n<n (toℕ a + toℕ b) (suc p))`.
{-# OPTIONS --without-K #-}
module ModArTransformer.Semantics.Language where

open import Data.Nat using (ℕ; suc; _+_)
open import Data.Nat.DivMod using (m%n<n)
open import Data.Fin using (Fin; zero; suc; toℕ; fromℕ<)
open import Data.List using (List; []; _∷_; length)
open import Data.Bool using (Bool; true; false; _∧_; if_then_else_)
open import Data.Product using (_×_; _,_)

open import ModArTransformer.Semantics.Interval
open import ModArTransformer.Semantics.Enriched
open import ModArTransformer.Semantics.Copresheaf

-- The whole language is parameterized by p (so vocab size = suc p).
module Lang (p : ℕ) where

  n : ℕ
  n = suc p

  -- | Vocabulary token.
  Vocab : Set
  Vocab = Fin n

  -- | A length-2 context (the two operands).
  Context : Set
  Context = Vocab × Vocab

  -- target a b = (a + b) mod n, as a token.  (Matches ModArTransformer.Data.)
  target : Vocab → Vocab → Vocab
  target a b = fromℕ< (m%n<n (toℕ a + toℕ b) n)

  -- ─── Decidable token / expression equality (Bool) ────────────────────────────

  finEqBool : {m : ℕ} → Fin m → Fin m → Bool
  finEqBool zero    zero    = true
  finEqBool zero    (suc _) = false
  finEqBool (suc _) zero    = false
  finEqBool (suc i) (suc j) = finEqBool i j

  exprEqBool : List Vocab → List Vocab → Bool
  exprEqBool []       []       = true
  exprEqBool []       (_ ∷ _)  = false
  exprEqBool (_ ∷ _)  []       = false
  exprEqBool (x ∷ xs) (y ∷ ys) = finEqBool x y ∧ exprEqBool xs ys

  -- ─── Discrete vocabulary category and the ground-truth copresheaf ────────────

  -- VocabCat: objects are tokens, hom is the discrete (Kronecker) hom δ(i,j).
  VocabCat : EnrichedCat
  VocabCat = record
    { Obj = Vocab
    ; hom = λ i j → if finEqBool i j then 1ᴵ else 0ᴵ
    }

  -- | The ground-truth next-token meaning of a context [a,b]: the Dirac
  --   copresheaf concentrated on (a+b) mod n.  The semantic *target* of training.
  truth : Context → Copresheaf VocabCat
  truth (a , b) v = if finEqBool v (target a b) then 1ᴵ else 0ᴵ

  -- ─── The full deterministic language category L⋆ ─────────────────────────────

  -- Objects are expressions (token sequences).  The single modeled continuation
  -- of a 2-token context [a,b] is its append with (a+b) mod n; every object also
  -- extends to itself (identity).  hom = 1 on these, 0 otherwise.  Raw enriched
  -- category; `IdLaw` holds since hom xs xs = 1.
  Expr : Set
  Expr = List Vocab

  private
    isLen2 : Expr → Bool
    isLen2 (_ ∷ _ ∷ []) = true
    isLen2 _            = false

    step : Expr → Expr
    step (a ∷ b ∷ []) = a ∷ b ∷ target a b ∷ []
    step xs           = xs

  L⋆ : EnrichedCat
  L⋆ = record
    { Obj = Expr
    ; hom = λ xs ys →
        if exprEqBool ys xs then 1ᴵ
        else if (isLen2 xs ∧ exprEqBool ys (step xs)) then 1ᴵ
        else 0ᴵ
    }
