-- The magnitude of a category of texts enriched by a language model
-- (Tai-Danae Bradley, 2025).
--
-- For the finite tree of texts T a model generates from ⊥ (up to a cutoff),
-- enriched with similarities t^{d(x,y)} where d(x,y) = −ln π(y|x), the
-- magnitude — the canonical "effective size" invariant of an enriched
-- category — collapses to a closed form:
--
--   Mag(tM) = (t − 1) · Σ_x H_t(p_x)  +  |T(⊥)|
--
-- where x ranges over the prompts (internal vertices), p_x is the model's
-- next-token distribution at x, |T(⊥)| counts terminating outputs, and H_t is
-- the Tsallis t-entropy
--
--   H_t(p) = (1 − Σᵢ pᵢᵗ) / (t − 1),       lim_{t→1} H_t = Shannon entropy H.
--
-- So  d/dt Mag(tM) |_{t=1} = Σ_x H(p_x):  the slope of the magnitude function
-- at 1 is the model's total Shannon entropy over its prompt set.  A
-- deterministic model has Mag = |outputs| at every t; spread-out models are
-- "bigger".  This is the model-level evaluation invariant the Phase-5 probe
-- computes over a trained market model.
--
-- Spec only: computable over an explicit finite enumeration of prompt
-- distributions (the probe samples them); no claim is made here about which
-- enumeration is taken.
{-# OPTIONS --guardedness #-}
module ModArTransformer.Semantics.Magnitude where

open import Data.Nat using (ℕ)
open import Data.List using (List; foldr; map)
open import Data.Bool using (if_then_else_)

open import ModArTransformer.Tensor

-- aᵗ for a ≥ 0, with 0ᵗ = 0: via exp/log, guarded at zero (probabilities
-- only, so the negative case is out of domain).
fpow : Float → Float → Float
fpow a t = if a f< 1.0e-300 then fzero else fexp (t f* flog a)

-- Tsallis t-entropy of a finite distribution (t ≠ 1).
tsallis : {n : ℕ} → Float → ℝVec n → Float
tsallis t p = (fone f- vsum (vmap (λ pi → fpow pi t) p)) f/ (t f- fone)

-- Shannon entropy: the t → 1 limit of `tsallis`.
shannon : {n : ℕ} → ℝVec n → Float
shannon p =
  fneg (vsum (vmap (λ pi → if pi f< 1.0e-300 then fzero else pi f* flog pi) p))

private
  lsum : List Float → Float
  lsum = foldr _f+_ fzero

-- | Mag(tM) over an explicit finite enumeration: the next-token distributions
--   at every prompt (internal vertex of the text tree), plus the number of
--   terminating outputs |T(⊥)|.
magnitude : {n : ℕ} → Float → List (ℝVec n) → Float → Float
magnitude t prompts nTerm =
  (t f- fone) f* lsum (map (tsallis t) prompts) f+ nTerm

-- | The slope of the magnitude function at t = 1: Σ_x H(p_x), the model's
--   total Shannon entropy over the prompt set.
magnitudeSlopeAt1 : {n : ℕ} → List (ℝVec n) → Float
magnitudeSlopeAt1 prompts = lsum (map shannon prompts)
