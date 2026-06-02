-- Exact MPS / tensor-train representation of modular addition.
--
-- The dense amplitude tensor is
--
--   ψ(a,b,c) = 1/n   if c = a + b mod n
--            = 0     otherwise.
--
-- A three-site MPS with bond dimension χ = n represents it exactly by carrying
-- the partial modular sum in the bond index:
--
--   A(a,i)   = 1/n   if i = a
--   B(i,b,j) = 1     if j = i + b mod n
--   C(j,c)   = 1     if c = j
--
-- Then Σᵢⱼ A(a,i) B(i,b,j) C(j,c) = ψ(a,b,c).  This is the many-body/finite-state
-- automaton view of the same Bradley copresheaf π(- | [a,b]).
{-# OPTIONS --guardedness #-}
module ModArTransformer.Semantics.MPS where

open import Agda.Builtin.Float using (Float)
open import Data.Nat as Nat using (ℕ)
open import Data.Fin as Fin using (Fin)
open import Data.Bool using (if_then_else_)
open import Data.Product using (_,_)

open import ModArTransformer.Tensor
open import ModArTransformer.Semantics.TensorNetwork

CoreA : ℕ → Set
CoreA n = Fin n → Fin n → Float

CoreB : ℕ → Set
CoreB n = Fin n → Fin n → Fin n → Float

CoreC : ℕ → Set
CoreC n = Fin n → Fin n → Float

private
  fAbs : Float → Float
  fAbs x = if x f< fzero then fneg x else x

  fMax : Float → Float → Float
  fMax x y = if x f< y then y else x

maxFin : (n : ℕ) → (Fin n → Float) → Float
maxFin Nat.zero    _ = fzero
maxFin (Nat.suc n) f = fMax (f Fin.zero) (maxFin n (λ i → f (Fin.suc i)))

module ExactMPS (p : ℕ) where
  open ManyBody p public using (n; nF; Vocab; Context; target; finEqBool; amplitude)

  bondDim : ℕ
  bondDim = n

  coreA : CoreA n
  coreA a i = if finEqBool i a then fone f/ nF else fzero

  coreB : CoreB n
  coreB i b j = if finEqBool j (target i b) then fone else fzero

  coreC : CoreC n
  coreC j c = if finEqBool c j then fone else fzero

  -- Full tensor-network contraction.  Useful as the specification of the MPS,
  -- but diagnostics avoid using it over every a,b,c because that would add two
  -- extra O(n) sums inside an O(n^3) traversal.
  evalMPS : Vocab → Vocab → Vocab → Float
  evalMPS a b c =
    sumFin n (λ i →
      sumFin n (λ j →
        coreA a i f* coreB i b j f* coreC j c))

  -- Collapsed deterministic path through the same cores: i = a and
  -- j = target a b.  This is the closed-form value of `evalMPS` for this exact
  -- automaton-style MPS.
  closedMPS : Vocab → Vocab → Vocab → Float
  closedMPS a b c = coreA a a f* coreB a b (target a b) f* coreC (target a b) c

  closedMPSSquared : Vocab → Vocab → Vocab → Float
  closedMPSSquared a b c = closedMPS a b c f* closedMPS a b c

  closedContextNorm : Context → Float
  closedContextNorm (a , b) = sumFin n (λ c → closedMPSSquared a b c)

  closedConditional : Context → Vocab → Float
  closedConditional (a , b) c = closedMPSSquared a b c f/ closedContextNorm (a , b)

  maxClosedAmplitudeError : Float
  maxClosedAmplitudeError =
    maxFin n (λ a →
      maxFin n (λ b →
        maxFin n (λ c → fAbs (amplitude a b c f- closedMPS a b c))))

  meanTargetMass : Float
  meanTargetMass =
    sumFin n (λ a →
      sumFin n (λ b → closedConditional (a , b) (target a b))) f/ (nF f* nF)

  -- A cheap full-contraction smoke test for the first amplitude entry.
  zeroZeroZeroContractionError : Float
  zeroZeroZeroContractionError =
    fAbs (amplitude Fin.zero Fin.zero Fin.zero f- evalMPS Fin.zero Fin.zero Fin.zero)
