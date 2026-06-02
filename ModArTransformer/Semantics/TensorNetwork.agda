-- Many-body encoding of the modular-arithmetic language.
--
-- Bradley's reduced-density / tensor-network program starts from a probability
-- distribution on sequences, embeds it as a quantum-like amplitude tensor, and
-- studies the reduced density operators obtained by partial trace.  For the toy
-- modular language we can write the exact distribution down:
--
--   P(a,b,c) = 1 / n^2   if c = a + b mod n
--            = 0         otherwise
--
-- The amplitude tensor ψ = sqrt(P) is a pure three-site state over the operand,
-- operand, and result sites.  Conditioning recovers Bradley's copresheaf
-- π(- | [a,b]); partial traces expose the many-body correlations directly.
{-# OPTIONS --guardedness #-}
module ModArTransformer.Semantics.TensorNetwork where

open import Agda.Builtin.Float using (Float; primNatToFloat)
open import Data.Nat as Nat using (ℕ)
open import Data.Fin as Fin using (Fin)
open import Data.Bool using (if_then_else_)
open import Data.Product using (_×_; _,_)
open import Data.Vec.Base using (Vec; tabulate)

open import ModArTransformer.Tensor
open import ModArTransformer.Semantics.Language

-- Function-view tensors are cheap to define and compose in Agda.  Materializers
-- below turn them into Vec-backed tensors when an explicit finite array is useful.
Tensor3 : ℕ → Set
Tensor3 n = Fin n → Fin n → Fin n → Float

Density1 : ℕ → Set
Density1 n = Fin n → Fin n → Float

Density2 : ℕ → Set
Density2 n = Fin n → Fin n → Fin n → Fin n → Float

sumFin : (n : ℕ) → (Fin n → Float) → Float
sumFin Nat.zero    _ = fzero
sumFin (Nat.suc n) f = f Fin.zero f+ sumFin n (λ i → f (Fin.suc i))

module ManyBody (p : ℕ) where
  open Lang p public using (n; Vocab; Context; target; finEqBool; truth; VocabCat)

  nF : Float
  nF = primNatToFloat n

  uniformContextMass : Float
  uniformContextMass = fone f/ (nF f* nF)

  -- The exact joint distribution over the three sites A,B,C.
  jointP : Tensor3 n
  jointP a b c = if finEqBool c (target a b) then uniformContextMass else fzero

  -- Quantum-like amplitude encoding ψ = sqrt(P).  The state is real and
  -- nonnegative here, so reduced densities do not need complex conjugation.
  amplitude : Tensor3 n
  amplitude a b c = fsqrt (jointP a b c)

  amplitudeSquared : Tensor3 n
  amplitudeSquared a b c = amplitude a b c f* amplitude a b c

  -- Recover the conditional next-token distribution from the amplitude tensor:
  --   π(c | a,b) = |ψ(a,b,c)|² / Σ_k |ψ(a,b,k)|².
  -- For the exact modular tensor this is the same Dirac copresheaf as `truth`.
  contextNorm : Context → Float
  contextNorm (a , b) = sumFin n (λ c → amplitudeSquared a b c)

  conditional : Context → Vocab → Float
  conditional (a , b) c = amplitudeSquared a b c f/ contextNorm (a , b)

  conditionalCopresheaf : Context → Vocab → Float
  conditionalCopresheaf ctx c = conditional ctx c

  -- Pure-state density matrix ρ = |ψ⟩⟨ψ| over A×B×C.
  ρABC : Vocab → Vocab → Vocab → Vocab → Vocab → Vocab → Float
  ρABC a b c a' b' c' = amplitude a b c f* amplitude a' b' c'

  -- One-site reduced density operators.
  ρA : Density1 n
  ρA a a' = sumFin n (λ b → sumFin n (λ c → ρABC a b c a' b c))

  ρB : Density1 n
  ρB b b' = sumFin n (λ a → sumFin n (λ c → ρABC a b c a b' c))

  ρC : Density1 n
  ρC c c' = sumFin n (λ a → sumFin n (λ b → ρABC a b c a b c'))

  -- Two-site reduced density operators.
  ρAB : Density2 n
  ρAB a b a' b' = sumFin n (λ c → ρABC a b c a' b' c)

  ρAC : Density2 n
  ρAC a c a' c' = sumFin n (λ b → ρABC a b c a' b c')

  ρBC : Density2 n
  ρBC b c b' c' = sumFin n (λ a → ρABC a b c a b' c')

  -- Basic diagnostics for density operators.
  trace1 : Density1 n → Float
  trace1 ρ = sumFin n (λ i → ρ i i)

  purity1 : Density1 n → Float
  purity1 ρ = sumFin n (λ i → sumFin n (λ j → ρ i j f* ρ j i))

  -- For the exact modular-addition state, each one-site marginal is maximally
  -- mixed: diagonal entries are 1/n and off-diagonal entries are 0.  Keeping this
  -- closed-form value lets diagnostics report purity without doing an O(n^4)
  -- generic density contraction in the executable.
  oneSitePurityExact : Float
  oneSitePurityExact = fone f/ nF

  -- Explicit finite tensor/matrix views for experiments and serialization later.
  jointTensor : Vec (Vec (Vec Float n) n) n
  jointTensor = tabulate (λ a → tabulate (λ b → tabulate (λ c → jointP a b c)))

  amplitudeTensor : Vec (Vec (Vec Float n) n) n
  amplitudeTensor = tabulate (λ a → tabulate (λ b → tabulate (λ c → amplitude a b c)))

  density1Matrix : Density1 n → ℝMat n n
  density1Matrix ρ = tabulate (λ i → tabulate (λ j → ρ i j))

  ρAMatrix : ℝMat n n
  ρAMatrix = density1Matrix ρA

  ρBMatrix : ℝMat n n
  ρBMatrix = density1Matrix ρB

  ρCMatrix : ℝMat n n
  ρCMatrix = density1Matrix ρC
