-- Standalone diagnostics for the exact modular-addition many-body encoding.
-- This executable does not train the transformer.  It evaluates cheap invariants
-- of the exact amplitude/density semantics so we can inspect the Bradley +
-- many-body path independently of the slow Agda/MAlonzo training loop.
{-# OPTIONS --guardedness #-}
module modArTensorDiagnostics where

open import Agda.Builtin.Float using (Float; primShowFloat; primNatToFloat)
open import Agda.Builtin.String using (String)
open import Data.Nat using (ℕ)
open import Data.String using (_++_)
open import Data.Unit.Polymorphic.Base using (⊤; tt)
open import IO using (IO; Main; run; putStrLn; _>>_; pure)
open import Level using (0ℓ)

open import ModArTransformer.Semantics.TensorNetwork
open import ModArTransformer.Semantics.MPS

private
  p : ℕ
  p = 52

module MB = ManyBody p
module EM = ExactMPS p

open MB using (n; trace1; ρA; ρB; ρC; oneSitePurityExact)
open EM using
  ( bondDim
  ; maxClosedAmplitudeError
  ; zeroZeroZeroContractionError
  ; meanTargetMass
  )

showF : Float → String
showF = primShowFloat

showN : ℕ → String
showN n = primShowFloat (primNatToFloat n)

lineF : String → Float → IO {0ℓ} ⊤
lineF label x = putStrLn (label ++ showF x)

main : Main
main = run (
  putStrLn "Modular arithmetic tensor diagnostics"
  >> putStrLn "====================================="
  >> putStrLn ("vocab n = " ++ showN n)
  >> putStrLn "Exact state: psi(a,b,c)=sqrt(1/n^2) when c=a+b mod n"
  >> putStrLn ""
  >> lineF "trace rhoA = " (trace1 ρA)
  >> lineF "trace rhoB = " (trace1 ρB)
  >> lineF "trace rhoC = " (trace1 ρC)
  >> putStrLn ""
  >> lineF "one-site purity exact = " oneSitePurityExact
  >> putStrLn "Expected: traces near 1.0; purity near 1/n for maximally mixed sites."
  >> putStrLn ""
  >> putStrLn "Exact MPS / tensor-train"
  >> putStrLn "------------------------"
  >> putStrLn ("bond dimension chi = " ++ showN bondDim)
  >> lineF "max closed-form amplitude error = " maxClosedAmplitudeError
  >> lineF "zero-zero-zero contraction error = " zeroZeroZeroContractionError
  >> lineF "mean target conditional mass = " meanTargetMass
  >> putStrLn "Expected: amplitude errors near 0.0; target mass near 1.0."
  >> pure tt)
