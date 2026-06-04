{-# OPTIONS --guardedness #-}
module ModArTransformer.Random where

open import Agda.Builtin.Float using (Float; primNatToFloat)
open import Data.Nat     using (ℕ; zero; suc; _+_; _*_; _%_)
open import Data.Bool    using (Bool; true; false; if_then_else_)
open import Data.Nat.Properties using (m<n+m; ≤-refl)
open import Data.Product using (_×_; _,_)
open import ModArTransformer.Tensor using (_f+_; _f-_; _f*_; _f/_; fzero)

-- ─── Combined LCG (the LEGACY, pre-1.2 GHC System.Random StdGen) ──────────────
-- L'Ecuyer (1988), two-component combined generator: g1 advances with
-- (a1*s mod m1), g2 with (a2*s mod m2), output = (g1 - g2) mod m1.
--
-- NOTE (RNG matching): this reproduces the OLD `random` (< 1.2) StdGen.  Since
-- `random` 1.2 (2020) the Haskell StdGen is SplitMix — and the backend resolves
-- `random-1.2.1.3` — so this generator and the backend's `mkStdGen`/`randomR` do
-- NOT produce the same stream.  This RNG drives only the spec's *own* training
-- loop (init, shuffle); Agda↔Haskell RNG / training-data equivalence is NOT
-- claimed.  The conformance oracle is unaffected: it loads shared fixed params
-- from a file, so the RNG is never exercised (it verifies the model, not the
-- training recipe — see CONFORMANCE.md "Scope of the guarantee").

record StdGen : Set where
  constructor mkStdGen
  field s1 s2 : ℕ

private
  m1 = 2147483563
  m2 = 2147483399
  a1 = 40014
  a2 = 40692

-- One step: produces a value in [1, m1) and advances the generator.
next : StdGen → ℕ × StdGen
next (mkStdGen s1 s2) =
  let s1' = (a1 * s1) % m1
      s2' = (a2 * s2) % m2
      -- ensure positive difference: if s1' ≤ s2' add m1
      z   = if s1' Data.Nat.≤ᵇ s2'
              then (s1' + m1) Data.Nat.∸ s2'
              else s1' Data.Nat.∸ s2'
  in  (z , mkStdGen s1' s2')
  where open import Data.Nat using (_≤ᵇ_; _∸_)

-- Normalize to Float in [0, 1)
toFloat01 : ℕ → Float
toFloat01 z = primNatToFloat z f/ primNatToFloat m1

-- Float in [lo, hi)
nextFloat : Float → Float → StdGen → Float × StdGen
nextFloat lo hi g =
  let (z , g') = next g
  in  (lo f+ (hi f- lo) f* toFloat01 z , g')

-- Make a StdGen from an integer seed (matches Haskell mkStdGen).
-- GHC: mkStdGen n = StdGen (s1+1) (s2+1) where s1 = n%m1, s2 = n%m2
mkStdGenFromSeed : ℕ → StdGen
mkStdGenFromSeed seed =
  mkStdGen ((seed % m1) + 1) ((seed % m2) + 1)
