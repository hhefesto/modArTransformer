-- Agda↔Haskell conformance oracle (the deepest guarantee that the backend follows
-- the spec).  Reads a shared flat-float parameter file, rebuilds the *same*
-- TransformerParams the Haskell backend used (the serialization leaf order is
-- verified identical — see CONFORMANCE.md), and for a few fixed (a,b,target)
-- cases emits, one float per line:  logits ++ [loss] ++ gradient.  A Haskell
-- harness computes the same and a flake check tolerance-diffs the two streams.
--
-- Dims are tiny (vocab 3, dModel 4, dFF 8, dK 4) to match the Haskell gradcheck
-- model (Params 3 4 8 4) and keep evaluation cheap.
{-# OPTIONS --guardedness #-}
module modArConformanceOracle where

open import Agda.Builtin.Float  using (Float)
open import Agda.Builtin.String using (String)
open import Data.Nat            using (ℕ)
open import Data.Fin            using (Fin; zero; suc)
open import Data.List           using (List; []; _∷_; _++_)
open import Data.Product        using (_×_; _,_; proj₁; proj₂)
open import Level               using (0ℓ)
open import Data.Unit.Polymorphic.Base using (⊤; tt)
open import Data.Unit.Base            using () renaming (⊤ to Unit; tt to unit)
open import IO.Base                   using (lift; lift′)
import IO.Primitive.Core as Prim
open import IO  using (IO; Main; run; putStrLn; _>>_; _>>=_; pure)

open import ModArTransformer.Tensor              using (ℝVec)
open import ModArTransformer.Layers.Transformer  using (TransformerParams; transformerLogits; transformerLoss)
open import ModArTransformer.Cat.Grad            using (eval; gradAndLoss)
open import ModArTransformer.Cat.Additive
open import ModArTransformer.Cat.AdditiveTensor
open import ModArTransformer.Cat.Serialize       -- opened fully so instances resolve
open import ModArTransformer.Checkpoint          using (floatsToString; stringToFloats)

-- ── concrete tiny model (matches Haskell Params 3 4 8 4) ──────────────────────
private
  p      : ℕ ; p      = 2     -- vocab = suc p = 3
  dModel : ℕ ; dModel = 4
  dFF    : ℕ ; dFF    = 8
  dK     : ℕ ; dK     = 4

Par : Set
Par = TransformerParams p dModel dFF dK

-- ── file I/O via Haskell FFI (mirrors modArTransformer.agda) ──────────────────
{-# FOREIGN GHC import qualified Data.Text as T #-}
postulate
  primReadFile  : String → Prim.IO String
  primWriteFile : String → String → Prim.IO Unit
{-# COMPILE GHC primReadFile  = \ q -> fmap T.pack (readFile (T.unpack q)) #-}
{-# COMPILE GHC primWriteFile = \ q s -> writeFile (T.unpack q) (T.unpack s) #-}

readFileIO : String → IO String
readFileIO q = lift (primReadFile q)
writeFileIO : String → String → IO {0ℓ} ⊤
writeFileIO q s = lift′ (primWriteFile q s)

-- ── tokens (Fin 3) and the three test cases ──────────────────────────────────
ix0 ix1 ix2 : Fin 3
ix0 = zero
ix1 = suc zero
ix2 = suc (suc zero)

-- (a, b, target) with target = (a+b) mod 3
cases : List (Fin 3 × Fin 3 × Fin 3)
cases = (ix1 , ix2 , ix0)        -- (1+2) % 3 = 0
      ∷ (ix0 , ix1 , ix1)        -- (0+1) % 3 = 1
      ∷ (ix2 , ix2 , ix1)        -- (2+2) % 3 = 1
      ∷ []

-- logits ++ [loss] ++ gradient, as a flat float list, for one case.
caseFloats : Par → (Fin 3 × Fin 3 × Fin 3) → List Float
caseFloats params (a , b , t) =
  let logits = eval (transformerLogits a b) params
      gl     = gradAndLoss (transformerLoss a b t) params
  in toFloats logits ++ (proj₂ gl ∷ toFloats (proj₁ gl))

allFloats : Par → List Float
allFloats params = go cases
  where
    go : List (Fin 3 × Fin 3 × Fin 3) → List Float
    go []       = []
    go (c ∷ cs) = caseFloats params c ++ go cs

main : Main
main = run (do
  s ← readFileIO "conformance-params.txt"
  let params : Par
      params = proj₁ (fromFloats (stringToFloats s))
  writeFileIO "conformance-agda.txt" (floatsToString (allFloats params))
  putStrLn "[oracle] wrote conformance-agda.txt")
