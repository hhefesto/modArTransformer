-- Agda↔Haskell conformance oracle (the deepest guarantee that the backend follows
-- the spec).  Reads a shared flat-float parameter file, rebuilds the *same*
-- TransformerParams the Haskell backend used (the serialization leaf order is
-- verified identical — see README.md §4), and for a few fixed (a,b,target)
-- cases emits, one float per line:  logits ++ [loss] ++ gradient.  A Haskell
-- harness computes the same and a flake check tolerance-diffs the two streams.
--
-- Dims are tiny (vocab 3, dModel 4, dFF 8, dK 4) to match the Haskell gradcheck
-- model (Params 3 4 8 4) and keep evaluation cheap.  A second section gates the
-- GENERALIZED sequence model (Layers.SeqTransformer: n=4 positions, causal,
-- two heads — vocab 5, dM 4, dF 8, dK 2) against the backend's ParamsSeq.
{-# OPTIONS --guardedness #-}
module modArConformanceOracle where

open import Agda.Builtin.Float  using (Float)
open import Agda.Builtin.String using (String)
open import Data.Nat            using (ℕ)
open import Data.Fin            using (Fin; zero; suc)
open import Data.List           using (List; []; _∷_; _++_; concatMap)
open import Data.Product        using (_×_; _,_; proj₁; proj₂)
open import Data.Vec.Base as Vec using (Vec; []; _∷_)
open import Level               using (0ℓ)
open import Data.Unit.Polymorphic.Base using (⊤; tt)
open import Data.Unit.Base            using () renaming (⊤ to Unit; tt to unit)
open import IO.Base                   using (lift; lift′)
import IO.Primitive.Core as Prim
open import IO  using (IO; Main; run; putStrLn; _>>_; _>>=_; pure)

open import ModArTransformer.Tensor              using (ℝVec)
open import ModArTransformer.Layers.Transformer  using (TransformerParams; transformerLogits; transformerLoss)
open import ModArTransformer.Layers.SeqTransformer using (SeqTransformerParams; seqLogits; seqTransformerLoss)
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

-- ── concrete tiny sequence model (matches Haskell ParamsSeq 5 4 4 8 2) ────────
ParS : Set
ParS = SeqTransformerParams 4 4 4 8 2    -- vocab 5, n 4, dM 4, dF 8, dK 2

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

-- ── the sequence-model section ────────────────────────────────────────────────

jx0 jx1 jx2 jx3 jx4 : Fin 5
jx0 = zero
jx1 = suc zero
jx2 = suc (suc zero)
jx3 = suc (suc (suc zero))
jx4 = suc (suc (suc (suc zero)))

-- same two token sequences as the Haskell harness
seqCases : List (Vec (Fin 5) 4)
seqCases = (jx1 ∷ jx4 ∷ jx2 ∷ jx3 ∷ [])
         ∷ (jx0 ∷ jx2 ∷ jx4 ∷ jx1 ∷ [])
         ∷ []

-- per-position logits ++ [loss] ++ gradient, matching perSeqCase in
-- backend/transformer/Conformance.hs.
caseSeqFloats : ParS → Vec (Fin 5) 4 → List Float
caseSeqFloats params toks =
  let ls = Vec.toList (Vec.map (λ m → eval m params) (seqLogits toks))
      gl = gradAndLoss (seqTransformerLoss toks) params
  in concatMap toFloats ls ++ (proj₂ gl ∷ toFloats (proj₁ gl))

allSeqFloats : ParS → List Float
allSeqFloats params = go seqCases
  where
    go : List (Vec (Fin 5) 4) → List Float
    go []       = []
    go (c ∷ cs) = caseSeqFloats params c ++ go cs

main : Main
main = run (do
  s  ← readFileIO "conformance-params.txt"
  s2 ← readFileIO "conformance-seq-params.txt"
  let params : Par
      params = proj₁ (fromFloats (stringToFloats s))
      paramsS : ParS
      paramsS = proj₁ (fromFloats (stringToFloats s2))
  writeFileIO "conformance-agda.txt"
    (floatsToString (allFloats params ++ allSeqFloats paramsS))
  putStrLn "[oracle] wrote conformance-agda.txt")
