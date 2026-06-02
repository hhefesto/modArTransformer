-- Modular Arithmetic Transformer — denotational rewrite.
--
-- Meaning (Tai-Danae Bradley): the model denotes a [0,1]-enriched language
--   category; ⟦θ⟧ ctx = softmax(logits) is a hom-object π(·|ctx).  See
--   ModArTransformer.Semantics.* and .Meaning.
-- Tooling (Conal Elliott): the forward pass is a morphism in D (Dual AddFun)
--   built on felix; gradients are derived by the chain rule — no hand-written
--   backward anywhere.  See ModArTransformer.Cat.* and .Layers.*.
{-# OPTIONS --guardedness #-}
module modArTransformer where

open import Agda.Builtin.Float  using (Float; primNatToFloat; primShowFloat)
open import Agda.Builtin.String using (String)
open import Data.Nat            using (ℕ; zero; suc; _%_)
open import Data.List           using (List; length)
open import Data.Bool           using (Bool; true; false; if_then_else_)
open import Data.Product        using (_×_; _,_; proj₁)
open import Level using (0ℓ)
open import Data.Unit.Polymorphic.Base using (⊤; tt)
open import Data.Unit.Base            using () renaming (⊤ to Unit; tt to unit)
open import IO.Base                   using (lift; lift′)
import IO.Primitive.Core as Prim
open import Data.String         using (_++_)
open import IO                  using (IO; Main; run; putStrLn; _>>_; _>>=_; pure)

open import ModArTransformer.Tensor using (_f*_)
open import ModArTransformer.Layers.Transformer using (TransformerParams)
open import ModArTransformer.Cat.Adamable
open import ModArTransformer.Cat.Additive
open import ModArTransformer.Cat.AdditiveTensor
open import ModArTransformer.Cat.Scale
open import ModArTransformer.Cat.Force
open import ModArTransformer.Cat.Serialize   -- opened fully so its instances resolve
open import ModArTransformer.Random
open import ModArTransformer.Data
open import ModArTransformer.Train
open import ModArTransformer.Init
open import ModArTransformer.Checkpoint
open import ModArTransformer.Semantics.TensorNetwork using ()

-- ─── Hyperparameters (Main.hs:1148-1167) ──────────────────────────────────────

private
  p         : ℕ ; p         = 52   -- vocab = suc p = 53
  dModel    : ℕ ; dModel    = 64
  dFF       : ℕ ; dFF       = 256
  dK        : ℕ ; dK        = 64
  batchSize : ℕ ; batchSize = 32
  epochs    : ℕ ; epochs    = 500000
  seed0     : ℕ ; seed0     = 42
  checkpointEvery : ℕ ; checkpointEvery = 10

  baseLR   : Float ; baseLR  = 1.0e-3
  minLR    : Float ; minLR   = 1.0e-5

  cfg : AdamConfig
  cfg = mkAdamCfg 0.9 0.999 1.0e-8 1.0e-3

  warmupSteps : ℕ ; warmupSteps = 2000

-- The concrete parameter / optimizer types.
Par : Set
Par = TransformerParams p dModel dFF dK

St : Set
St = AdamState Par

-- ─── File I/O via Haskell FFI ─────────────────────────────────────────────────

{-# FOREIGN GHC
  import System.Directory (doesFileExist)
  import System.IO        (hFlush, stdout)
#-}

postulate
  primDoesFileExist : String → Prim.IO Bool
  primReadFile      : String → Prim.IO String
  primWriteFile     : String → String → Prim.IO Unit
  primHFlushStdout  : Prim.IO Unit

{-# FOREIGN GHC import qualified Data.Text as T #-}
{-# COMPILE GHC primDoesFileExist = \p -> doesFileExist (T.unpack p) #-}
{-# COMPILE GHC primReadFile      = \p -> fmap T.pack (readFile (T.unpack p)) #-}
{-# COMPILE GHC primWriteFile     = \p s -> writeFile (T.unpack p) (T.unpack s) #-}
{-# COMPILE GHC primHFlushStdout  = hFlush stdout #-}

doesFileExistIO : String → IO Bool
doesFileExistIO q = lift (primDoesFileExist q)
readFileIO : String → IO String
readFileIO q = lift (primReadFile q)
writeFileIO : String → String → IO Unit
writeFileIO q s = lift (primWriteFile q s)
flushStdoutIO : IO {0ℓ} ⊤
flushStdoutIO = lift′ primHFlushStdout

saveCheckpointIO : Par → IO {0ℓ} ⊤
saveCheckpointIO params = do
  _ ← writeFileIO "checkpoint.ckpt" (floatsToString (toFloats params))
  putStrLn "[checkpoint] saved checkpoint.ckpt"
  flushStdoutIO

-- ─── Helpers ──────────────────────────────────────────────────────────────────

showN : ℕ → String
showN n = primShowFloat (primNatToFloat n)
showF : Float → String
showF = primShowFloat
pct : Float → String
pct f = showF (f f* 100.0) ++ "%"

-- ─── Training loop ────────────────────────────────────────────────────────────

trainLoop : ℕ → ℕ → Par → St
          → List (Example (suc p)) → List (Example (suc p)) → StdGen → IO {0ℓ} ⊤
trainLoop zero    _     _      _    _  _  _ = putStrLn "Done." >> pure tt
trainLoop (suc e) epoch params adam tr te g =
  putStrLn ("[epoch] starting " ++ showN epoch)
  >> flushStdoutIO
  >> continue (trainEpoch epoch batchSize warmupSteps baseLR minLR cfg params adam tr g)
  where
    open import Data.Nat using (_≡ᵇ_)
    continue : Par × St × Float × StdGen → IO {0ℓ} ⊤
    continue (params' , adam' , loss , g') =
      let lossLine = showN epoch ++ " | loss=" ++ showF loss
          evalLine = lossLine
                  ++ " | train=" ++ pct (accuracy params' tr)
                  ++ " | test="  ++ pct (accuracy params' te)
      in (if (epoch % 100) Data.Nat.≡ᵇ 0 then putStrLn evalLine else putStrLn lossLine)
      >> flushStdoutIO
      >> (if (epoch % checkpointEvery) Data.Nat.≡ᵇ 0 then saveCheckpointIO params' else pure tt)
      >> trainLoop e (suc epoch) params' adam' tr te g'

-- ─── Checkpoint loading ────────────────────────────────────────────────────────

loadOrInit : Bool → StdGen → IO {0ℓ} (Par × St × ℕ × StdGen)
loadOrInit true g0 = do
  content ← readFileIO "checkpoint.ckpt"
  let params' = proj₁ (fromFloats {Par} (stringToFloats content))
      as0     = initAdam {Par} 0.9 0.999
  putStrLn "Loaded checkpoint.ckpt"
  pure (params' , as0 , 1 , g0)
loadOrInit false g0 =
  continue (initTransformer p dModel dFF dK seed0 g0)
  where
    continue : Par × StdGen → IO {0ℓ} (Par × St × ℕ × StdGen)
    continue (params' , g') = do
      let as0 = initAdam {Par} 0.9 0.999
      putStrLn "Initialized fresh parameters"
      pure (params' , as0 , 1 , g')

-- ─── Main ─────────────────────────────────────────────────────────────────────

main : Main
main = run (do
  putStrLn "Modular Arithmetic Transformer — denotational (Tai-Danae × Conal)"
  putStrLn "Backprop: gradients derived in D (Dual AddFun) on felix"
  putStrLn "=========================================================="

  let allData                = allExamples (suc p)
      (trainData , testData) = splitData allData
      g0                     = mkStdGenFromSeed seed0

  exists ← doesFileExistIO "checkpoint.ckpt"
  init-result ← loadOrInit exists g0
  let (params0 , adam0 , startEpoch , g1) = init-result

  putStrLn ("Train: " ++ showN (length trainData)
         ++ "  Test: " ++ showN (length testData))
  putStrLn ""

  trainLoop epochs startEpoch params0 adam0 trainData testData g1)
