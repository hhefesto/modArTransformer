-- Modular Arithmetic Transformer — Agda port
-- Backpropagation via Conal Elliott's categorical AD:
--   D (Dual AddFun) TransformerParams Float
-- No hand-written *Backward functions — gradients are derived from
-- the forward morphism via the chain rule in the D category.
--
-- Reference: Elliott, "The Simple Essence of Automatic Differentiation" (2018)

{-# OPTIONS --guardedness #-}
module modArTransformer where

open import Agda.Builtin.Float  using (Float; primNatToFloat; primShowFloat)
open import Agda.Builtin.String using (String)
open import Data.Nat            using (ℕ; zero; suc; _+_; _%_)
open import Data.Fin            using (Fin; zero; suc; toℕ)
open import Data.List           using (List; []; _∷_; length)
open import Data.Bool           using (Bool; true; false; if_then_else_)
open import Data.Product        using (_×_; _,_)
open import Level using (0ℓ)
open import Data.Unit.Polymorphic.Base using (⊤; tt)
open import Data.Unit.Base            using () renaming (⊤ to Unit; tt to unit)
open import IO.Base                   using (lift; lift′)
import IO.Primitive.Core as Prim
open import Data.String         using (_++_)
open import IO                  using (IO; Main; run; putStrLn; _>>_; _>>=_; pure)

open import ModArTransformer.Tensor
open import ModArTransformer.Random
open import ModArTransformer.Layers.Transformer
open import ModArTransformer.Optimizer.Adam
open import ModArTransformer.Optimizer.Schedule
open import ModArTransformer.Data
open import ModArTransformer.Train
open import ModArTransformer.Init
open import ModArTransformer.Checkpoint

-- ─── Hyperparameters (matching Main.hs:1148-1167) ─────────────────────────────

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
doesFileExistIO p = lift (primDoesFileExist p)

readFileIO : String → IO String
readFileIO p = lift (primReadFile p)

writeFileIO : String → String → IO Unit
writeFileIO p s = lift (primWriteFile p s)

flushStdoutIO : IO {0ℓ} ⊤
flushStdoutIO = lift′ primHFlushStdout

saveCheckpointIO : TransformerParams (suc p) dModel dFF dK → IO {0ℓ} ⊤
saveCheckpointIO params = do
  let floats = serializeParams (suc p) dModel dFF dK params
      text   = floatsToString floats
  _ ← writeFileIO "checkpoint.ckpt" text
  putStrLn "[checkpoint] saved checkpoint.ckpt"
  flushStdoutIO

-- ─── Helpers ──────────────────────────────────────────────────────────────────

showN : ℕ → String
showN n = primShowFloat (primNatToFloat n)

showF : Float → String
showF = primShowFloat

pct : Float → String
pct f = showF (f f* 100.0) ++ "%"
  where open import ModArTransformer.Tensor using (_f*_)

-- ─── Training loop ────────────────────────────────────────────────────────────

trainLoop : ℕ    -- remaining epochs
          → ℕ    -- current epoch number (for logging and schedule)
          → TransformerParams (suc p) dModel dFF dK
          → AdamState p dModel dFF dK
          → List (Example (suc p))
          → List (Example (suc p))
          → StdGen
          → IO {0ℓ} ⊤
trainLoop zero    _     _      _    _  _  _ = putStrLn "Done." >> pure tt
trainLoop (suc e) epoch params adam tr te g =
  continue (trainEpoch epoch batchSize warmupSteps baseLR minLR cfg params adam tr g)
  where
    open import Data.Nat using (_≡ᵇ_; _%_)

    continue : TransformerParams (suc p) dModel dFF dK
             × AdamState p dModel dFF dK
             × Float × StdGen
             → IO {0ℓ} ⊤
    continue (params' , adam' , loss , g') =
      let lossLine = showN epoch ++ " | loss=" ++ showF loss
          evalLine = lossLine
                  ++ " | train=" ++ pct (accuracy params' tr)
                  ++ " | test="  ++ pct (accuracy params' te)
      in
      (if (epoch % 100) Data.Nat.≡ᵇ 0
        then putStrLn evalLine
        else putStrLn lossLine)
      >> flushStdoutIO
      >> (if (epoch % checkpointEvery) Data.Nat.≡ᵇ 0
          then saveCheckpointIO params'
          else pure tt)
      >> trainLoop e (suc epoch) params' adam' tr te g'

-- ─── Checkpoint loading helper with explicit types ────────────────────────────

loadOrInit : Bool → StdGen
           → IO {0ℓ} (TransformerParams (suc p) dModel dFF dK
                     × AdamState p dModel dFF dK
                     × ℕ × StdGen)
loadOrInit true g0 = do
  content ← readFileIO "checkpoint.ckpt"
  let floats            = stringToFloats content
      (params' , _)     = deserializeParams p dModel dFF dK floats
      as0               = initAdamState {p} {dModel} {dFF} {dK} 0.9 0.999
  putStrLn "Loaded checkpoint.ckpt"
  pure (params' , as0 , 1 , g0)
loadOrInit false g0 =
  continue (initTransformer (suc p) dModel dFF dK seed0 g0)
  where
    continue : TransformerParams (suc p) dModel dFF dK × StdGen
             → IO {0ℓ} (TransformerParams (suc p) dModel dFF dK
                       × AdamState p dModel dFF dK
                       × ℕ × StdGen)
    continue (params' , g') = do
      let as0 = initAdamState {p} {dModel} {dFF} {dK} 0.9 0.999
      putStrLn "Initialized fresh parameters"
      pure (params' , as0 , 1 , g')

-- ─── Main ─────────────────────────────────────────────────────────────────────

main : Main
main = run (do
  putStrLn "Modular Arithmetic Transformer — Agda port with Conal-style AD"
  putStrLn "Backprop: D (Dual AddFun) TransformerParams Float"
  putStrLn "=========================================================="

  let allData              = allExamples (suc p)
      (trainData , testData) = splitData allData
      g0                   = mkStdGenFromSeed seed0

  exists ← doesFileExistIO "checkpoint.ckpt"
  init-result ← loadOrInit exists g0
  let (params0 , adam0 , startEpoch , g1) = init-result

  putStrLn ("Train: " ++ showN (length trainData)
         ++ "  Test: " ++ showN (length testData))
  putStrLn ""

  trainLoop epochs startEpoch params0 adam0 trainData testData g1)
