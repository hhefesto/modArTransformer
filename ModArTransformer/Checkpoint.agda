-- Checkpoint I/O helpers.  Parameter (de)serialization is now the generic
-- `Cat.Serialize` (toFloats/fromFloats over the product of tensor leaves); this
-- module keeps only the Float↔String line format and the FFI float parser.
{-# OPTIONS --guardedness #-}
module ModArTransformer.Checkpoint where

open import Agda.Builtin.Float   using (Float; primShowFloat)
open import Agda.Builtin.String  using (String)
open import Data.List            using (List; map)

-- Agda stdlib has no primFloatRead; bind Haskell's reader via FFI.
{-# FOREIGN GHC import Text.Read (readMaybe) #-}
{-# FOREIGN GHC import qualified Data.Text as T #-}

postulate
  parseFloat : String → Float
{-# COMPILE GHC parseFloat = \s -> case (readMaybe (T.unpack s) :: Maybe Double) of Just x -> x; Nothing -> error ("parseFloat: invalid float: " <> T.unpack s) #-}

-- One float per line.
floatsToString : List Float → String
floatsToString fs = Data.String.unlines (map primShowFloat fs)
  where open import Data.String using (unlines)

stringToFloats : String → List Float
stringToFloats s = map parseFloat (Data.String.lines s)
  where open import Data.String using (lines)
