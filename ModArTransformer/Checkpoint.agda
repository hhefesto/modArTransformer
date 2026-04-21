{-# OPTIONS --guardedness #-}
module ModArTransformer.Checkpoint where

open import Agda.Builtin.Float   using (Float; primShowFloat)
open import Agda.Builtin.String  using (String; primStringToList; primStringFromList)
open import Data.Nat             using (ℕ; zero; suc)
open import Data.Fin             using (Fin)
open import Data.List            using (List; []; _∷_; _++_; map; foldl; length)
open import Data.Vec.Base as Vec using (Vec)
open import Data.Char            using (Char)
open import Data.Product         using (_×_; _,_)
open import Data.String          using (lines; unlines; words)
open import ModArTransformer.Tensor
open import ModArTransformer.Layers.Linear
open import ModArTransformer.Layers.LayerNorm
open import ModArTransformer.Layers.FFN
open import ModArTransformer.Layers.Attention
open import ModArTransformer.Layers.Transformer
open import ModArTransformer.Optimizer.Adam

-- ─── Float parsing ────────────────────────────────────────────────────────────
-- Agda stdlib doesn't have primFloatRead directly; we use the FFI binding.

{-# FOREIGN GHC import Text.Read (readMaybe) #-}
{-# FOREIGN GHC import Data.Maybe (fromMaybe) #-}
{-# FOREIGN GHC import qualified Data.Text as T #-}

postulate
  parseFloat : String → Float

{-# COMPILE GHC parseFloat = \s -> fromMaybe 0.0 (readMaybe (T.unpack s) :: Maybe Double) #-}

-- ─── Vec ↔ List Float ─────────────────────────────────────────────────────────

vecToList : {n : ℕ} → ℝVec n → List Float
vecToList Vec.[] = []
vecToList (x Vec.∷ xs) = x ∷ vecToList xs

listToVec : (n : ℕ) → List Float → ℝVec n
listToVec zero    _        = Vec.[]
listToVec (suc n) []       = fzero Vec.∷ listToVec n []
listToVec (suc n) (x ∷ xs) = x Vec.∷ listToVec n xs

matToList : {m n : ℕ} → ℝMat m n → List Float
matToList Vec.[]       = []
matToList (r Vec.∷ rs) = vecToList r ++ matToList rs

listToMat : (m n : ℕ) → List Float → ℝMat m n
listToMat zero    _ _  = Vec.[]
listToMat (suc m) n xs =
  listToVec n (takeN n xs) Vec.∷ listToMat m n (dropN n xs)
  where
    takeN : ℕ → List Float → List Float
    takeN zero    _        = []
    takeN (suc n) []       = []
    takeN (suc n) (x ∷ xs) = x ∷ takeN n xs

    dropN : ℕ → List Float → List Float
    dropN zero    xs       = xs
    dropN (suc n) []       = []
    dropN (suc n) (_ ∷ xs) = dropN n xs

-- ─── Serialize / deserialize helpers (Main.hs:956-983) ───────────────────────

serializeLinear : {out inp : ℕ} → LinearParams out inp → List Float
serializeLinear p = matToList (linW p) ++ vecToList (linB p)

deserializeLinear : (out inp : ℕ) → List Float → LinearParams out inp × List Float
deserializeLinear out inp xs =
  let (wData , xs1) = splitAt (out * inp) xs
      (bData , xs2) = splitAt out xs1
  in  (mkLinear (listToMat out inp wData) (listToVec out bData) , xs2)
  where
    open import Data.Nat using (_*_)
    splitAt : {A : Set} → ℕ → List A → List A × List A
    splitAt zero    xs       = ([] , xs)
    splitAt (suc n) []       = ([] , [])
    splitAt (suc n) (x ∷ xs) = let (ys , zs) = splitAt n xs in (x ∷ ys , zs)

serializeLN : {n : ℕ} → LayerNormParams n → List Float
serializeLN p = vecToList (lnGamma p) ++ vecToList (lnBeta p)

deserializeLN : (n : ℕ) → List Float → LayerNormParams n × List Float
deserializeLN n xs =
  let (gData , xs1) = splitAt n xs
      (bData , xs2) = splitAt n xs1
  in  (mkLN (listToVec n gData) (listToVec n bData) , xs2)
  where
    splitAt : {A : Set} → ℕ → List A → List A × List A
    splitAt zero    xs       = ([] , xs)
    splitAt (suc n) []       = ([] , [])
    splitAt (suc n) (x ∷ xs) = let (ys , zs) = splitAt n xs in (x ∷ ys , zs)

-- ─── Full parameter serialization (Main.hs:974-1002) ─────────────────────────

serializeParams : {p dModel dFF dK : ℕ}
                → ℕ → ℕ → ℕ → ℕ
                → TransformerParams (suc p) dModel dFF dK
                → List Float
serializeParams vocab dModel dFF dK params =
  matToList (tokEmbed params)
  ++ matToList (posEmbed params)
  ++ serializeLinear (attnWq (attnP params))
  ++ serializeLinear (attnWk (attnP params))
  ++ serializeLinear (attnWv (attnP params))
  ++ serializeLinear (attnWo (attnP params))
  ++ serializeLN (ln1P params)
  ++ serializeLinear (ffnLinear1 (ffnP params))
  ++ serializeLinear (ffnLinear2 (ffnP params))
  ++ serializeLN (ln2P params)
  ++ serializeLinear (unembed params)

deserializeParams : (p dModel dFF dK : ℕ)
                  → List Float
                  → TransformerParams (suc p) dModel dFF dK × List Float
deserializeParams p dModel dFF dK xs0 =
  let vocab       = suc p
      (tokD , xs1)  = splitAt (vocab * dModel) xs0
      (posD , xs2)  = splitAt (2 * dModel)     xs1
      (wq   , xs3)  = deserializeLinear dK    dModel xs2
      (wk   , xs4)  = deserializeLinear dK    dModel xs3
      (wv   , xs5)  = deserializeLinear dK    dModel xs4
      (wo   , xs6)  = deserializeLinear dModel dK    xs5
      (ln1p , xs7)  = deserializeLN dModel xs6
      (ff1  , xs8)  = deserializeLinear dFF   dModel xs7
      (ff2  , xs9)  = deserializeLinear dModel dFF   xs8
      (ln2p , xs10) = deserializeLN dModel xs9
      (unemb , xs11) = deserializeLinear vocab dModel xs10
  in  (mkTransformer (listToMat vocab dModel tokD)
                     (listToMat 2     dModel posD)
                     (mkAttn wq wk wv wo)
                     ln1p (mkFFN ff1 ff2) ln2p unemb
      , xs11)
  where
    open import Data.Nat using (_*_)
    splitAt : {A : Set} → ℕ → List A → List A × List A
    splitAt zero    xs       = ([] , xs)
    splitAt (suc n) []       = ([] , [])
    splitAt (suc n) (x ∷ xs) = let (ys , zs) = splitAt n xs in (x ∷ ys , zs)

-- ─── Write / read checkpoint files (line-per-float format of Main.hs) ────────

floatsToString : List Float → String
floatsToString fs = Data.String.unlines (map primShowFloat fs)
  where open import Data.String using (unlines)

stringToFloats : String → List Float
stringToFloats s = map parseFloat (Data.String.lines s)
  where open import Data.String using (lines)
