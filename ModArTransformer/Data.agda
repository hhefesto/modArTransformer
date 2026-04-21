{-# OPTIONS --guardedness #-}
module ModArTransformer.Data where

open import Data.Nat     using (ℕ; suc; zero; _+_; _%_)
open import Data.Fin     using (Fin; zero; suc; toℕ; fromℕ<)
open import Data.Fin.Properties using (toℕ<n)
open import Data.Nat.DivMod    using (m%n<n)
open import Data.Nat.Properties using (s≤s; z≤n)
open import Data.List    using (List; []; _∷_; _++_; map; length; foldl)
open import Data.Product using (_×_; _,_)
open import Data.Vec.Base as Vec using (Vec; []; _∷_; lookup; toList; fromList)
open import ModArTransformer.Random

-- ─── Training example ─────────────────────────────────────────────────────────

record Example (p : ℕ) : Set where
  constructor mkEx
  field exA exB exTarget : Fin p

-- ─── Generate all (a, b, (a+b) mod p) examples ────────────────────────────────

-- Enumerate all Fins
allFins : (n : ℕ) → List (Fin n)
allFins zero    = []
allFins (suc n) = zero ∷ map suc (allFins n)

-- All p² examples (Main.hs:820-828)
allExamples : (p : ℕ) → List (Example p)
allExamples zero    = []
allExamples (suc p) = go (allFins (suc p))
  where
    go : List (Fin (suc p)) → List (Example (suc p))
    go fins = Data.List.foldl step [] fins
      where
        step : List (Example (suc p)) → Fin (suc p) → List (Example (suc p))
        step acc a = acc ++ Data.List.map (λ b →
              let lt = m%n<n (toℕ a + toℕ b) (suc p)
              in  mkEx a b (fromℕ< lt))
            fins

-- ─── 50/50 train/test split by even/odd position ─────────────────────────────

splitData : {p : ℕ} → List (Example p) → List (Example p) × List (Example p)
splitData []            = ([] , [])
splitData (x ∷ [])     = (x ∷ [] , [])
splitData (x ∷ y ∷ xs) =
  let (tr , te) = splitData xs
  in  (x ∷ tr , y ∷ te)

-- ─── Shuffle a list via Fisher-Yates on Vec ───────────────────────────────────

private
  -- Swap indices i and j in a Vec
  swapV : {n : ℕ} {A : Set} → Vec A n → Fin n → Fin n → Vec A n
  swapV v i j =
    let vi = lookup v i
        vj = lookup v j
        v1 = v Vec.[ i ]≔ vj
    in  v1 Vec.[ j ]≔ vi

  -- Fisher-Yates shuffle of a Vec
  shuffleVec : {n : ℕ} {A : Set} → Vec A n → StdGen → Vec A n × StdGen
  shuffleVec {zero}  v g = (v , g)
  shuffleVec {suc n} v g =
    let (z , g')  = next g
        j         = fromℕ< (Data.Nat.DivMod.m%n<n z (suc n))
        v'        = swapV v zero j
        -- recurse on tail
        tail      = Vec.tail v'
        (tail' , g'') = shuffleVec tail g'
    in  (Vec.head v' ∷ tail' , g'')

shuffleList : {A : Set} → List A → StdGen → List A × StdGen
shuffleList xs g =
  let v          = Vec.fromList xs
      (v' , g')  = shuffleVec v g
  in  (Vec.toList v' , g')

-- ─── Chunk a list into batches ────────────────────────────────────────────────

private
  takeL : {A : Set} → ℕ → List A → List A
  takeL zero    _        = []
  takeL (suc n) []       = []
  takeL (suc n) (x ∷ xs) = x ∷ takeL n xs

  dropL : {A : Set} → ℕ → List A → List A
  dropL zero    xs       = xs
  dropL (suc n) []       = []
  dropL (suc n) (_ ∷ xs) = dropL n xs

  -- chunks via decreasing list length
  chunksOfAux : {A : Set} → ℕ → ℕ → List A → List (List A)
  chunksOfAux _     _         []       = []
  chunksOfAux zero  _         xs       = xs ∷ []
  chunksOfAux (suc fuel) bsz xs@(_ ∷ _) =
    takeL bsz xs ∷ chunksOfAux fuel bsz (dropL bsz xs)

chunksOf : {A : Set} → ℕ → List A → List (List A)
chunksOf bsz xs = chunksOfAux (length xs) bsz xs
