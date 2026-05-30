-- Generic flat-Float (de)serialization over a parameter bundle, by structural
-- recursion on the product of tensor leaves.  Replaces the old per-record
-- Checkpoint code.  Leaf order follows the product structure; matching the old
-- on-disk order is a Phase-6 nicety, not required to type-check/compile.
{-# OPTIONS --guardedness #-}
module ModArTransformer.Cat.Serialize where

open import Data.Nat using (ℕ; zero; suc)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Data.List using (List; []; _∷_; _++_)
open import Data.Vec.Base as Vec using (Vec; []; _∷_; toList)

open import ModArTransformer.Tensor

private variable A B : Set

record Serializable (A : Set) : Set where
  field
    toFloats   : A → List Float
    fromFloats : List Float → A × List Float
open Serializable ⦃ … ⦄ public

private
  splitVec : (n : ℕ) → List Float → ℝVec n × List Float
  splitVec zero    xs       = ([] , xs)
  splitVec (suc n) []       = let (v , _) = splitVec n [] in (fzero ∷ v , [])
  splitVec (suc n) (x ∷ xs) = let (v , r) = splitVec n xs in (x ∷ v , r)

  splitMat : (m n : ℕ) → List Float → ℝMat m n × List Float
  splitMat zero    n xs = ([] , xs)
  splitMat (suc m) n xs =
    let (row , r)  = splitVec n xs
        (rows , r') = splitMat m n r
    in (row ∷ rows , r')

instance
  Serializable-ℝVec : {n : ℕ} → Serializable (ℝVec n)
  Serializable-ℝVec {n} = record { toFloats = toList ; fromFloats = splitVec n }

  Serializable-ℝMat : {m n : ℕ} → Serializable (ℝMat m n)
  Serializable-ℝMat {m} {n} = record
    { toFloats   = λ M → concatMap toList (toList M)
    ; fromFloats = splitMat m n }
    where open import Data.List using (concatMap)

  Serializable-× : ⦃ Serializable A ⦄ → ⦃ Serializable B ⦄ → Serializable (A × B)
  Serializable-× = record
    { toFloats   = λ p → toFloats (proj₁ p) ++ toFloats (proj₂ p)
    ; fromFloats = λ xs →
        let (a , r)  = fromFloats xs
            (b , r') = fromFloats r
        in ((a , b) , r') }
