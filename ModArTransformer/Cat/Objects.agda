-- The object structure (products, coproducts, exponentials) for our categories,
-- with objects = Agda `Set`.  felix (at the pinned commit) ships these instances
-- only for its Function category in a module that the project's precompiled
-- felix closure does not include, so we provide them here once for the whole
-- project and re-export felix's `Raw` categorical vocabulary.
{-# OPTIONS --without-K #-}
module ModArTransformer.Cat.Objects where

open import Felix.Object using (Products; Coproducts; Exponentials)
open import Felix.Raw public          -- Category, Cartesian, id, _∘_, _▵_, exl, exr, _×_, ⊤, …

import Data.Product as P
import Data.Sum     as S
import Data.Unit    as U
import Data.Empty   as E

instance
  objProducts : Products Set
  objProducts = record { ⊤ = U.⊤ ; _×_ = P._×_ }

  objCoproducts : Coproducts Set
  objCoproducts = record { ⊥ = E.⊥ ; _⊎_ = S._⊎_ }

  objExponentials : Exponentials Set
  objExponentials = record { _⇛_ = λ A B → (A → B) }
