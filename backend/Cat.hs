{-# LANGUAGE InstanceSigs #-}

-- A tiny categorical target interface for the backend work.
--
-- This is not Conal's GHC Compiling-to-Categories plugin.  It is the shape of
-- the target we want the plugin, or a handwritten CTC-style IR, to elaborate into:
-- category composition plus cartesian structure.  Keeping this interface small
-- gives us a stable target while we evaluate plugin availability separately.

module Cat
  ( Cat(..)
  , Cartesian(..)
  , AddFun(..)
  ) where

import Prelude hiding (id, (.))
import qualified Prelude as P

class Cat k where
  idC :: k a a
  (<<<) :: k b c -> k a b -> k a c

infixr 9 <<<

class Cat k => Cartesian k where
  exlC :: k (a, b) a
  exrC :: k (a, b) b
  forkC :: k a b -> k a c -> k a (b, c)

instance Cat (->) where
  idC :: a -> a
  idC = P.id

  (<<<) :: (b -> c) -> (a -> b) -> a -> c
  (<<<) = (P..)

instance Cartesian (->) where
  exlC = fst
  exrC = snd
  forkC f g a = (f a, g a)

newtype AddFun a b = AddFun { applyL :: a -> b }

instance Cat AddFun where
  idC = AddFun P.id
  AddFun g <<< AddFun f = AddFun (g P.. f)

instance Cartesian AddFun where
  exlC = AddFun fst
  exrC = AddFun snd
  forkC (AddFun f) (AddFun g) = AddFun (\a -> (f a, g a))
