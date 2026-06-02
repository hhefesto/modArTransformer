{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FlexibleContexts #-}

-- Conal Elliott's AD-as-categories, reverse mode, with accumulator-threaded
-- cotangents.
--
--   D a b = a -> (b, Dual a b)
--   Dual a b = b -> a -> a            -- "add my contribution to the accumulator"
--
-- The accumulator form is the efficiency crux: forkD threads ONE input-cotangent
-- accumulator through both branches, and the projection adjoints (exlD/exrD) touch
-- only their own component.  So a parameter read by many downstream branches is
-- summed into its own leaf only — never via dense adds of the whole parameter
-- product.  Intermediate (activation) cotangents are materialised at composition,
-- where they are small.  Still: one backward pass, gradient by the chain rule, no
-- hand-written backward for any composite.
module AD
  ( Dual(..)
  , D(..)
  , mkD
  , linearD
  , idD
  , (>->)
  , forkD
  , exlD
  , exrD
  , constD
  , gradAndLoss
  , evalD
  , Lens(..)
  , fstL
  , sndL
  , (.<)
  , projD
  ) where

import Tensor (Additive(..))

-- Reverse linear map: output-cotangent -> accumulator -> accumulator'.
newtype Dual a b = Dual { pullInto :: b -> a -> a }

newtype D a b = D { runD :: a -> (b, Dual a b) }

-- Smart constructor from an old-style adjoint (b -> a); the contribution is added
-- into the accumulator.  Used by leaf primitives whose domain is small.
mkD :: Additive a => (a -> (b, b -> a)) -> D a b
mkD f = D (\a -> let (b, back) = f a in (b, Dual (\db acc -> addA acc (back db))))
{-# INLINE mkD #-}

linearD :: Additive a => (a -> b) -> (b -> a) -> D a b
linearD fwd back = mkD (\a -> (fwd a, back))
{-# INLINE linearD #-}

idD :: Additive a => D a a
idD = D (\a -> (a, Dual (\da acc -> addA acc da)))

-- Composition (chain rule).  `g >-> f` is g ∘ f.  Materialises the intermediate
-- cotangent (type b — an activation, small) starting from zeroA, then threads it
-- back through f.
(>->) :: forall a b c. Additive b => D b c -> D a b -> D a c
D g >-> D f = D $ \a ->
  let (!b, f') = f a
      (!c, g') = g b
  in (c, Dual (\dc acc -> pullInto f' (pullInto g' dc (zeroA :: b)) acc))
infixr 9 >->
{-# INLINE (>->) #-}

-- Fork: run both on the same input; thread ONE accumulator through both adjoints.
-- No full-domain addition — each branch adds only into the leaves it touches.
forkD :: D a b -> D a c -> D a (b, c)
forkD (D f) (D g) = D $ \a ->
  let (!b, f') = f a
      (!c, g') = g a
  in ((b, c), Dual (\(db, dc) acc -> pullInto g' dc (pullInto f' db acc)))
{-# INLINE forkD #-}

-- Projections: adjoint updates ONLY its component of the pair accumulator.
exlD :: Additive a => D (a, b) a
exlD = D (\(a, _) -> (a, Dual (\da (af, bf) -> (addA af da, bf))))

exrD :: Additive b => D (a, b) b
exrD = D (\(_, b) -> (b, Dual (\db (af, bf) -> (af, addA bf db))))

constD :: b -> D a b
constD b = D (\_ -> (b, Dual (\_ acc -> acc)))

-- ── lens-based projection (field-local, no intermediate materialisation) ──────
--
-- A getter built from a Lens injects an output cotangent straight into ONE leaf of
-- the parameter accumulator.  Lenses compose (.<) by composing their modify
-- functions, so a deep leaf is reached without ever allocating a zero of any
-- enclosing structure — unlike chaining exlD/exrD through (>->), which materialises
-- a full-product zero at every link.
data Lens s a = Lens { lget :: s -> a, lmod :: (a -> a) -> s -> s }

fstL :: Lens (a, b) a
fstL = Lens fst (\f (a, b) -> (f a, b))

sndL :: Lens (a, b) b
sndL = Lens snd (\f (a, b) -> (a, f b))

(.<) :: Lens s a -> Lens a b -> Lens s b
Lens g1 m1 .< Lens g2 m2 = Lens (g2 . g1) (\f -> m1 (m2 f))
infixr 9 .<

projD :: Additive a => Lens s a -> D s a
projD (Lens g m) = D (\s -> (g s, Dual (\da acc -> m (`addA` da) acc)))
{-# INLINE projD #-}

evalD :: D a b -> a -> b
evalD (D f) a = fst (f a)

-- Gradient of a scalar morphism: one forward + one backward, accumulating from the
-- zero parameter cotangent.
gradAndLoss :: Additive a => D a Double -> a -> (a, Double)
gradAndLoss (D f) a =
  let (!l, f') = f a
      !g = pullInto f' 1.0 zeroA
  in (g, l)
{-# INLINE gradAndLoss #-}
