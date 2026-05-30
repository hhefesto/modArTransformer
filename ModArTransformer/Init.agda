-- Xavier-uniform initialization, building the product-typed TransformerParams
-- leaf by leaf (each leaf needs its own fan-in/out, so init stays a direct
-- constructor rather than a generic instance).  Reuses `xavierMat` (Main.hs:772).
{-# OPTIONS --guardedness #-}
module ModArTransformer.Init where

open import Agda.Builtin.Float using (Float; primNatToFloat)
open import Data.Nat     using (ℕ; zero; suc; _+_; _*_)
open import Data.List    using (List) renaming ([] to L[]; _∷_ to _L∷_)
open import Data.Product using (_×_; _,_)
open import Data.Vec.Base as Vec using (Vec)
open import ModArTransformer.Tensor
open import ModArTransformer.Random
open import ModArTransformer.Layers.Linear      using (LinParams)
open import ModArTransformer.Layers.LayerNorm   using (LNParams)
open import ModArTransformer.Layers.FFN         using (FFNParams)
open import ModArTransformer.Layers.Attention   using (AttnParams)
open import ModArTransformer.Layers.Transformer using (TransformerParams)

private
  genFloats : ℕ → Float → StdGen → List Float × StdGen
  genFloats zero    _     g = (L[] , g)
  genFloats (suc n) bound g =
    let (v  , g1) = nextFloat (fneg bound) bound g
        (vs , g2) = genFloats n bound g1
    in  (v L∷ vs , g2)

  listToVecN : (n : ℕ) → List Float → ℝVec n × List Float
  listToVecN zero    xs        = (Vec.[] , xs)
  listToVecN (suc n) L[]       = let (v , rest) = listToVecN n L[] in (fzero Vec.∷ v , rest)
  listToVecN (suc n) (x L∷ xs) = let (v , rest) = listToVecN n xs in (x Vec.∷ v , rest)

  listToMatMN : (m n : ℕ) → List Float → ℝMat m n × List Float
  listToMatMN zero    _ xs = (Vec.[] , xs)
  listToMatMN (suc m) n xs =
    let (row  , xs1) = listToVecN n xs
        (rows , xs2) = listToMatMN m n xs1
    in  (row Vec.∷ rows , xs2)

-- bound = sqrt(6 / (r + c)), uniform in [-bound, bound]
xavierMat : (r c : ℕ) → StdGen → ℝMat r c × StdGen
xavierMat r c g0 =
  let bound     = fsqrt (6.0 f/ primNatToFloat (r + c))
      (fs , g1) = genFloats (r * c) bound g0
      (mat , _) = listToMatMN r c fs
  in  (mat , g1)

-- a linear layer: Xavier weights, zero bias
initLin : (out inp : ℕ) → StdGen → LinParams out inp × StdGen
initLin out inp g = let (w , g') = xavierMat out inp g in ((w , vzero) , g')

-- LayerNorm: γ = 1, β = 0
initLN : {n : ℕ} → LNParams n
initLN = (vkonst fone , vzero)

initTransformer : (p dModel dFF dK : ℕ) → ℕ → StdGen
                → TransformerParams p dModel dFF dK × StdGen
initTransformer p dModel dFF dK _seed g0 =
  let (tokE , g1) = xavierMat (suc p) dModel g0
      (posE , g2) = xavierMat 2       dModel g1
      (wq   , g3) = initLin dK     dModel g2
      (wk   , g4) = initLin dK     dModel g3
      (wv   , g5) = initLin dK     dModel g4
      (wo   , g6) = initLin dModel dK     g5
      (ff1  , g7) = initLin dFF    dModel g6
      (ff2  , g8) = initLin dModel dFF    g7
      (unemb , g9) = initLin (suc p) dModel g8
  in  ( ( tokE , posE , (wq , wk , wv , wo) , initLN , (ff1 , ff2) , initLN , unemb )
      , g9 )
