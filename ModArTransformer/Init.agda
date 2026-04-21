{-# OPTIONS --guardedness #-}
module ModArTransformer.Init where

open import Agda.Builtin.Float using (Float; primNatToFloat)
open import Data.Nat     using (ℕ; zero; suc; _+_; _*_)
open import Data.Fin     using (Fin; zero; suc)
open import Data.List    using (List) renaming ([] to L[]; _∷_ to _L∷_)
open import Data.Product  using (_×_; _,_)
open import Data.Vec.Base as Vec using (Vec)
open import ModArTransformer.Tensor
open import ModArTransformer.Random
open import ModArTransformer.Layers.Linear
open import ModArTransformer.Layers.LayerNorm
open import ModArTransformer.Layers.FFN
open import ModArTransformer.Layers.Attention
open import ModArTransformer.Layers.Transformer

-- ─── Generate n floats in [-bound, bound] ─────────────────────────────────────

private
  genFloats : ℕ → Float → StdGen → List Float × StdGen
  genFloats zero    _     g = (L[] , g)
  genFloats (suc n) bound g =
    let (v  , g1) = nextFloat (fneg bound) bound g
        (vs , g2) = genFloats n bound g1
    in  (v L∷ vs , g2)

-- ─── Build ℝVec n from a List Float ───────────────────────────────────────────

private
  listToVecN : (n : ℕ) → List Float → ℝVec n × List Float
  listToVecN zero    xs        = (Vec.[] , xs)
  listToVecN (suc n) L[]       = let (v , rest) = listToVecN n L[] in (fzero Vec.∷ v , rest)
  listToVecN (suc n) (x L∷ xs) =
    let (v , rest) = listToVecN n xs
    in  (x Vec.∷ v , rest)

  listToMatMN : (m n : ℕ) → List Float → ℝMat m n × List Float
  listToMatMN zero    _ xs = (Vec.[] , xs)
  listToMatMN (suc m) n xs =
    let (row  , xs1) = listToVecN n xs
        (rows , xs2) = listToMatMN m n xs1
    in  (row Vec.∷ rows , xs2)

-- ─── Xavier uniform init (Main.hs:772-778) ────────────────────────────────────
-- bound = sqrt(6 / (r + c)), uniform in [-bound, bound]

xavierMat : (r c : ℕ) → StdGen → ℝMat r c × StdGen
xavierMat r c g0 =
  let bound   = fsqrt (6.0 f/ primNatToFloat (r + c))
      total   = r * c
      (fs , g1) = genFloats total bound g0
      (mat , _) = listToMatMN r c fs
  in  (mat , g1)

initLinear : (out inp : ℕ) → StdGen → LinearParams out inp × StdGen
initLinear out inp g =
  let (w , g') = xavierMat out inp g
  in  (mkLinear w vzero , g')

initLayerNorm : {n : ℕ} → LayerNormParams n
initLayerNorm = mkLN (vkonst fone) vzero   -- gamma=1, beta=0

-- ─── Full model initialization (Main.hs:785-805) ─────────────────────────────

initTransformer : (vocab dModel dFF dK : ℕ) → ℕ
                → StdGen
                → TransformerParams vocab dModel dFF dK × StdGen
initTransformer vocab dModel dFF dK _seed g0 =
  let (tokE  , g1) = xavierMat vocab dModel g0
      (posE  , g2) = xavierMat 2     dModel g1
      (wq    , g3) = initLinear dK    dModel g2   -- LinearParams dK dModel
      (wk    , g4) = initLinear dK    dModel g3
      (wv    , g5) = initLinear dK    dModel g4
      (wo    , g6) = initLinear dModel dK    g5   -- LinearParams dModel dK
      (ff1   , g7) = initLinear dFF   dModel g6
      (ff2   , g8) = initLinear dModel dFF   g7
      (unemb , g9) = initLinear vocab  dModel g8
      params = mkTransformer tokE posE
                 (mkAttn wq wk wv wo)
                 initLayerNorm
                 (mkFFN ff1 ff2)
                 initLayerNorm
                 unemb
  in  (params , g9)
