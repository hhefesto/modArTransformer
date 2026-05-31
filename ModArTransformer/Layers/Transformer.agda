-- The full transformer as ONE composite morphism in D (Dual AddFun): embed both
-- tokens, self-attention + residual, LayerNorm, FFN + residual, LayerNorm, then
-- unembed and cross-entropy.  Parameters are a named *product alias* so the
-- generic Additive / Scale / Adamable / Serializable / Forceable instances apply
-- while names stay readable.  The whole backward pass is derived by the chain
-- rule — there is no hand-written gradient code.  Forward reference:
-- old Transformer.agda:80-111 (≡ Main.hs:347-406).  Only position 0 feeds the
-- readout (as in the original); position-1 LN/FFN are omitted since they do not
-- affect the loss (they received zero upstream in the old code).
{-# OPTIONS --guardedness #-}
module ModArTransformer.Layers.Transformer where

open import Data.Nat using (ℕ; suc)
open import Data.Fin using (Fin; zero; suc)

open import ModArTransformer.Tensor using (Float; ℝVec; ℝMat)
open import ModArTransformer.Cat.Objects
open import ModArTransformer.Cat.Additive
open import ModArTransformer.Cat.D
open import ModArTransformer.Cat.VecPrim using (vaddD; crossEntropyAtD)
open import ModArTransformer.Cat.AdditiveTensor
open import ModArTransformer.Layers.Linear
open import ModArTransformer.Layers.LayerNorm
open import ModArTransformer.Layers.FFN
open import ModArTransformer.Layers.Attention
open import ModArTransformer.Layers.Embedding

private variable p dModel dFF dK : ℕ

-- Parameter bundle (vocab = suc p): tokEmbed, posEmbed, attention, LN1, FFN,
-- LN2, unembed.
TransformerParams : ℕ → ℕ → ℕ → ℕ → Set
TransformerParams p dModel dFF dK =
    ℝMat (suc p) dModel                 -- tokEmbed
  × ℝMat 2 dModel                       -- posEmbed
  × AttnParams dModel dK
  × LNParams dModel                     -- LN1
  × FFNParams dModel dFF
  × LNParams dModel                     -- LN2
  × LinParams (suc p) dModel            -- unembed

-- The logits morphism (no target): used for both inference and the loss.
transformerLogits : Fin (suc p) → Fin (suc p)
                  → D (TransformerParams p dModel dFF dK) (ℝVec (suc p))
transformerLogits {p} {dModel} {dFF} {dK} tokA tokB = logits
  where
    Par : Set
    Par = TransformerParams p dModel dFF dK

    -- projections onto the seven parameter groups
    r1 = exrD {A = ℝMat (suc p) dModel}
    getTok  : D Par (ℝMat (suc p) dModel)
    getTok  = exlD
    getPos  : D Par (ℝMat 2 dModel)
    getPos  = exlD ∘D r1
    getAttn : D Par (AttnParams dModel dK)
    getAttn = exlD ∘D (exrD ∘D r1)
    getLn1  : D Par (LNParams dModel)
    getLn1  = exlD ∘D (exrD ∘D (exrD ∘D r1))
    getFFN  : D Par (FFNParams dModel dFF)
    getFFN  = exlD ∘D (exrD ∘D (exrD ∘D (exrD ∘D r1)))
    getLn2  : D Par (LNParams dModel)
    getLn2  = exlD ∘D (exrD ∘D (exrD ∘D (exrD ∘D (exrD ∘D r1))))
    getUn   : D Par (LinParams (suc p) dModel)
    getUn   = exrD ∘D (exrD ∘D (exrD ∘D (exrD ∘D (exrD ∘D r1))))

    -- 1. embeddings (token + position) at the two sequence positions
    embedAt : Fin (suc p) → Fin 2 → D Par (ℝVec dModel)
    embedAt tk ps = vaddD ∘D ((embedRowD tk ∘D getTok) ▵D (embedRowD ps ∘D getPos))
    e0 = embedAt tokA zero
    e1 = embedAt tokB (suc zero)

    -- 2. self-attention (+ residual on position 0)
    a0 : D Par (ℝVec dModel)
    a0 = attn0D ∘D (getAttn ▵D (e0 ▵D e1))
    r10 : D Par (ℝVec dModel)
    r10 = vaddD ∘D (e0 ▵D a0)

    -- 3. LayerNorm 1
    n10 : D Par (ℝVec dModel)
    n10 = layerNormD ∘D (getLn1 ▵D r10)

    -- 4. FFN (+ residual)
    f0 : D Par (ℝVec dModel)
    f0 = ffnD ∘D (getFFN ▵D n10)
    r20 : D Par (ℝVec dModel)
    r20 = vaddD ∘D (n10 ▵D f0)

    -- 5. LayerNorm 2
    o0 : D Par (ℝVec dModel)
    o0 = layerNormD ∘D (getLn2 ▵D r20)

    -- 6. unembed → logits
    logits : D Par (ℝVec (suc p))
    logits = linearLayerD ∘D (getUn ▵D o0)

-- Cross-entropy loss against the target token: the morphism the trainer
-- differentiates.  `softmaxCED` is the softmax+CE primitive (Cat.VecPrim).
transformerLoss : Fin (suc p) → Fin (suc p) → Fin (suc p)
                → D (TransformerParams p dModel dFF dK) Float
transformerLoss tokA tokB target = crossEntropyAtD target ∘D transformerLogits tokA tokB
