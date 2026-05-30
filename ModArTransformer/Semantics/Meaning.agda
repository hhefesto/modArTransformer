-- The denotational-design keystone: it connects the differentiable model
-- (Cat.*) to Tai-Danae Bradley's semantics (Semantics.*).
--
--   ⟦ model ⟧ ctx  =  softmax (logits θ ctx)   : a [0,1]-copresheaf on VocabCat,
--
-- i.e. the model's softmax output *is* a hom-object π(· | ctx) of the learned
-- [0,1]-enriched language category — the meaning of the context ctx.  Training
-- minimizes the cross-entropy of this copresheaf against the ground-truth Dirac
-- copresheaf `truth ctx` (Semantics.Language); cross-entropy = relative entropy
-- to a one-hot target, which is the Shannon (t→1) limit of Bradley's
-- magnitude / Tsallis-entropy invariant.  So the loss is the semantic quantity,
-- not an ad-hoc objective.
{-# OPTIONS --guardedness #-}
module ModArTransformer.Semantics.Meaning where

open import Data.Nat using (ℕ; suc)
open import Data.Fin using (Fin)
open import Data.Product using (_,_)
open import Data.Vec.Base using (tabulate; lookup)

open import ModArTransformer.Tensor
open import ModArTransformer.Cat.Grad    using (eval)
open import ModArTransformer.Cat.VecPrim using (crossEntropyAtD)
open import ModArTransformer.Semantics.Interval
open import ModArTransformer.Semantics.Enriched
open import ModArTransformer.Semantics.Copresheaf
open import ModArTransformer.Semantics.Language

module MeaningOf (p : ℕ) where
  open Lang p

  -- softmax: turn a logit vector into a probability vector (numerically stable).
  -- (n = suc p, so a logit vector is always nonempty and `vmaxElement` applies.)
  softmax : ℝVec n → ℝVec n
  softmax logits =
    let mx   = vmaxElement logits          -- subtract the max for stability
        exps = vmap (λ z → fexp (z f- mx)) logits
        z    = vsum exps
    in  vmap (λ e → e f/ z) exps

  -- A probability vector *is* a [0,1]-copresheaf on the discrete VocabCat.
  asCopresheaf : ℝVec n → Copresheaf VocabCat
  asCopresheaf v i = lookup v i

  -- | ⟦ model ⟧ : the meaning of the model — each context denotes a copresheaf
  --   π(· | ctx).  `logits` is the model's forward pass with parameters baked in.
  ⟦_⟧ : (Context → ℝVec n) → Context → Copresheaf VocabCat
  ⟦ logits ⟧ ctx = asCopresheaf (softmax (logits ctx))

  -- Cross-entropy of a model copresheaf q against a target copresheaf t:
  --   − Σ_v  t(v) · log q(v).
  -- Against the one-hot `truth ctx` this is the relative entropy / training loss.
  crossEntropy : Copresheaf VocabCat → Copresheaf VocabCat → I
  crossEntropy t q =
    fneg (vsum (tabulate {n} (λ v → t v ⊗ᴵ flog (q v))))

  -- The semantic per-example loss is now the *forward projection of the very
  -- morphism the trainer differentiates* — `crossEntropyAtD (target a b)` from
  -- Cat.VecPrim, the same term `transformerLoss` is built from.  By the
  -- log-sum-exp identity its value equals `crossEntropy (truth ctx) (⟦logits⟧ ctx)`
  -- (the copresheaf cross-entropy above), so the denotational spec and the
  -- running loss are ONE expression, not two provably-equal ones.
  semanticLoss : (Context → ℝVec n) → Context → I
  semanticLoss logits (a , b) = eval (crossEntropyAtD (target a b)) (logits (a , b))
