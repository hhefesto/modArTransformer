# Modular Arithmetic Transformer — denotational (Agda)

A transformer that learns `(a + b) mod p`, written as a **denotational design** in Agda:

- **Meaning** — Tai-Danae Bradley's *enriched category theory of language*: a language is a
  category enriched over the unit interval `[0,1]`, with hom-objects `L(x,y) = π(y|x)`
  (the probability that `y` extends `x`). The model's softmax output **is** such a hom-object
  `π(·|ctx)` — a copresheaf — so `⟦θ⟧ ctx` *is* the meaning of the context.
- **Tooling** — Conal Elliott's *automatic differentiation as categories* (built on the
  [`felix`](https://github.com/conal/felix) library): the forward pass is a morphism in
  `D (Dual AddFun)`, and **every gradient is derived by one chain rule** — there is no
  hand-written backward pass anywhere.

Training minimizes cross-entropy = relative entropy between `⟦θ⟧ ctx` and the ground-truth
Dirac copresheaf `truth ctx` (mass 1 on `(a+b) mod p`).

## Layout

```
ModArTransformer/
  Semantics/   -- the meaning (Tai-Danae): Interval, Enriched, Copresheaf, Language, Meaning
  Cat/         -- the AD (Conal, on felix): Objects, Additive, AddFun, Dual, D, NumCat, Grad,
                  VecPrim, AdditiveTensor, and the generic param infra Scale/Adamable/Serialize/Force
  Layers/      -- Linear, FFN, Attention, LayerNorm, Embedding, Transformer — pure compositions
  Tensor, Random, Data, Init, Train, Checkpoint, Optimizer/Schedule
modArTransformer.agda   -- entry point / training loop
```

`Semantics/*` type-checks as a standalone denotational spec; the runnable program is the
`Cat/`+`Layers/` AD implementation it is the meaning of.

## Build / run (Nix)

```bash
nix develop                 # agda preloaded with stdlib + felix interfaces
agda modArTransformer.agda            # type-check (fast)
agda --compile modArTransformer.agda  # compile to a native binary (MAlonzo → GHC)

nix build                              # native binary  → result/bin/agda-modArTransformer
nix build .#agda-modArTransformer-check   # type-check gate only (CI)
nix run                                # train: prints `epoch | loss …`, writes checkpoint.ckpt
```

The flake precompiles the Agda standard-library and felix interfaces once (cached in the Nix
store) so type-checks and builds don't re-check the libraries every time.

`scripts/agda-guard.sh` optionally caps memory for long runs; the virtual-memory cap is opt-in
via `AGDA_GUARD_VMEM_PCT` (default off, since it throttles GHC).

## References

- Tai-Danae Bradley, John Terilla, Yiannis Vlassopoulos. *An Enriched Category Theory of
  Language: From Syntax to Semantics* (2021). https://arxiv.org/abs/2106.07890
- Tai-Danae Bradley. *Language Modeling with Reduced Densities* (2020).
- Conal Elliott. *The Simple Essence of Automatic Differentiation*. ICFP 2018.
  https://arxiv.org/abs/1804.00746
- Conal Elliott. *Compiling to Categories*. ICFP 2017. — and the `felix` Agda library.
- Alethea Power et al. *Grokking: Generalization Beyond Overfitting on Small Algorithmic
  Datasets*. arXiv:2201.02177 (2022).
