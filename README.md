# The denotational transformer
### Tai-Danae Bradley's enriched-category language model × Conal Elliott's AD-as-categories, with a machine-checked spec↔implementation tie

*This is the project's single document. It absorbs the former `WALKTHROUGH.md`,
`EXPLANATION.md`, `CONFORMANCE.md`, `REVIEW.md`, `GROKKING_PROGRESS.md`, `PLAN.md`
and `SESSION_PROGRESS.md` (full chronological detail remains in their git history).
§10 is the production roadmap; §11 is the living progress log.*

---

## TL;DR

A transformer that learns **`(a + b) mod p`**, built as a **denotational design**:

- A **type-checked Agda specification** says what the model *means* — Tai-Danae
  Bradley's `[0,1]`-enriched category theory of language (the softmax output **is**
  the meaning of the context) — and how its gradient is *derived* — Conal Elliott's
  automatic-differentiation-as-categories (no hand-written backward pass anywhere).
- A **fast Haskell backend** (reverse-mode AD on a Wengert tape over hmatrix/BLAS) is
  a faithful transliteration of that spec, and it **reproduces grokking** (the
  memorize-then-generalize phenomenon of Power et al. 2022) on CPU in minutes.
- The backend is **tied to the spec by a numeric conformance oracle that runs in
  CI**: forward pass, loss, and gradient are identical to the Agda spec to machine
  precision (~1e-16).

Headline results: **p=97 reaches 100% test accuracy in ~29,400 steps (~13.2 min on
CPU, 1-layer; ~29.7 min, 2-layer)**; p=53 grokked to 99.6% test. All guarantees are
green in `nix flake check`.

The project is now being extended toward a **production language model over market
data** (quantized candle-return tokens from Hyperliquid via `chain-query`), keeping
the same discipline: spec first, implementation tied to it by conformance — see the
roadmap in **§10**.

---

## 1. The idea — a denotational design

**Meaning (Tai-Danae Bradley).** A language is a category *enriched over the unit
interval `[0,1]`*; the hom-object `L(x, y) = π(y | x)` is the probability that `y`
extends `x`. The transformer's softmax output **is** such a hom-object `π(· | ctx)` —
a representable copresheaf — so the model's output literally *is* the meaning of the
context. Training minimizes cross-entropy = **relative entropy** between that
copresheaf and the ground-truth Dirac copresheaf (all mass on `(a+b) mod p`).

**Gradient (Conal Elliott).** The forward pass is a single morphism in the category
`D (Dual AddFun)`; **every gradient is one chain rule** — the backward pass is the
*transpose* of the forward morphism, derived compositionally. There is no
hand-written backward for any layer.

Both are written first as type-checked Agda (`ModArTransformer/`), then realized in
a fast Haskell trainer whose engine is a Wengert tape (`backend/transformer/`).

### 1.1 For Tai-Danae Bradley: the language model *is* enriched-categorical

The semantics layer follows *"An Enriched Category Theory of Language"* (2021)
directly, in type-checked Agda:

- **The language category, enriched over `[0,1]`** (`Semantics/Enriched.agda`):
  objects are expressions, hom-objects are conditional probabilities
  `L(x, y) := π(y | x)`, with the enrichment laws `π(x|x)=1` and
  `π(z|y)·π(y|x) ≤ π(z|x)` recorded as `IdLaw`/`CompLaw`.
- **Meaning as a copresheaf** (`Semantics/Copresheaf.agda`): the semantic category
  is `L̂ = [0,1]^L`; the representable copresheaf `hˣ = L(x,−)` is *the meaning of
  `x`* — "the varying potential of all contexts in which `x` is used." The Yoneda
  embedding `よ` is in the code, and the enriched-logic structure (products = AND,
  coproducts = OR, internal hom = IMPLIES, pointwise in `I`) is named per Bradley
  Thms 2–3, Def. 12.
- **The task as a Dirac copresheaf** (`Semantics/Language.agda`): for context
  `[a,b]` the only correct continuation is `(a+b) mod n`, so the ground truth is the
  Dirac copresheaf.
- **The keystone** (`Semantics/Meaning.agda`): `⟦ model ⟧ ctx = softmax (logits θ
  ctx)` *is* a `[0,1]`-copresheaf on the vocab category, and training minimizes
  relative entropy against the Dirac copresheaf — **the loss is the semantic
  quantity, not an ad-hoc objective.** (Cross-entropy to a one-hot target is the
  Shannon `t→1` limit of Bradley's magnitude/Tsallis invariant — now spelled out in
  `Semantics/Magnitude.agda`.)
- **The market language** (`Semantics/MarketLanguage.agda`, new): per Bradley's 2025
  *Magnitude of Categories of Texts*, any autoregressive next-token model over a
  finite vocabulary with begin/end tokens (⊥, †) induces a `[0,1]`-enriched category
  of texts with `hom x y = π(y|x)` (product of next-token probabilities along the
  extension). The module constructs that category for an arbitrary next-token model,
  **proves its identity law**, states the composition law as an obligation, and
  gives the denotation of the market tokenizer (quantile binning of candle returns —
  monotone, total, surjective by construction).
- There is also a tensor-network / MPS view (`Semantics/MPS.agda`,
  `Semantics/TensorNetwork.agda`) kept green as a flake check.

Honest gaps she would point at: the original task is a two-token arithmetic
language, so compositionality across a real syntax category is barely exercised
(§10 addresses this — real sequences, every prefix a copresheaf); the enrichment
laws are recorded (identity now proved for the induced text category) but the
composition law is not yet proved for the Float-valued model; magnitude at general
`t` is specified but so far only the `t→1` point is used in training.

### 1.2 For Conal Elliott: differentiation is a functor, and we kept it honest

The gradient machinery follows *"The Simple Essence of Automatic Differentiation"*:

- **The category of derivatives** (`Cat/D.agda`, `Cat/Dual.agda`): `Dual AddFun` is
  linear maps with their arrows reversed (*Simple Essence* §4.3, Fig. 10); the
  cartesian ops of `Dual k` are the cocartesian ops of `k` (`exl ↦ inl`, `▵ ↦ join`,
  `dup ↦ jam`) — exactly why `Dual AddFun` is where gradients flow backward.
  Primitives (`Cat/VecPrim.agda`) are `D`-morphisms each carrying its transpose;
  **layers are compositions and contain no backward code of their own**. Extraction
  is one line (`Cat/Grad.agda`): run forward, feed the unit cotangent `1` into the
  reversed linear map.
- **The runtime: a Wengert tape as one faithful interpretation**
  (`backend/transformer/Tape.hs`). The local adjoints are *exactly* the
  `Dual`-category adjoints (`tMatvec`: `(vouter dy x, matvec wᵀ dy)`; `tVdot`:
  `(d·b, d·a)`; `tCenter` is self-adjoint; …). Each intermediate is a node with a
  mutable cotangent cell; consumers add into it; `tGradLoss` seeds `dL/dL = 1` and
  runs the recorded adjoint actions once in reverse creation order. `AD.hs` keeps
  the pure point-free `Dual` category as the conceptual mirror of the Agda; at
  runtime only its `Lens` projections are used — the tape is the engine.

Honest gaps he would point at: the fast path is an operational shortcut (mutable
`ST` tape, not a swappable category instance); numerical stabilizations
(detached-max softmax/loss) deliberately diverge from the naive spec
(shift-invariant — same value and gradient up to rounding); and
compile-to-categories did not pan out here (§8).

---

## 2. Repository map

| Area | Where | What |
|---|---|---|
| **Formal spec (Agda)** | `ModArTransformer/**` (38 modules) | Meaning + AD + layers + training, type-checked |
| **Fast backend (Haskell)** | `backend/transformer/*.hs` | Wengert-tape reverse-mode AD over hmatrix; trainers, benchmarks, probes |
| **Conformance oracle** | `modArConformanceOracle.agda` + `backend/transformer/Conformance.hs` | Agda↔Haskell numeric check, a CI flake check |
| **Diagram** | `diagram/index.html` | Standalone presentation-grade SVG (no build) |
| **Interactive frontend** | `frontend/` | Reflex → GHCJS port of the diagram |
| **Build** | `flake.nix` | Agda + backend builds, all CI checks/apps |

Agda layout:

```
ModArTransformer/
  Semantics/   -- the meaning (Tai-Danae): Interval, Enriched, Copresheaf, Language,
                  Meaning, MarketLanguage, Magnitude, MPS, TensorNetwork
  Cat/         -- the AD (Conal, on felix): Objects, Additive, AddFun, Dual, D, NumCat,
                  Grad, VecPrim, AdditiveTensor, Scale/Adamable/Serialize/Force
  Layers/      -- Linear, FFN, Attention, LayerNorm, Embedding, Transformer
  Tensor, Random, Data, Init, Train, Checkpoint, Optimizer/Schedule
modArTransformer.agda   -- entry point / training loop
```

Haskell executables (`modartransformer-backend.cabal`):

| executable | purpose |
|---|---|
| `backend-transformer-train` | the modular-arithmetic grokking trainer (modes p5…p97l2) |
| `transformer-gradcheck` | finite-difference gradient gate (CI) |
| `transformer-conformance` | Haskell half of the Agda↔backend oracle (CI) |
| `transformer-benchmarks` | Dyck-1 benchmark harness (fixed-length + OOD modes) |
| `enriched-probe` | reads a grokked checkpoint; measures Yoneda collapse + the ℤ/p circle |
| `synth-lang-probe` | trains on a synthetic ℤ/12 rotation language; verifies the composition law |

---

## 3. Why it's fast — the Wengert tape (the corrected speed story)

The naive way to run the point-free `Dual` program composes `runD` over the whole
expression tree, which **recomputes shared sub-graphs**: the transformer's DAG has
diamonds (`e0 → Q/K/V` + residual; the centered vector twice in LayerNorm; `q0` in
both attention scores; the readout feeding every unembed dot). Measured:
**~17.7 s/epoch**, ~224 GB allocated over 3 epochs.

The tape fixes this: every intermediate is created *once* as a node with a mutable
cotangent cell; multiple consumers add into the same cell; backprop runs each node's
action exactly once, in reverse-topological (reverse-creation) order. O(graph)
forward and backward, no diamond recomputation. Measured: **~0.67 s/epoch — a ~27×
speedup** — which is what makes grokking-scale training practical on CPU. The math
is identical to the categorical form; the tape just *sequences* the same adjoints
with sharing. (So: the tape is the speed win; "continuation style" was the slow form
it replaced.)

Later additions: example-level **batch parallelism** (`parMap rdeepseq`; the
cotangent monoid `addA` is associative and order is preserved, so results are
numerically identical; ~1.4× on training, near-linear on eval sweeps) with BLAS
pinned to one thread so parallelism lives at the example level.

---

## 4. Guarantees — how the backend is tied to the spec

A ladder of guarantees, all green in `nix flake check`:

1. **The spec type-checks** — `agda-modArTransformer-check` (the enriched-category
   semantics, the AD category, the layers). Raw constructions accepted by Agda;
   semantic laws recorded as obligations are not thereby proved.
2. **The gradient is numerically correct** — `transformer-gradcheck` compares the
   backend's reverse-mode gradient to central finite differences: **max abs err
   1.65e-11**. Because the backward is *derived* (composed adjoints), this verifies
   the chain-rule machinery the spec prescribes.
3. **Serialization is aligned** — Agda and Haskell agree leaf-by-leaf on the flat
   parameter layout (row-major matrices; Lin = (W, b); LN = (γ, β); FFN = (up,
   down); attention flattens Wq, Wk, Wv, Wo; whole tree = tokEmbed, posEmbed, Attn,
   LN1, FFN, LN2, unembed), so one `[Float]` denotes the same model on both sides.
4. **The numeric oracle** — `checks.conformance` feeds *identical* params to the
   Agda spec (via MAlonzo) and the Haskell backend and tolerance-diffs the outputs:

   | quantity | max \|Δ\| (Agda vs Haskell) |
   |---|---|
   | logits (forward) | **1.1e-16** |
   | loss             | **1.1e-16** |
   | gradient         | **2.3e-16** |

   The check gates CI at `max|Δ| ≤ 1e-9` (tolerance, not bit-equality, because the
   backend's shift-invariant max-stabilizations deliberately diverge from the naive
   spec). A historical find: the oracle originally exposed a ~0.33 gradient delta —
   the Agda LayerNorm pullback needed the centered expression scaled as a whole by
   `invStd`; after the fix, machine-precision agreement.

### Scope of the guarantee — what "follows the Agda spec" means

The conformance guarantee is about the **denotational model**, *not* the training
procedure:

- **Verified surface:** the model's forward pass, loss, and gradient as pure
  functions of `(params, input)`.
- **Out of scope (intentionally not bit-faithful):** the training harness. Known
  divergences:

  | aspect | Agda spec | Haskell backend |
  |---|---|---|
  | train/test split | even/odd by position, deterministic | shuffle then halve, RNG-dependent |
  | shuffle | front-swap Fisher–Yates on `Vec` | selection shuffle on a list |
  | RNG | legacy L'Ecuyer LCG | SplitMix (`random-1.2.1.3`) |
  | init | same Xavier formula/leaf order, different RNG ⇒ different numbers | — |
  | LR schedule | keyed on epoch, warmup minLR→base | keyed on global step, warmup 0→base |
  | optimizer (AdamW) | `Cat/Adamable.agda` | historical `Main.hs@62b0b4d` (eps-placement/warmup differ) |
  | checkpoint | params-only; resets Adam+epoch | header + params + Adam m/v; full resume |

This split is **by design**: the spec's value is proving the *model + AD* correct;
the training recipe is engineering. A fresh Agda run and a fresh Haskell run from
the same seed share the model, not the trajectory. (The "shuffle lesson" that
produced this boundary: an O(n) shuffle "optimization" silently changed the
seed→split mapping and slowed grokking; it was reverted, and the scope boundary was
documented.)

One more caveat: the oracle gates one fixed model shape. New harnesses
(benchmarks, probes, future sequence models) are experimental until deliberately
pulled into the oracle.

---

## 5. Results — grokking reproduced

Reproducing Power et al. 2022's *grokking* (delayed generalization on small
algorithmic datasets) was the empirical goal.

| run | modulus | layers | result | steps | wall-clock (CPU) |
|---|---|---|---|---|---|
| sanity | p=5 | 1 | 100% train (overfits) | — | seconds |
| grok | p=53 | 1 | **99.6% test** | — | overnight |
| `p97hi` | p=97 | 1 | **100% test** | **29,400** | **~13.2 min** |
| `p97l2` | p=97 | 2 | **99.7% test** | **29,400** | **~29.7 min** |

The canonical p=53 grokking curve (wd=1e-2, flat lr=1e-3, batch 32, split 0.5):

| epoch | train | test | phase |
|------:|------:|-----:|-------|
| 100  | 100% | 1.2%  | memorized training set |
| 2000 | 100% | 4.4%  | end of memorization plateau (test ≈ chance 1/53) |
| 2100 | 100% | 18.6% | **sharp grokking onset** |
| 3000 | 100% | 65.5% | S-curve |
| 4000 | 100% | 94.2% | generalizing |
| 4200 | 100% | **99.6%** | grokked |

**Weight decay is the demonstrated grokking lever**: wd=1e-2 groks by epoch ~4200
while the otherwise-identical wd=1e-3 run was still at test ~3.4% past epoch 7000 —
consistent with the historical note that wd was essential.

The 2-layer p=97 model (112,609 params vs 62,625) shows the more gradual,
paper-like curve: 39.4% test at epoch 100 → 99.7% at epoch 200, vs the 1-layer's
near-immediate climb. Both grok at the same *step* count; the 2-layer costs ~2.25×
wall-clock.

### Comparison to the grokking paper (Power et al. 2022)

| | Power et al. 2022 | This implementation |
|---|---|---|
| Task | `(a ∘ b) mod p`, incl. addition | `(a + b) mod p` |
| Modulus | p = 97 | p = 97 (also 5, 53) |
| Split | 50 / 50 | 50 / 50 (4,704 train pairs at p=97) |
| Architecture | 2-layer decoder, multi-head, width ~128 *(approx.)* | 1- or 2-layer, single head, dM 64 / dFF 256 / dK 64 |
| Params | ~4×10⁵ *(approx.)* | 62,625 / 112,609 |
| Optimizer | AdamW, wd ≈ 1, batch 512 *(approx.)* | AdamW, wd = 1e-2, batch 32 |
| Steps to generalize | onset ~10⁵, full demos ~10⁶ *(approx.)* | ~2.9×10⁴ |
| Hardware | GPU | CPU, minutes |

**Honest framing.** Step counts are not directly comparable (batch 32 sees 16× less
data per step than batch 512), and at a 50/50 split we land in the
*fast-generalization* regime rather than the paper's dramatic delayed grok (which
needs a smaller training fraction). Same task, same modulus, same
memorize-then-generalize signature under weight decay — a faithful reproduction of
the *phenomenon* at small scale, not a bit-for-bit replication of the curves.

### The enriched-semantics showcase (probes)

- **`enriched-probe`** (reads the grokked p97l2 checkpoint, read-only): Yoneda
  copresheaf collapse — synonymous contexts (same `(a+b) mod p`) have intra-class
  KL ~1900× smaller than inter-class; PCA of the pre-unembed "meaning"
  representations recovers the **ℤ/97 Fourier circle** (the clock algorithm of
  Nanda et al. 2023), dominant DFT frequency explaining the geometry, var-explained
  0.87. Writes `enriched-modp-circle.csv`.
- **`synth-lang-probe`**: a synthetic ℤ/12 rotation-walk language with ground-truth
  enriched structure; the trained model's copresheaves verify the **group-action
  composition law** (rotation residual 0.011; DFT k=1 at 99.9%).

What's reproduction vs new: the grokking phenomenon (Power et al.) and the Fourier
circle (Nanda et al.) are reproductions; the **quantitative bridge to Bradley's
enriched-category semantics** (copresheaf collapse, composition law, meaning-as-
copresheaf measured on a real trained model) is the novel part.

---

## 6. Backend code review (condensed)

Verdict: solid, correct, small (~1.1k LOC core). No correctness blockers.

- **`Tape.hs`** — the heart, clean (see §3). Every primitive's local adjoint
  verified by gradcheck.
- **`AD.hs`** — full Conal `Dual` category; at runtime only its `Lens` part is used.
  Kept deliberately as the conceptual reference mirroring the Agda `Cat/*`.
- **`Transformer.hs`** — the model as a straight-line tape program, transliterated
  from `Layers/Transformer.agda`; 1- and 2-layer variants; stabilizations documented
  as shift-invariant divergences.
- **`Optimizer.hs`** — decoupled AdamW (matrix-only weight decay), warmup→cosine
  keyed on global step; transcribed from the historical grokking success
  `Main.hs@62b0b4d`, not from `Cat/Adamable.agda` (documented divergence). The
  shuffle is intentionally the O(n²) seed-preserving one (see the shuffle lesson,
  §4).
- **`Tensor.hs`** — shape-indexed `V n`/`M m n` over hmatrix with phantom Nats;
  BLAS sits *below* the categorical interface.
- **`Serialize.hs`** — generic flat `[Double]` layout shared leaf-for-leaf with
  `Cat/Serialize.agda` (the precondition of the oracle).
- **`Checkpoint.hs`** — shared atomic save (`.tmp` + rename), validated load
  (float count = 3×nParam), tagged `CKPT` header; used by all trainers; all modes
  also have an interactive `--prompt` REPL constrained to their vocabulary and
  `--list-modes`.

---

## 7. How to run

```bash
# dev shell (Agda + GHC + deps; pins BLAS to 1 thread)
nix develop

# train modular arithmetic (default mode p97l2 — 2-layer, p=97)
cabal run backend-transformer-train -- --help
cabal run backend-transformer-train -- -m p97l2 -e 200 -s 36 +RTS -N

# talk to a trained model (REPL constrained to the mode's vocabulary)
cabal run backend-transformer-train -- -m p97l2 -c checkpoint-p97l2.ckpt --prompt

# Dyck-1 benchmark harness (fixed-length and OOD length-generalization modes)
cabal run transformer-benchmarks -- -m dyck12-2 -e 20
cabal run transformer-benchmarks -- -m dyck-ood-12-16-2 -e 500 --eval-every 10

# enriched-semantics probes
cabal run enriched-probe -- -c checkpoint-p97l2.ckpt
cabal run synth-lang-probe -- -e 2000

# every guarantee (Agda type-check + gradcheck + conformance oracle)
nix flake check

# type-check a single Agda module locally
agda --no-default-libraries -i . ModArTransformer/Semantics/MarketLanguage.agda

# Agda spec build/run
nix build                  # native binary → result/bin/agda-modArTransformer
nix run                    # train via the Agda/MAlonzo path

# the interactive diagram
nix run                    # (in frontend/) — or just open diagram/index.html
```

Toolchain: **GHC 9.10.3**, Agda 2.8.0 (+ felix), `random-1.2.1.3` (SplitMix),
frontend on ghcjs-8.10.7. The flake precompiles the Agda stdlib and felix
interfaces so type-checks don't re-check the libraries.

---

## 8. Engineering history (incl. the compile-to-categories verdict)

- **Wengert tape replaces point-free Dual/Cont** — the ~27× story of §3.
  Gradcheck held at 1.65e-11 through every rewrite.
- **Grokking reproduced** (2026-06-02, p=53 → 99.6%; then p=97 variants — §5).
- **Compile-to-Categories (CTC): explored thoroughly, then concluded.** Conal's
  `concat` plugin (`toCcc`, at a ghc948 pin) elaborated a full forward/loss ladder
  (scalars → dot → matvec → softmax → NLL → tiny MLP → two-key attention → mini
  transformer-block NLL) with no `Double#` panic; reverse mode worked for
  tuple-shaped models via `ConCat.RAD.gradR` (avoiding the `LinearRow` `Pointed`
  simplifier loop), including **parallel chunked training** of an MLP and of softmax
  self-attention. But the **full transformer block would not compile in reverse
  mode** (simplifier-ticks exhausted ~3.9M / 16 GB RSS runaways; the Q/K/V/O matvecs
  and sqrt-LayerNorms are the amplifiers), consistent with ConCat's own
  `Vector`-based tests being commented out as failing. And where CTC compiled, the
  speedup was plain data parallelism — achievable on the tape without the plugin,
  with no per-step FLOP win. **Verdict: not beneficial for this small-model/CPU
  task; the tape/hmatrix backend is the production trainer.** The `ctc-*` packages
  were removed in consolidation (history in git).
- **CLI** — `optparse-applicative` flags everywhere (`-m/--mode`, `-e/--epochs`,
  `-s/--seed`, `-c/--checkpoint`, `--resume`, `--prompt`, `--list-modes`).
- **Batch parallelism** (+~1.4× train, near-linear eval) with BLAS pinned to 1
  thread; **hardened checkpoints** (atomic save, validating load) shared by all
  trainers; **the shuffle lesson** (§4).
- **Checkpoint + prompt REPLs for every mode**; best-by-test checkpoints
  (`PATH.best`) for the benchmark harness.

---

## 9. Honest limitations

- **seqLen = 2, single head** — the architecture is specialized to two-token
  contexts; §10 generalizes it.
- **Training harness not bit-faithful** to the spec (by design; §4 table).
- **Benchmark harness not formalized** — the Dyck modes are outside the oracle.
- **Toy task** — modular arithmetic, not natural language; the probes (§5) show the
  learned model *realizes* enriched structure, but the enrichment laws are recorded,
  not proved, for the learned Float model.
- **CPU only** — hmatrix/BLAS; production scale needs the GPU tier of §10.

---

## 10. Roadmap — the production market-LLM plan

Goal: train a production next-token model on market data (Hyperliquid via
[`chain-query`](https://github.com/hhefesto/chain-query)) for market prediction,
keeping this project's discipline end to end:

- the **Agda spec stays the guide** (extended first, implementation conforms);
- **gradient descent stays Conal's**: adjoints are the categorical `Dual`
  pullbacks composed by the chain rule — *no opaque autograd anywhere*; GPU
  libraries supply kernels only;
- the model is **Bradley-faithful**: per her 2025 magnitude paper, a causal
  next-token transformer over a finite vocabulary with ⊥/† *is* the enriched
  category of texts; market data enters as a quantized-candle-return token
  language; magnitude/Tsallis is the evaluation invariant.

Decisions fixed (2026-06-09): hardware = local NVIDIA GPU; fast tier = categorical
AD over libtorch kernels (hasktorch, *kernels only*); tokenization = quantized
candle returns (~512-token vocabulary incl. ⊥/†).

**Phase 0 — the denotation first (Agda).**
`Semantics/MarketLanguage.agda` (tokenizer denotation + the model-induced enriched
category of texts), `Semantics/Magnitude.agda` (Mag(tM), Tsallis/Shannon);
generalized n-position causal multi-head model in `Cat/D` + `Layers/*` at tiny
oracle dims (v=5, dM=4, dF=8, dK=2, h=2, n=4).

**Phase 1 — generalized reference implementation (CPU tape).**
New tape primitives with pullback adjoints (`tMatmul`, `tSoftmaxRowsMasked` with
causal mask + detached row max, row pack/unpack, head split/concat); `MHAttn`,
`blockSeqT`, `forwardSeq`, `seqGradLoss` (mean next-token relative entropy over all
prefixes) in `Transformer.hs`; existing seqLen-2 paths untouched. Gates: FD
gradcheck on every new primitive + the seq model; n=2/h=1 degeneracy vs
`forwardT2` ≤1e-12; the conformance oracle extended to the generalized model.

**Phase 2 — GPU tier: categorical AD over CUDA kernels.**
`backend/gpu/{TensorG,TapeG,TransformerG,OptimizerG}.hs`: the same tape design with
batched-tensor nodes (one node = one CUDA kernel ⇒ bookkeeping amortized), adjoints
written per batched primitive, **no torch autograd in the training path**.
Conformance chain: GPU tier vs CPU tape ≤1e-9 in float64 ⇒ transitively vs Agda.
Riskiest step is the hasktorch+CUDA nix pin (timeboxed; fallback to raw
`libtorch-ffi`).

**Phase 3 — data pipeline.**
In `chain-query`: historical backfill via the documented-but-unbuilt
`candleSnapshot` REST endpoint (1-minute candles), durable resumable live ingestion
(persist max `tid`, schema-validated appends), live trade→candle aggregator. Here:
`backend/data/MarketTokenizer.hs` realizing the Phase-0 denotation (per-asset
normalized log-returns, quantile bin edges **fitted on the training window only** —
the leakage rule — saved as an artifact; decode = bin midpoints), walk-forward
dataset builder, property tests (monotonicity, round-trip, leak check).

**Phase 4 — production training harness** (engineering tier, out of spec scope).
Binary checkpoint format behind the same `CKPT` header (text format dies at 13M
params); `market-train` executable in the house CLI style with GPU batches, grad
clipping, best-by-val checkpoints, and a `--prompt` REPL over the token vocabulary;
first model ~13M params (4 layers, dM=256, h=8, dK=32, dF=1024, n=256, v=512);
walk-forward evaluation (perplexity vs unconditional baseline, directional
accuracy, calibration, cost-aware backtest).

**Phase 5 — the Bradley evaluation layer.**
Extend the probe pattern to sequences: magnitude function Mag(tM) over sampled
prompts (slope at t=1 = mean Shannon entropy), Yoneda copresheaf KL between market
contexts, and the composition inequality `π(z|y)·π(y|x) ≤ π(z|x)` as an empirical
model diagnostic.

Ordering: 0 → 1 sequential; 2 and 3 independent (both need 1); 4 needs 2+3;
5 needs 4. Merge gate at every phase: `nix flake check` green.

---

## 11. Progress log (living)

- **2026-06-09** — Roadmap (§10) adopted after a full project review and surveys of
  Conal Elliott's corpus, Tai-Danae Bradley's papers (the 2025 magnitude paper is
  the bridge to market data), and `chain-query` (live Hyperliquid trade streamer;
  needs backfill + durability for training use). Decisions: local NVIDIA GPU;
  categorical-AD-over-libtorch fast tier; quantized candle-return tokens.
- **2026-06-09** — **Phase 0 (semantics half) done**:
  `Semantics/MarketLanguage.agda` (tokenizer denotation; the model-induced enriched
  category of texts; identity law proved, composition law stated) and
  `Semantics/Magnitude.agda` (Tsallis/Shannon, Mag(tM), slope-at-1) both
  **type-check**. Docs consolidated into this single README.
- **2026-06-09** — **Phase 0 (model half) + Phase 1 done.** The generalized
  model exists on BOTH sides and is oracle-gated:
  - Agda: `Cat/SeqPrim.agda` (linear pack/concat primitives + the masked n-ary
    softmax generalizing `softmax2D`) and `Layers/SeqTransformer.agda` (n
    positions, causal, two heads, mean next-token relative-entropy loss over
    all prefixes) — type-check.
  - Haskell: `tConcatV`/`tEvalMany` on the tape, `vconcatT`/`vsplitT` in
    Tensor, and `ParamsSeq`/`forwardSeqT`/`seqLogitsVal`/`seqGradLoss` in
    `Transformer.hs`, with the parameter tree nested exactly like the Agda
    side (Serialize alignment preserved).
  - Gates: gradcheck now checks BOTH models (sequence model FD max err
    1.39e-11); the conformance oracle gained a second section — **both models
    agree Agda↔Haskell at max|Δ| = 4.4e-16 over 1141 floats**
    (`checks.conformance` green).
- **2026-06-09** — **Phase 3 (tokenizer half) done**:
  `backend/data/MarketTokenizer.hs` realizes the Agda quantizer denotation
  verbatim (same `quantize` recursion, same ⊥/†/bin vocabulary layout),
  quantile edges fitted on the train window only, per-bin median decoder,
  walk-forward splitter + context windows, spec artifact I/O. New
  `market-tokenizer-check` executable: 10 property checks PASS (monotonicity,
  totality, decode coherence, no-lookahead with a power check, sorted edges,
  artifact round-trip, windowing).
- **2026-06-09** — **Phase 3 (data half) done**: `chain-backfill` in
  chain-query pulls paginated 1m OHLCV history from Hyperliquid's
  `candleSnapshot` (the endpoint returns the *latest* ~5000 in a range, so
  pagination is forward-windowed), validated rows, **idempotent top-up**
  (re-running appends only newer candles — the durable candle-ingestion path).
  Verified: 1 day of BTC = 1441 gapless 1m candles; second run appends 0.
- **2026-06-09** — **Phase 4 done**: `Checkpoint.hs` now saves a BINARY format
  (`CKPTB` header + raw little-endian doubles; legacy text formats still
  load); new `market-train` executable trains the conformance-gated
  architecture on real candles (walk-forward split, train-window-only
  tokenizer artifact saved as `<ckpt>.tok`, AdamW + warmup-cosine + global-norm
  clip, parallel batch gradients, best-by-val `PATH.best`, `--resume`,
  `--prompt` REPL that encodes raw returns to bins and decodes the predicted
  distribution to an expected next return, `--list-modes`). Verified
  end-to-end on the backfilled BTC data (mkt-small, ~5.4 s/epoch CPU; resume
  continues optimizer state; prompt round-trips).
- **2026-06-09** — **Phase 5 done**: `market-seq-probe` (read-only) computes
  the magnitude function's prompt term at several t with slope-at-1 = total
  Shannon entropy (mean H 4.758 vs uniform 4.868 on a 5-epoch checkpoint),
  Yoneda synonymy (contexts sharing their last bin have **33× closer
  copresheaves**; last two bins, 49×), and verifies the composition-law
  equality numerically (max deviation 3e-61 — the chain rule; Bradley's
  "autoregressive models are enriched categories by construction").
  Writes `market-magnitude.csv`. PROBE OK.
- **2026-06-09** — **Live mode (`market-live`) + local data pipeline**: the
  Hyperliquid client is vendored into `backend/hyperliquid/` (origin:
  chain-query@38a2f7c), so the whole pipeline is local — `market-backfill`
  pulls/tops-up candle CSVs, and `market-live` streams trades in-process.
  market-live warm-starts from a market-train checkpoint (+ .tok), aggregates
  rolling 1m candles, takes **one online gradient step per candle close** (the
  verified `seqGradLoss` + AdamW, constant lr, clip; checkpoint saved every
  `--save-every` candles) and emits **a recommendation on every trade**
  (provisional last token → copresheaf → expected return μ, P(up), entropy →
  FLAT/LONG/SHORT paper state machine vs `--threshold-bps` net of
  `--fee-bps` → OPEN_LONG/OPEN_SHORT/CLOSE/FLIP/HOLD as JSONL; position
  persists across restarts).  `--replay CSV` drives the identical pipeline
  offline (fee-aware paper backtest).  Verified: replay over the BTC file
  (1527 candles, per-close online loss; θ=0.5bp exercises the state machine —
  43 opens/42 closes, PnL accounting consistent) and a live smoke against the
  real market (2 live candle closes trained; position resumed across
  restart).  Recommendations only — no order placement.
- **2026-06-09** — **Data reality check**: Hyperliquid's `candleSnapshot`
  retains only **~5000 candles per interval** (1m ≈ 3.6 days, 15m ≈ 52 days,
  1h ≈ 208 days) — a 180-day 1m pull is impossible from REST alone.  Archives
  on disk: BTC 1m/15m/1h at the full available depth; the idempotent top-up
  (cron `market-backfill`) accumulates 1m history beyond the API's retention
  going forward.  Until the 1m archive deepens, offline training should use
  15m/1h corpora (the live mode's online learning is unaffected).
- **2026-06-09** — **Phase 2 foundation done, CUDA pending hardware**: `gpu/`
  is a separate flake (hasktorch pin, fully cache-fetched) whose
  `ConformanceG.hs` implements the sequence model with **our categorical
  reverse-mode AD over libtorch kernels** — same tape design, same
  Dual-pullback adjoints, no torch autograd — in float64 CPU mode.
  **GPU tier vs CPU tape: max|Δ| = 2.2e-16 over 508 floats** ⇒ transitively
  conformant to the Agda spec. The three-tier chain (Agda ↔ tape ↔ libtorch)
  is now real. Remaining for Phase 2 (needs the NVIDIA machine — this one has
  no GPU): flip the flake to the CUDA flavor, batch the nodes
  (`B×n×d` tensors, one kernel per node), and port `market-train` onto it.

---

## 12. References

- Tai-Danae Bradley, John Terilla, Yiannis Vlassopoulos. *An Enriched Category
  Theory of Language: From Syntax to Semantics* (2021). arXiv:2106.07890.
- Tai-Danae Bradley. *Language Modeling with Reduced Densities* (2020); *At the
  Interface of Algebra and Statistics* (PhD thesis, 2020); *The Magnitude of
  Categories of Texts Enriched by Language Models* (2025).
- Conal Elliott. *The Simple Essence of Automatic Differentiation*. ICFP 2018.
  arXiv:1804.00746. *Compiling to Categories*. ICFP 2017. And the `felix` Agda
  library.
- Power, Burda, Edwards, Babuschkin, Misra. *Grokking: Generalization Beyond
  Overfitting on Small Algorithmic Datasets*. arXiv:2201.02177 (2022).
- Nanda, Chan, Lieberum, Smith, Steinhardt. *Progress measures for grokking via
  mechanistic interpretability* (2023) — the Fourier-circle mechanism.
