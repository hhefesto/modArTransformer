# The denotational modular-arithmetic transformer
### What it is, why it's built this way, and an honest appraisal for Conal Elliott and Tai-Danae Bradley

This project learns `(a + b) mod n` with a one-layer transformer, but the point is
*how* it is specified and trained:

- **Meaning** is Tai-Danae Bradley's `[0,1]`-enriched-category semantics: the model's
  softmax output *is* a copresheaf (a hom-object of a learned language category), and
  the loss *is* the semantic distance to the ground-truth meaning.
- **Gradient descent** is Conal Elliott's "AD as a category": the backward pass is the
  *transpose* of the forward morphism, derived by the chain rule, with no hand-written
  backward code.
- Both are written first as a **type-checked Agda specification** (`ModArTransformer/`)
  and then transliterated to a fast **Haskell** trainer whose runtime engine is a
  **Wengert tape** (`backend/transformer/`).

The result reproduces grokking for `(a+b) mod 53` (test 99.6%) and learns `(a+b) mod 97`
to 100% test in ~29.4k steps / ~13 min on CPU.

---

## Part 1 — For Tai-Danae Bradley: the language model *is* enriched-categorical

The semantics layer follows *"An Enriched Category Theory of Language"* (2021) directly,
in type-checked Agda.

**The language category, enriched over `[0,1]`** (`Semantics/Enriched.agda`): objects are
expressions, and the hom-object is a conditional probability
`L(x, y) := π(y | x)`, with the enrichment laws `π(x|x)=1` and
`π(z|y)·π(y|x) ≤ π(z|x)` (identity and composition) recorded as `IdLaw`/`CompLaw`.

**Meaning as a copresheaf** (`Semantics/Copresheaf.agda`): the semantic category is
`L̂ = [0,1]^L`, and the representable copresheaf `hˣ = L(x,−)`, `hˣ(c) = π(c|x)`, is *the
meaning of `x`* — "the varying potential of all contexts in which `x` is used." The Yoneda
embedding `よ` is in the code, and the enriched-logic structure (products = AND, coproducts
= OR, internal hom = IMPLIES, pointwise in `I`) is named per Bradley Thms 2–3, Def. 12.

**The task as a Dirac copresheaf** (`Semantics/Language.agda`): for context `[a,b]` the only
correct continuation is `(a+b) mod n`, so the ground-truth meaning is the *Dirac* copresheaf
`π((a+b) mod n | [a,b]) = 1`, else `0`.

**The keystone** (`Semantics/Meaning.agda`): this file literally states the design —
> `⟦ model ⟧ ctx = softmax (logits θ ctx)` *is* a `[0,1]`-copresheaf on the vocab category,
> i.e. the model's softmax output is the hom-object `π(· | ctx)` — the learned meaning of
> the context — and training minimizes cross-entropy against the Dirac copresheaf
> `truth ctx`. Cross-entropy = relative entropy to a one-hot target = the Shannon (`t→1`)
> limit of Bradley's magnitude / Tsallis-entropy invariant. **So the loss is the semantic
> quantity, not an ad-hoc objective.**

There is also a tensor-network / MPS view (`Semantics/MPS.agda`, `Semantics/TensorNetwork.agda`)
and exact diagnostics that are kept green as a flake check.

**What she would likely appreciate**
- Softmax is *genuinely* treated as an enriched copresheaf, not analogized — the
  copresheaf, the representable/Yoneda structure, and the Dirac ground truth are all in the
  type-checked spec.
- The training objective is *derived from the semantics* (relative entropy to the Dirac
  meaning as the Shannon limit of the magnitude invariant), rather than bolted on.
- The enriched-logic operations (AND/OR/IMPLIES as products/coproducts/internal-hom in `I`)
  are present, so "meaning has logical structure" is concretely realized.

**Areas she would likely want improved (honest)**
- It is a *modular-arithmetic* language, not natural language — the "expressions" are
  two-token contexts and the only structure is `(a+b) mod n`. The framework's richness
  (compositionality of meaning across a real syntax category) is barely exercised.
- The enrichment **laws** (`IdLaw`/`CompLaw`) are recorded as predicates but **not proved**
  for the Float-valued learned model (the comment flags this as "future hardening"); only the
  ground-truth category satisfies them by construction.
- The copresheaf structure drives the *loss* but the *learned* representations are not yet
  shown to realize the enriched-categorical structure she theorizes (e.g. that learned
  `π(·|ctx)` respects composition); the MPS/tensor-network view is diagnostic, not trained.
- Magnitude / Tsallis-entropy at general `t` is referenced but only the `t→1` (Shannon) point
  is used in training.

---

## Part 2 — For Conal Elliott: differentiation is a functor, and we kept it honest

The gradient machinery follows *"The Simple Essence of Automatic Differentiation"* in
type-checked Agda, then is realized efficiently in Haskell.

**The category of derivatives** (`Cat/D.agda`, `Cat/Dual.agda`): `Dual AddFun` is "linear
maps with their arrows reversed" — `Cat/Dual.agda` cites *Simple Essence* §4.3 & Fig. 10 and
notes that the cartesian ops of `Dual k` are the *cocartesian* ops of `k`
(`exl ↦ inl`, `▵ ↦ join`, `dup ↦ jam`) — which is exactly why `Dual AddFun` is where
gradients flow backward. Primitives (`Cat/VecPrim.agda`) are `D`-morphisms each carrying its
transpose; **layers are compositions of these and contain no backward code of their own —
the gradient comes from the chain rule in `D`.** Extraction is one line (`Cat/Grad.agda`):
`gradient f a = applyL (unDual (proj₂ (runD f a))) 1.0` — run forward, feed the unit
cotangent `1` into the reversed linear map.

**The runtime: a Wengert tape as one faithful interpretation** (`backend/transformer/`). The
local adjoints in the Haskell trainer are *exactly* the `Dual`-category adjoints — `tMatvec`'s
backward is `(vouter dy x, matvec wᵀ dy)`, `tVdot`'s is `(d·b, d·a)`, etc. (`Tape.hs`). But
instead of composing them point-free, each intermediate is a node with a **mutable cotangent
cell** (`R { primalR, adjR }`); consumers add into it; `tGradLoss` seeds `dL/dL = 1` and runs
the recorded adjoint actions once in reverse-creation order. `AD.hs` still contains the pure
point-free `Dual` category (`D`, `forkD`, `▵`, `exl/exr`, `gradAndLoss`) as the conceptual
mirror of the Agda — but at runtime only its `Lens` projections are used; the tape is the
engine.

**What he would likely appreciate**
- The forward pass is a single morphism and the backward pass is *derived* — there is
  genuinely **no hand-written gradient** anywhere; every adjoint is a local transpose and the
  composite is the chain rule.
- The Agda `Cat/*` is a faithful, type-checked rendering of *Simple Essence* (`Dual AddFun`,
  cartesian↦cocartesian, `gradient` = pullback of `1`).
- The tape is presented honestly as **one interpretation** of the same morphism — the
  "different categories, same program" spirit — and the hmatrix/BLAS numerics sit *below* the
  categorical interface (BLAS is one model of the linear-map category).

**Areas he would likely want improved (honest)**
- The fast path is an **operational shortcut**: a mutable `ST` tape, not a pure categorical
  interpretation. We *abandoned* the elegant point-free `Dual` form for it — see the speed
  story below — so the runtime is "categorical adjoints, imperatively sequenced," not a
  category instance you could swap by type.
- **Compile-to-Categories did not pan out here.** We genuinely tried `toCcc` (Conal's plugin):
  forward/loss elaborated fine, and `ConCat.RAD.gradR` gave gradients for scalars/MLPs, but
  **reverse-mode of attention would not compile** at the `concat`/ghc948 pin (ticks-exhausted /
  16 GB OOM for a tiny block). So the "compile the morphism to a parallel/GPU target"
  pay-off — the thing that would make CTC worth it — was not realized; the tape wins in
  practice. (Full write-up in `GROKKING_PROGRESS.md`.)
- **Numerical stability diverges from the clean spec**: the Haskell loss/attention subtract a
  detached max (shift-invariant, so same value & gradient up to rounding), whereas the Agda
  `logSumExp`/`softmax` are naive. Deliberate, but a divergence.
- **Parallelism-as-an-interpretation is unrealized.** The only parallelism we got is plain
  batch data-parallelism (which needs no plugin); the graph-parallelism a categorical
  compilation could expose never materialized for this small CPU model.

---

## Part 3 — Why it's fast: the Wengert tape (the corrected speed story)

The naive way to run Conal's point-free `Dual` program is to compose `runD` over the whole
expression tree. That **recomputes shared sub-graphs**: the transformer's DAG has diamonds
(`e0 → Q/K/V` and the residual; the centered vector used twice in LayerNorm; `q0` in both
attention scores; the readout feeding `suc p` unembed dots), so a point-free interpreter
re-traverses each shared producer once per consumer. Measured cost: **~17.7 s/epoch**, ~224 GB
allocated over 3 epochs.

The **Wengert tape** (`Tape.hs`) fixes this. Every intermediate is created *once* as a node
with a mutable cotangent cell; multiple consumers register multiple adjoint actions that all
**add into the same cell**; the backward pass runs each node's action exactly once, in reverse
topological order, so a node's cotangent is fully accumulated before it propagates to its
parents. That is O(graph) forward and O(graph) backward, each node touched once — **no diamond
recomputation**. Measured cost: **~0.67 s/epoch** (a ~27× speedup), which is what makes
reaching grokking (thousands of epochs) practical on CPU.

The math is identical to the categorical version — the tape just *sequences* the same
`Dual`-category adjoints with sharing. (So: the tape is the speed win; "continuation style" is
**not** what makes it fast — the point-free/continuation form was the *slow* one we replaced.)

---

## Part 4 — Guarantees that the backend follows the Agda spec

Today:
- The Agda spec **type-checks** (a CI gate), so the categorical/semantic construction is
  well-formed.
- The Haskell gradient is **finite-difference-verified** (`transformer-gradcheck`, max abs err
  1.6e-11) — the tape's adjoints are numerically correct.
- The Haskell `Transformer.hs` is a documented **transliteration** of `Layers/Transformer.agda`.

Not yet (the missing guarantee, in progress): an **Agda↔Haskell conformance oracle** that
MAlonzo-evaluates the Agda spec and the Haskell backend on identical parameters/inputs and
checks that logits, loss, and gradient agree within tolerance. That is the next deliverable
(it will tolerance-compare, since the max-subtraction stability tweak is shift-invariant but
not bit-identical). Once green as a flake check, the backend is *guaranteed* to track the spec.

---

### One-paragraph summary to read aloud

"Meaning is a `[0,1]`-enriched copresheaf — the softmax output is literally the hom-object
`π(· | context)` of a learned language category, and the loss is its relative entropy to the
Dirac ground-truth meaning (Bradley). Its gradient is the transpose of the forward morphism in
`Dual AddFun`, derived by the chain rule with no hand-written backward (Elliott). Both are a
type-checked Agda spec; the Haskell trainer realizes the *same* adjoints on a Wengert tape so
each node is visited once (≈27× faster than the point-free form), and a conformance oracle is
being added so the fast path is provably faithful to the spec."
