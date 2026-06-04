# Walkthrough — what this project is and what we've achieved

A start-here tour of the whole project. It links to the deeper docs rather than
repeating them; read this first, then dive into whichever detailed doc you need.

---

## TL;DR

A transformer that learns **`(a + b) mod p`**, built as a **denotational design**:

- A **type-checked Agda specification** says what the model *means* — Tai-Danae
  Bradley's `[0,1]`-enriched category theory of language (the softmax output **is** the
  meaning of the context) — and how its gradient is *derived* — Conal Elliott's
  automatic-differentiation-as-categories (no hand-written backward pass anywhere).
- A **fast Haskell backend** (reverse-mode AD on a Wengert tape over hmatrix/BLAS) is a
  faithful transliteration of that spec, and it **reproduces grokking** (the
  memorize-then-generalize phenomenon from Power et al. 2022) on CPU in minutes.
- The backend is **tied to the spec by a numeric conformance oracle that runs in CI**:
  the forward pass and loss are provably identical to the Agda spec to machine precision
  (~1e-16).

Headline results: **p=97 reaches 100% test accuracy in ~29,400 steps (~13.2 min on CPU,
1-layer; ~29.7 min, 2-layer)**; p=53 grokked to 99.6% test. All guarantees are green in
`nix flake check`.

---

## 1. The idea — a denotational design

The project is organized around two pieces of theory, each realized as type-checked Agda
and then implemented in Haskell.

**Meaning (Tai-Danae Bradley).** A language is a category *enriched over the unit
interval `[0,1]`*; the hom-object `L(x, y) = π(y | x)` is the probability that `y`
extends `x`. The transformer's softmax output **is** such a hom-object `π(· | ctx)` — a
copresheaf — so the model's output literally *is* the meaning of the context. Training
minimizes cross-entropy = **relative entropy** between that copresheaf and the
ground-truth Dirac copresheaf (all mass on `(a+b) mod p`).

**Gradient (Conal Elliott).** The forward pass is a single morphism in the category
`D (Dual AddFun)`; **every gradient is one chain rule** — the backward pass is the
*transpose* of the forward morphism, derived compositionally. There is no hand-written
backward for any layer.

→ The full story, written to be read by Conal Elliott and Tai-Danae Bradley themselves,
is in **[EXPLANATION.md](EXPLANATION.md)**. The Agda-spec quickstart is in
**[README.md](README.md)**.

---

## 2. Repository map

| Area | Where | What |
|---|---|---|
| **Formal spec (Agda)** | `ModArTransformer/**` (36 modules) | The denotational meaning + AD + layers + training, type-checked |
| **Fast backend (Haskell)** | `backend/transformer/*.hs` (9 files) | Wengert-tape reverse-mode AD over hmatrix; the production trainer |
| **Conformance oracle** | `modArConformanceOracle.agda` + `backend/transformer/Conformance.hs` | Agda↔Haskell numeric check, wired as a CI flake check |
| **Diagram** | `diagram/index.html` | Standalone presentation-grade SVG (no build) |
| **Interactive frontend** | `frontend/` | Reflex → GHCJS port of the diagram |
| **Build** | `flake.nix` | Agda + backend builds, and all CI checks/apps |

Which doc to read for what:

| Question | Doc |
|---|---|
| What is this, conceptually? | this file, then [EXPLANATION.md](EXPLANATION.md) |
| How do I build/run the Agda spec? | [README.md](README.md) |
| Is the backend faithful to the spec? | [CONFORMANCE.md](CONFORMANCE.md) |
| Is the backend code any good? | [REVIEW.md](REVIEW.md) |
| What were the training results / history? | [GROKKING_PROGRESS.md](GROKKING_PROGRESS.md) |

---

## 3. The formal specification (Agda)

The entire development type-checks (`agda-modArTransformer-check`, a CI gate):

- `Semantics/*` — the meaning layer: `Interval`, `Enriched`, `Copresheaf`, `Language`,
  `Meaning`, plus `MPS` / `TensorNetwork` (the matrix-product-state / tensor-network
  view). This is a standalone denotational spec.
- `Cat/*` — the AD: `Objects`, `Additive`, `AddFun`, `Dual`, `D`, `NumCat`, `Grad`,
  `VecPrim`, and the generic parameter infrastructure (`Scale`, `Adamable`, `Serialize`,
  `Force`). This mirrors *The Simple Essence of Automatic Differentiation*, built on
  Conal's `felix` library.
- `Layers/*` — `Linear`, `FFN`, `Attention`, `LayerNorm`, `Embedding`, `Transformer`:
  pure compositions, the program the `Semantics/*` layer is the meaning of.
- Runnable training modules — `Data`, `Random`, `Init`, `Train`, `Checkpoint`,
  `Optimizer/Schedule` — and the entry point `modArTransformer.agda`.

---

## 4. The fast backend (Haskell)

`backend/transformer/*.hs` is a hand transliteration of the Agda spec, built for speed:

- **`Tape.hs`** — reverse-mode AD with a **Wengert tape**: each intermediate is one
  mutable node; consumers add into its cotangent cell; backprop runs nodes once in
  reverse order. The local adjoints are exactly the Conal `Dual`-category adjoints — the
  tape just sequences their accumulation, so there's still no hand-written backward.
- **The corrected speed story:** the tape (not "continuation style") is what makes it
  fast — each shared sub-graph is visited once, no diamond recomputation, ~27× over the
  point-free `Dual`/`Cont` form it replaced.
- `Transformer.hs` (1- and 2-layer models), `Tensor.hs` (hmatrix/BLAS primitives),
  `Optimizer.hs` (AdamW + warmup/cosine schedule + Fisher–Yates shuffle), `Serialize.hs`
  (the flat-`[Float]` layout shared with Agda), `Main.hs` (the training loop + CLI).

→ Module-by-module review (correctness, clarity, fixes) in **[REVIEW.md](REVIEW.md)**.

---

## 5. Results — grokking reproduced

Reproducing Power et al. 2022's *grokking* (delayed generalization on small algorithmic
datasets) is the empirical goal. Recorded results (authoritative log:
**[GROKKING_PROGRESS.md](GROKKING_PROGRESS.md)**):

| run | modulus | layers | result | steps | wall-clock (CPU) |
|---|---|---|---|---|---|
| sanity | p=5 | 1 | 100% train (overfits) | — | seconds |
| grok | p=53 | 1 | **99.6% test** | ~tens of k | — |
| `p97hi` | p=97 | 1 | **100% test** | **29,400** | **~13.2 min** |
| `p97l2` | p=97 | 2 | **99.7% test** (≈100%) | **29,400** | **~29.7 min** |

The 2-layer p=97 model has **112,609 parameters** and shows a more gradual, paper-like
curve (39.4% test at epoch 100 → 99.7% at epoch 200), versus the 1-layer's near-immediate
climb. Weight decay is the demonstrated grokking lever (wd=1e-2 groks far sooner than
wd=1e-3, all else equal).

---

## 6. Comparison to the grokking paper (Power et al. 2022)

We reproduce the **phenomenon**, at smaller scale, on CPU. Side by side:

| | Power et al. 2022 | This implementation |
|---|---|---|
| Task | `(a ∘ b) mod p`, incl. addition | **`(a + b) mod p`** |
| Modulus | **p = 97** | **p = 97** (also p=5, p=53) |
| Train/test split | 50 / 50 | 50 / 50 (4,704 train pairs at p=97) |
| Architecture | 2-layer decoder, multi-head, width ~128 *(approx.)* | 1- or 2-layer, **single head**, dModel 64 / dFF 256 / dK 64 |
| Params | ~4×10⁵ *(approx.)* | 62,625 (1-layer) / **112,609 (2-layer)** |
| Optimizer | AdamW, wd ≈ 1, β₂ ≈ 0.98 *(approx.)*, **batch 512** | AdamW, **wd = 1e-2**, β₂ = 0.999, **batch 32** |
| Steps to generalize | onset ~10⁵, full grok demos ~10⁶ *(approx.)* | **~2.9×10⁴** (100% test) |
| Hardware / time | GPU | **CPU, ~13.2 min (1-layer) / ~29.7 min (2-layer)** |

**Honest framing.** Step counts are *not* directly comparable: at batch 32 each of our
steps sees **16× less data** than the paper's batch 512, so "fewer steps" doesn't mean
"less compute." More importantly, at a 50/50 split with 4,704 training pairs we land in
the **fast-generalization regime** — test accuracy rises almost in step with train —
rather than the paper's dramatic **delayed** grok (the long memorize-then-suddenly-grok
gap), which requires a *smaller* training fraction. So this is the same task, the same
modulus, and the same memorize-then-generalize signature under weight decay — a faithful
reproduction of the *phenomenon* at small scale, **not** a bit-for-bit replication of the
paper's curves. (Paper figures marked *approx.* are from the literature, not measured
here; our figures are from [GROKKING_PROGRESS.md](GROKKING_PROGRESS.md).)

---

## 7. Guarantees — how the backend is tied to the spec

A ladder of guarantees, all green in `nix flake check` (details in
**[CONFORMANCE.md](CONFORMANCE.md)**):

1. **The spec type-checks** — `agda-modArTransformer-check`.
2. **The gradient is numerically correct** — `transformer-gradcheck` compares the
   backend's reverse-mode gradient to central finite differences: **max abs err
   1.65e-11**. Because the backward is *derived* (composed adjoints), this verifies the
   chain-rule machinery the spec prescribes.
3. **Serialization is aligned** — Agda and Haskell agree leaf-by-leaf on the flat
   parameter layout, so one `[Float]` denotes the same model on both sides.
4. **The numeric oracle** — `checks.conformance` feeds *identical* params to both the
   Agda spec (via MAlonzo) and the Haskell backend and tolerance-diffs the outputs:
   **forward pass and loss are identical to ~1e-16** (machine precision).

Two honest caveats, both documented:

- **Gradient discrepancy (open finding).** Forward+loss match exactly, but the oracle's
  gradient comparison differs by ~0.33. Since the Haskell gradient is finite-difference-
  verified correct, this points to the **Agda spec's reverse-mode** — a real
  spec-vs-implementation gap the oracle surfaced. It's *reported* by the check but does
  not gate CI; reconciling it is the precise next step.
- **Scope boundary.** The guarantee covers the **model** (forward/loss/gradient as a pure
  function of `params + input`). It does **not** cover the **training harness** — data
  split, shuffle, RNG, init values, LR schedule, optimizer constants, checkpoint format
  all differ between Agda and Haskell, *by design*. The denotational value is the model +
  AD; the training recipe is engineering. So "the backend follows the Agda spec" holds
  for the model, not the training procedure.

---

## 8. The engineering journey

How we got here (the git history is the timeline):

- **Compile-to-Categories (CTC) explored, then abandoned.** A large effort to use Conal's
  `concat` plugin (`toCcc`) for compiled, parallel gradients. It *works* as a mechanism
  (forward/loss + reverse-mode via `ConCat.RAD.gradR`; parallel chunked training of an
  MLP and of softmax self-attention), but the **full attention block wouldn't compile at
  feasible cost** (RAM/time/tick blowups), and where it did compile the speedup was plain
  data parallelism with no per-step FLOP win. **Verdict: not beneficial for this
  small-model/CPU grok task.** The hand-written Wengert-tape backend is the production
  trainer.
- **Consolidation** to that tape trainer (removed the `ctc-*` packages and playgrounds).
- **CLI** — replaced positional args with `optparse-applicative` flags
  (`-m/--mode`, `-e/--epochs`, `-s/--seed`, `-c/--checkpoint`, `--help`); default `p97l2`.
- **Batch parallelism** — the per-example reverse passes are independent and `addA` is
  the cotangent monoid, so they run concurrently and sum (order-preserved ⇒ numerically
  identical). Measured **~1.4× on training**, near-linear on the accuracy sweeps; BLAS
  pinned to 1 thread so parallelism lives at the example level.
- **Hardened checkpoints** — atomic save (`.tmp` + `rename`, so an interrupted save can't
  corrupt the resumable file) and a validating load with a clear error.
- **The shuffle lesson** — an "optimization" (O(n) shuffle) silently changed the
  seed→split mapping and slowed grokking; once understood (the split derives from the
  shuffle), it was reverted. The audit that followed produced the scope boundary in §7.

---

## 9. Frontend & diagram

- **`diagram/index.html`** — a standalone, presentation-grade SVG (open in a browser, no
  build): the forward spine (embed → attention → LayerNorm → FFN → unembed → softmax =
  copresheaf → relative-entropy loss), the backward arrow as the transpose of the forward
  morphism run on a Wengert tape, and "For Conal" / "For Tai-Danae" callouts.
- **`frontend/`** — a Reflex-dom port of that diagram (interactive: click a pipeline stage
  for a detail panel), built with reflex-platform to a static GHCJS `.jsexe` bundle.
  Verified to build; `nix run` (in `frontend/`) serves it.

---

## 10. How to run

```bash
# enter the dev shell (Agda + GHC + deps; pins BLAS to 1 thread)
nix develop

# train (the flag CLI; default mode is p97l2 — 2-layer, p=97)
cabal run backend-transformer-train -- --help
cabal run backend-transformer-train -- -m p97l2 -e 200 -s 36 +RTS -N   # use all cores

# run every guarantee (Agda type-check + gradcheck + conformance oracle)
nix flake check

# serve the interactive diagram
nix run            # (in frontend/) — or just open diagram/index.html
```

Toolchain: **GHC 9.10.3**, `random-1.2.1.3` (SplitMix), frontend on **ghcjs-8.10.7**.

---

## 11. Honest limitations / open items

- **Gradient reconciliation** — the Agda spec's reverse-mode disagrees with the
  finite-difference-correct backend gradient (§7); auditing the Agda layer pullbacks is
  the precise follow-up.
- **Training harness not bit-faithful** to the spec (split/shuffle/RNG/schedule/optimizer/
  checkpoint differ) — intentional, but it means cross-stack training runs aren't the same
  experiment.
- **Toy task** — modular arithmetic, not natural language; the enriched/copresheaf
  structure drives the loss but the learned representations aren't shown to *realize* the
  enriched-categorical structure, and the enrichment laws are recorded, not proved for the
  learned model.

---

## 12. Documents & references

- [README.md](README.md) — Agda-spec quickstart and layout.
- [EXPLANATION.md](EXPLANATION.md) — the denotational story (for Conal & Tai-Danae).
- [CONFORMANCE.md](CONFORMANCE.md) — guarantees + the scope boundary.
- [REVIEW.md](REVIEW.md) — backend code review.
- [GROKKING_PROGRESS.md](GROKKING_PROGRESS.md) — chronological log + all measured results.
- *(historical)* `PLAN.md`, `SESSION_PROGRESS.md` — CTC-era artifacts, kept for provenance;
  the CTC effort is concluded (§8).

Papers:

- Bradley, Terilla, Vlassopoulos. *An Enriched Category Theory of Language* (2021).
  arXiv:2106.07890.
- Conal Elliott. *The Simple Essence of Automatic Differentiation*. ICFP 2018.
  arXiv:1804.00746. (and *Compiling to Categories*, ICFP 2017; the `felix` Agda library.)
- Power, Burda, Edwards, Babuschkin, Misra. *Grokking: Generalization Beyond Overfitting
  on Small Algorithmic Datasets*. arXiv:2201.02177 (2022).
