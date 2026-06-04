# Code review — fast trainer (backend/transformer/*)

Scope: the production training path after consolidation — the Wengert-tape /
hmatrix backend (`backend/transformer/{Tensor,AD,Tape,Transformer,Optimizer,
Serialize,Main,GradCheck}.hs`), built by `backend-transformer-train` and
`transformer-gradcheck`. Reviewed against the Agda spec (`ModArTransformer/**`)
and the `flake.nix` wiring.

## Verdict

Solid, correct, and small (~1.1k LOC). The reverse-mode AD is mathematically
sound (finite-difference gradcheck: max abs err **1.6e-11** on a p=3 model), and
the end-to-end pipeline reproduces grokking (p=53 → 99.6% test) and fast
generalization (p=97 → 100% test in ~29.4k steps). No correctness blockers found.
The notes below are clarity / dead-code / fidelity items, not bugs.

## Per-module

- **Tape.hs** — the heart, and it's clean. Reverse-mode AD via a Wengert tape:
  each node is `R { primalR, adjR }` (primal + mutable cotangent cell); `node`
  registers a newest-first backprop action; `tGradLoss` seeds `dL/dL=1`, runs the
  actions in creation-reverse order (a valid reverse-topological order), reads the
  accumulated parameter gradient. O(graph) forward and backward, every node touched
  once — no diamond recomputation. Each primitive's local adjoint is correct
  (`tMatvec`: `vouter dy x` / `matvec (mtr w) dy`; `tVdot`, `tScaleV`, `tCenter` as
  a symmetric map, etc.). This is the speed win and it's well done.

- **AD.hs** — defines the full Conal `Dual` category (`D a b = a→(b, Dual a b)`,
  `forkD`, `>->`, `exlD/exrD`, lens `projD`, `gradAndLoss`). **Finding:** the fast
  trainer only uses the **`Lens`** part of this module (`Tape.hs` / `Transformer.hs`
  import `Lens(..), fstL, sndL, (.<)`); the `D`/`Dual`/`forkD`/`>->`/`gradAndLoss`
  machinery is **not on the execution path** (the tape is). It's valuable as the
  conceptual "AD as categories" reference (mirrors the Agda `Cat/*`), but it is
  effectively dead code at runtime. *Recommendation:* either (a) keep it, clearly
  marked "conceptual reference, not the runtime path," or (b) split `Lens` into its
  own tiny module and drop the unused categorical combinators. (Documented honestly
  for Conal in EXPLANATION.md.)

- **Transformer.hs** — the model as a straight-line tape program (seqLen 2, single
  head, position-0 readout), transliterated from `Layers/Transformer.agda`.
  Correct. **Fidelity note:** the loss subtracts a detached max (`tDetachMax`) for
  numerical stability and the attention path is likewise stabilized; the Agda
  `Cat/VecPrim.agda` `logSumExp`/`softmax` are *naive* (no max-subtraction). These
  are **shift-invariant** (same value & gradient up to float rounding), so it's a
  deliberate, safe divergence — but it IS a divergence from the spec and should be
  called out (it is, in EXPLANATION.md and the conformance plan).

- **Optimizer.hs** — decoupled AdamW, correct: `m'/v'` EMA, bias-correct via
  `b1Pow/b2Pow`, `w' = w − lr·m̂/(√v̂+ε) − lr·wd·w`, with **weight decay on matrices
  only** (the `Adam (M m n)` instance passes `adamWD`; `Adam (V n)` passes 0).
  Warmup→cosine `lrWarmupCosine` keyed on the global step. **Fidelity note:**
  explicitly transcribed from the historical success `Main.hs@62b0b4d`, NOT from the
  Agda `Cat/Adamable.agda`/`Optimizer/Schedule.agda` (whose eps-placement/warmup
  differ) — intentional, documented in the module header. **Minor:** `shuffle` is
  O(n²) (`splitAt`+`pre++post` per draw); for p=97 (9409 items) that's ~tens of
  millions of ops/epoch — not the bottleneck (BLAS dominates), but a mutable-array
  Fisher–Yates would be a cheap win if shuffle ever shows up in a profile.

- **Tensor.hs** — shape-indexed `V n`/`M m n` over hmatrix with phantom Nats;
  `Additive`/`Scale`, native `vmapT`/`vzipT`, `matvec`/`vouter`/`mtr`, NFData. Clean;
  the categorical interface sits above hmatrix (BLAS is one interpretation). Good.

- **Serialize.hs** — generic flat `[Double]` (de)serialization over the parameter
  product tree in a fixed leaf order; backs the checkpoint format. Important: this
  leaf order must match the Agda `Cat/Serialize.agda` for the conformance oracle
  (Phase 3) to feed bit-identical params to both sides.

- **Main.hs** — training driver: Xavier init (zero bias, LN γ=1/β=0), full
  checkpoint/resume (params + Adam moments + step + b1/b2 powers + epoch), per-step
  schedule, argmax accuracy, the self-describing header + labeled columns + elapsed
  timing, CLI `<mode> [maxEpochs] [seed]` (p5/p53/p53hi/p97/p97hi). Reads cleanly.

- **GradCheck.hs** — finite-difference vs `transformerGradLoss` on a tiny model; the
  correctness gate. Good; consider promoting it to a flake `check` so the gradient
  stays verified in CI (currently only the build is checked).

## Cross-cutting

- **Strengths:** no hand-written backward anywhere (every adjoint is local and
  composed); shape-indexed tensors catch dimension errors at compile time; the
  backend is a faithful transliteration of a type-checked Agda spec; reproducible
  (fixed seed, now CLI-overridable).
- **Top recommendations (in priority order):**
  1. **Add an Agda↔backend conformance check** (the real guarantee — Phase 3 below).
     Today the spec is type-checked and the gradient is finite-diff-verified, but
     nothing proves the Haskell numerics match the Agda numerics. This is the most
     valuable missing test.
  2. Promote `transformer-gradcheck` to a flake `check`.
  3. Decide AD.hs's fate (mark-as-reference vs extract `Lens`).
  4. (Optional) O(n) shuffle if it ever matters.
- **No security/footgun issues** for a research trainer (file I/O is the checkpoint
  only; no network, no unsafe).
