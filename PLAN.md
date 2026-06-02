# Plan: De-risk the Compile-to-Categories plugin (toCcc) via staged ctc-smoke escalation

## Context

The user wants the transformer to eventually run in parallel via Conal Elliott's
Compile-to-Categories plugin (`toCcc`). The parallelism itself comes from a *parallel category
interpretation* (sparking `***`/`&&&` branches), which is separable from the plugin — but the
user chose to **first establish whether the literal `toCcc` plugin path is even viable here**,
before any large rewrite of the transformer into a category-polymorphic morphism.

State today:
- `ctc-smoke/Main.hs` is the green forward/loss baseline: Stage 0 Bool projection, Stage 1 scalar
  `Double`, Stage 2 dot product, Stage 3 fixed 2x2 matvec, Stage 4 two-class softmax, and Stage 5
  class-0 negative-log-likelihood all elaborate through `toCcc` and run successfully. Stage 6, a
  fixed tiny MLP loss (affine -> sigmoid -> affine -> NLL), and Stage 7, a fixed two-key attention
  readout (dot scores -> softmax -> value mix), also build and run. Stage 8, a fixed mini
  transformer-block loss (attention -> residual -> layernorm -> FFN logits -> NLL), builds and runs
  too. It is wired into the flake
  (`packages.ctc-smoke`, `apps.ctc-smoke`, `checks.ctc-smoke`) with `concat` pinned to **ghc948** and
  `dontCheck` on `concat-plugin` (`flake.nix:127-141`).
- The anticipated `Double#` panic did not occur for scalar, kernel, softmax, or loss `Double`
  arithmetic, including the fixed tiny MLP loss, attention readout, and mini block loss.
- Stage 3 `ConCat.AD.gradient` is isolated in `ctc-grad-smoke/` and wired as a package/app only, not
  as a check, because it still exhausts GHC simplifier ticks.
- This work stays entirely inside `ctc-smoke/` and the flake — the validated tape-based trainer
  (`backend-transformer-train`) is untouched and remains the working path.

Goal of this plan: a captured, reproducible answer to "does `toCcc` elaborate (a) a Bool
projection, (b) scalar `Double` arithmetic, (c) small linear kernels, and (d) the softmax/log-loss
numerics used by attention and cross-entropy?" — and, where it panics, the Conal-style resolution
(route numerics through `NumCat`/`FloatingCat`, correct plugin flags) or a precisely-documented
blocker.

## Plan (each stage gated; stop and capture on first failure)

### Stage -1 — Keep the baseline smoke green while fixing gradient CTC
- Treat the committed Bool/scalar/dot smoke (`7307203`) as the known-good baseline.
- Current unstaged Stage 3 (`ConCat.AD.gradient`) fails during build with GHC simplifier ticks
  exhausted around `$fPointedPar1` (`Total ticks: 1475201`). This is a new gradient-specific CTC
  issue, not a regression of the already-verified Stage 0-2 numeric `toCcc` path.
- First apply the GHC-recommended loop breakers to `ctc-smoke.cabal`:
  `-funfolding-case-threshold=1`, `-funfolding-case-scaling=5`, and only then raise
  `-fsimpl-tick-factor` if needed.
- If Stage 3 still fails, split it out of the baseline smoke: keep `ctc-smoke` as Stage 0-2 and move
  the gradient experiment into `ctc-grad-smoke/`. Do not add it to flake checks until it compiles.
  This preserves the green CTC numeric-kernel gate while documenting the heavier CTC-gradient blocker.
- Current implementation follows this split: `ctc-smoke` contains the green forward/loss stages;
  `ctc-grad-smoke/` contains the `ConCat.AD.gradient` experiment and its loop-breaking GHC flags.
  Even after raising `-fsimpl-tick-factor` from 4000 to 10000, the gradient build still fails with
  simplifier ticks exhausted (`$fPointed:*:`, total ticks 1864001), so CTC-gradient remains isolated
  as a gradient-specific blocker.
- Verification command convention: use `nix -Lv build ...` / `nix -Lv run ...` (or the user's `n`
  wrapper outside this tool shell) so build logs are verbose.

### Stage 0 — Does the existing Bool smoke build & run?
- `nix build .#ctc-smoke` then run the binary (or `nix run .#ctc-smoke`); also try the flake
  `checks` entry. Capture stdout/stderr verbatim.
- If it fails: diagnose the plugin setup. The likely fix is **plugin ghc-options** — Conal's
  `concat` examples compile with a specific set (e.g. `-fexpose-all-unfoldings`,
  `-fno-liberate-case`, `-fno-omit-interface-pragmas`, `-fsimpl-tick-factor=…`, `-O2`) beyond the
  lone `-fplugin=ConCat.Plugin` currently in `ctc-smoke.cabal:17`. Adopt that flag set from
  concat's own `examples` cabal. Confirm `concat-classes`/`concat-plugin` versions resolve under
  the pinned `ghc948` overlay.

### Stage 1 — Scalar `Double` arithmetic (the expected `Double#` panic)
- Add a second elaborated function in `ctc-smoke/Main.hs`, e.g.
  `ctcAffine = toCcc @(->) @(Double, Double) @Double (\(x, y) -> x * y + 1)`, and check it equals
  the direct computation.
- This is where the `Double#` panic is expected. Resolve **the Conal way**: ensure `+`/`*` are
  elaborated to `ConCat` `NumCat`/`FloatingCat` morphisms rather than leaking unboxed `Double#`
  primops — typically a matter of the reboxing/plugin flags above and importing the ConCat
  numeric vocabulary, not hand-writing primops. Capture the exact panic text if it occurs and the
  flag/import change that clears it.

### Stage 2 — A small numeric kernel (dot)
- Escalate to a fixed-size dot product over tuples of `Double`
  (e.g. `toCcc @(->) (\((a,b),(c,d)) -> a*c + b*d)`), proving numeric kernels — not just scalars —
  elaborate. This is the smallest thing resembling the transformer's real ops.

### Stage 3 — Fixed matvec
- Escalate to a fixed 2x2 matvec over nested tuples of `Double`, the smallest transformer-shaped
  linear kernel.

### Stage 4 — Softmax numerics
- Escalate to a two-class softmax, exercising `exp`, addition, and division through `toCcc`.

### Stage 5 — Loss numerics
- Escalate to a class-0 negative-log-likelihood, exercising `log` and `negate` in the loss path.

### Stage 6 — Tiny MLP loss
- Escalate to a fixed tiny MLP loss: affine -> sigmoid -> affine -> class-0 NLL. This is the first
  multi-layer forward/loss fragment and is the smallest CTC-able analogue of the transformer's FFN
  plus cross-entropy path.

### Stage 7 — Tiny attention readout
- Escalate to a fixed two-key attention readout: query-key dot scores -> softmax -> weighted value
  mix. This is the smallest CTC-able analogue of one attention head's score/read path.

### Stage 8 — Tiny transformer block loss
- Escalate to a fixed mini transformer-block loss: attention -> residual add -> 2D layer norm ->
  FFN-style logits -> class-0 NLL. This is the first composed attention+normalization+FFN+loss CTC
  smoke and exercises `sqrt` in addition to `exp`/`log`.

### Stage 9 — Record the verdict
- Update `GROKKING_PROGRESS.md` Phase 5 with: what builds, the resolved/unresolved `Double#`
  status, the working plugin flag set, and the concat/GHC versions. Record the separate gradient
  blocker without weakening the green forward/loss baseline verdict.

## Current Verdict

- Forward/loss CTC is viable through a composed mini block: scalar arithmetic, dot, matvec, softmax,
  NLL, sigmoid MLP, attention readout, residual add, layer norm via `sqrt`, FFN-style logits, and loss
  all elaborate through `toCcc` at the current pin.
- The anticipated `Double#` panic has not occurred in the forward/loss path.
- CTC-gradient is still blocked: even `ConCat.AD.gradient (\(x, y) -> x*x + y*y)` exhausts GHC
  simplifier ticks under ghc948 despite loop-breaking flags and a high simplifier tick factor.
- Therefore keep escalating `ctc-smoke` for forward/loss fragments, keep `ctc-grad-smoke` package/app
  only, and do not add gradient CTC to checks until the simplifier loop is solved.

## Gradient Blocker Implications

- This is not evidence that categorical differentiation is wrong; it is a compiler/plugin/Core-size
  problem in the `ConCat.AD.gradient` path.
- The working trainer remains the tape/hmatrix backend, whose gradients are still Conal-style local
  adjoints composed by the chain rule, with Wengert tape accumulation for sharing.
- Parallel categorical interpretation can proceed first on the green forward/loss `toCcc` fragments.
- Full training by CTC-compiled categorical gradients remains a separate research task: reduce the
  gradient smoke, inspect generated Core, specialize product/pointed instances, or try another known
  good `concat`/GHC pin.

## Critical files
- Modify: `ctc-smoke/Main.hs` (add Stage 1 & 2 functions), `ctc-smoke.cabal` (plugin ghc-options,
  deps), possibly `flake.nix` (concat rev/GHC or package overrides if Stage 0 needs it),
  `GROKKING_PROGRESS.md` (verdict).
- Untouched: all of `backend/transformer/*` (the working trainer).

## Verification
- `nix -Lv build .#ctc-smoke --no-link` succeeds and `nix -Lv run .#ctc-smoke` prints "passed" for
  Stage 0 (Bool), Stage 1 (scalar), Stage 2 (dot), Stage 3 (matvec), Stage 4 (softmax), and Stage 5
  (NLL), Stage 6 (tiny MLP NLL), Stage 7 (tiny attention readout), and Stage 8 (tiny block NLL) —
  each compared against the direct Haskell result.
- The flake `checks.ctc-smoke` is green.
- `GROKKING_PROGRESS.md` records the exact working flag set + versions (reproducible).

## Risks
- **Plugin/GHC incompatibility**: `concat` (rev pinned, ghc948) may not build cleanly; `dontCheck`
  already hints its own tests fail. Mitigation: try the documented ConCat flag set first; if the
  plugin is fundamentally broken at this pin, bump/pin a known-good `concat` rev or GHC, and if
  still blocked, **document it and fall back** to the plugin-independent parallel-category route
  (the user's option 2) — the parallelism goal does not strictly need the plugin.
- **`Double#` panic may be stubborn**: time-box Stage 1; capture the panic and the attempted fixes
  even if unresolved, so the verdict is actionable.
- **Scope creep**: this plan deliberately does NOT rewrite the transformer or build the parallel
  category yet — it only answers the viability question that gates that larger work.
