# Session Progress: CTC Forward/Loss Escalation

Date: 2026-06-02
Branch: `denotational-rewrite`

## Compact Session Summary

- User asked what had been done so far; compact context was restored around grokking reproduction,
  the fast Haskell backend, CTC smoke tests, and the isolated `ConCat.AD.gradient` blocker.
- We continued the CTC acceptance-gate path rather than changing the working tape/hmatrix trainer.
- We extended `ctc-smoke/Main.hs` beyond the previous green Stage 0-7 ladder.
- The next natural step implemented was Stage 8: a fixed mini transformer-block loss through
  `toCcc`, shaped as attention -> residual add -> 2D layer norm -> FFN-style logits -> class-0 NLL.
- Stage 8 builds, runs, and agrees exactly with direct Haskell for the tested input:
  `ctc=0.4669857292892942`, `direct=0.4669857292892942`.
- `PLAN.md` and `GROKKING_PROGRESS.md` were updated to record Stage 8 and the current verdict.
- `ConCat.AD.gradient` remains isolated in `ctc-grad-smoke/` because it exhausts GHC simplifier ticks;
  it is not included in checks.

## Current CTC Stage Ladder

All of these pass through Conal's Compile-to-Categories plugin via `toCcc`:

- Stage 0: Bool projection, `ctc=True`.
- Stage 1: scalar `Double` affine arithmetic, `ctc=13.0`.
- Stage 2: 2D dot product, `ctc=11.0`.
- Stage 3: fixed 2x2 matvec, `ctc=(17.0,39.0)`.
- Stage 4: two-class softmax, exact agreement to the test tolerance.
- Stage 5: class-0 NLL, exact agreement to the test tolerance.
- Stage 6: fixed tiny MLP NLL, `ctc=0.34622423561117693`.
- Stage 7: fixed two-key attention readout,
  `ctc=(0.7123283038410656,-0.8493132153642624)`.
- Stage 8: fixed mini transformer-block NLL, `ctc=0.4669857292892942`.

## Verification Performed

- `nix -Lv build .#ctc-smoke --no-link` passed.
- `nix -Lv run .#ctc-smoke` passed and printed `ctc smoke passed`.
- `nix -Lv build .#checks.x86_64-linux.ctc-smoke --no-link` passed.

## Current Verdict

- The feared `Double#` panic has not occurred at the current pin.
- `toCcc` can elaborate scalar arithmetic, linear kernels, `exp`, division, `log`, `sqrt`, softmax,
  NLL, a tiny MLP loss, a tiny attention readout, and a composed mini block loss.
- This makes the forward/loss CTC route viable enough to keep escalating toward a shape-fixed
  transformer fragment and eventually a `toCcc`-able `forwardT`.
- This does not yet make full categorical-gradient training via CTC viable.

## `ConCat.AD.gradient` Trouble — RESOLVED (use `ConCat.RAD.gradR`)

The isolated gradient smoke was:

```haskell
ConCat.AD.gradient (\(x, y) -> x * x + y * y)
```

which exhausted GHC simplifier ticks under `ghc948`/`concat`, looping on `$fPointed:*:` /
`$fPointedPar1` (~1.86M ticks) regardless of tick factor or loop-breaker flags.

Root cause (representation-specific): `ConCat.AD.gradient` uses `D s = GD (L s)` — `LinearRow`
row-matrix linear maps — which pull in the free-vector-space `Pointed` instances that loop.

Fix: compute the gradient through **`ConCat.RAD.gradR`** — reverse-mode AD via
`RAD = GD (Dual (-+>))` (`Dual AdditiveFun`), the path ConCat's own `BasicTests` uses
(`andGradR`/`andGrad2R`), constrained only by `Num s`, with no `L s`/`Pointed` machinery. Also
dropped the loop-breakers (`-funfolding-case-threshold=1`, `-funfolding-case-scaling=5`) and added
`-fexpose-all-unfoldings`; `-fsimpl-tick-factor=2000 -freduction-depth=0` suffice.

Result: `ctc-grad-smoke` compiles and `gradR (\(x,y)->x*x+y*y) (3,4) = (6.0,8.0)`. It is now a
green flake check (`checks.x86_64-linux.ctc-grad-smoke`). CTC gradients flow through `toCcc` with
no hand-written backward — the Conal way. The forward/loss `toCcc` path remains green through
Stage 8.

## Implications Of The Gradient Blocker

- We should not put `ConCat.AD.gradient` into flake checks yet; it would make the project red for a
  path that is not needed by the working trainer.
- The fast tape/hmatrix backend remains the practical autonomous training path.
- Forward/loss CTC can continue to be escalated independently, which still helps expose categorical
  product structure for future parallel interpretation.
- Full categorical-gradient compilation remains a separate compiler/GHC simplifier problem, not a
  mathematical rejection of Conal-style differentiation.
- The likely next investigation is to reduce or specialize the `gradient` instance search/Core shape,
  try smaller explicit scalar/product types, inspect plugin-generated Core, or test a different known
  good `concat`/GHC pin.

## Files Changed In This Session

- `ctc-smoke/Main.hs`: added Stage 8 `ctcTinyBlockNll0` and direct comparison.
- `PLAN.md`: updated current CTC state, Stage 8, verification, and gradient implications.
- `GROKKING_PROGRESS.md`: updated Phase 5 summary and Stage 8 verdict.
- `SESSION_PROGRESS.md`: this compact session record.

## Next Natural Step

Escalate from the tiny hand-expanded block to a larger shape-fixed block fragment, still with
ordinary forward/loss `toCcc` and no `ConCat.AD.gradient` in checks. Keep the gradient smoke isolated
until the GHC simplifier loop is understood.
