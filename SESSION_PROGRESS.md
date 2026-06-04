# Historical Session Progress: CTC Forward/Loss Escalation

This file is a compact record of the earlier CTC session. It is not an active build
plan. The `ctc-*` packages/apps/checks described here were later removed during the
consolidation to the Wengert-tape backend.

## What Was Demonstrated

- `toCcc` elaborated a staged forward/loss smoke ladder:
  - Bool projection
  - scalar `Double` arithmetic
  - 2D dot product
  - fixed 2x2 matvec
  - two-class softmax
  - class-0 NLL
  - fixed tiny MLP NLL
  - fixed two-key attention readout
  - fixed mini transformer-block NLL
- The mini block exercised attention, residual add, 2D LayerNorm via `sqrt`, FFN-style
  logits, and NLL.
- The feared `Double#` panic did not occur in these forward/loss fragments.
- `ConCat.AD.gradient` itself hit GHC simplifier trouble, but the viable gradient route
  for small tuple-shaped examples was later found to be `ConCat.RAD.gradR`.

## Why This Became Historical

- Small CTC gradient demos worked, including chunked parallel training demos.
- Full transformer-shaped reverse-mode CTC did not scale acceptably at the tested pin.
- The production trainer therefore remains the tape/hmatrix backend, which reproduced
  grokking and is covered by current flake checks.

## Current Source Of Truth

- `WALKTHROUGH.md` for the project overview.
- `GROKKING_PROGRESS.md` for chronological results.
- `CONFORMANCE.md` for the Agda↔backend forward/loss/gradient guarantee.
- `flake.nix` for active packages/apps/checks.
