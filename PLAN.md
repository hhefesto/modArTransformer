# Historical Plan: Compile-to-Categories De-Risking

This file is kept for provenance. It records the earlier Compile-to-Categories (CTC)
investigation, but the active production path is now the Wengert-tape/hmatrix backend.

## Final CTC Verdict

- Forward/loss CTC worked through small tuple-shaped fragments: scalar arithmetic, dot,
  matvec, softmax, NLL, tiny MLP, tiny attention, and a mini transformer-block loss.
- CTC reverse-mode also worked for small tuple-shaped models via `ConCat.RAD.gradR`.
- Parallel CTC training demos worked for tiny examples by evaluating chunk gradients in
  parallel.
- Scaling CTC reverse-mode to transformer-shaped attention/block fragments was not
  practical at the tested `concat`/GHC pin: compile time, simplifier ticks, and memory
  use dominated.
- The CTC packages/apps/checks were removed during consolidation. Current `flake.nix`
  no longer contains `ctc-smoke`, `ctc-grad-smoke`, `ctc-train`, `ctc-partrain`, or
  `ctc-attntrain`.

## Current Active Path

- Production training uses `backend-transformer-train`: Wengert-tape reverse-mode AD
  over hmatrix/BLAS.
- CI checks include Agda type-checking, tensor diagnostics, backend build,
  `transformer-gradcheck`, and the Agda↔backend conformance oracle.
- The conformance oracle gates forward+loss+gradient agreement between Agda and the backend.

## Remaining Follow-Ups

- Keep Agda/backend gradient conformance green as layer code evolves.
- Keep CTC as historical/research context unless a different `concat`/GHC pin or a
  smaller categorical representation makes full transformer reverse-mode practical.
- Continue improving the tape backend and its documentation, since it is the working
  full-scale grokking trainer.
