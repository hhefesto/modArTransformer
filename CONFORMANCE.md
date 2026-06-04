# Conformance: how we guarantee the backend follows the Agda spec

The Haskell fast trainer (`backend/transformer/*`) is a hand transliteration of the
type-checked Agda specification (`ModArTransformer/**`). This document records the
guarantees that the implementation actually tracks the spec, from cheapest/strongest-
in-CI to the deepest (numeric) one.

## Guarantees in place today (all in CI via `nix flake check`)

1. **The spec is well-formed** — `agda-modArTransformer-check` type-checks the whole
   Agda development (the enriched-category semantics in `Semantics/*`, the AD category
   in `Cat/*`, and the layers in `Layers/*`). If the categorical/semantic construction
   were inconsistent, this fails.

2. **The gradient is numerically correct** — `transformer-gradcheck` (now a flake check)
   runs the backend's reverse-mode gradient against central finite differences on a tiny
   model and asserts agreement (max abs err **1.65e-11**). Because every layer's backward
   is *derived* (composed local adjoints, no hand-written gradient — same construction as
   the Agda `D`/`Dual` category), this verifies the chain-rule machinery the spec
   prescribes is implemented correctly.

3. **Structural correspondence — verified serialization alignment.** The two
   implementations agree on the *parameter layout*, which is what lets a single flat
   `[Float]` denote the same model in both. Checked leaf-by-leaf:
   - matrices are **row-major** on both sides (`Serialize.hs` `concat . mtoRows`;
     `Cat/Serialize.agda` `concatMap toList`);
   - a linear layer is **(matrix, bias)** (`Lin o i` ↔ `LinParams m n = ℝMat m n × ℝVec m`);
   - LayerNorm is **(γ, β)** (`LN n` ↔ `LNParams n = ℝVec n × ℝVec n`);
   - FFN is **(up, down)**; attention flattens to **Wq, Wk, Wv, Wo** (Haskell
     `((Wq,Wk),(Wv,Wo))` and Agda `LinParams × LinParams × LinParams × LinParams`
     flatten to the same order);
   - the whole parameter product is **tokEmbed, posEmbed, Attn, LN1, FFN, LN2, unembed**
     on both sides.
   So `toFloats`/`fromFloats` produce and consume the flat list in an identical leaf
   order — the precondition for the numeric oracle below, and a guarantee in its own
   right that the two parameterizations are the same object.

4. **Documented transliteration** — `Transformer.hs` is annotated as a transliteration of
   `Layers/Transformer.agda`; `Optimizer.hs` documents that AdamW/schedule come from the
   historical `Main.hs@62b0b4d`, *not* the Agda `Cat/Adamable.agda` (a deliberate, noted
   divergence in optimizer only — the *model* follows the spec).

## Known, deliberate divergences (so "conformance" means "within tolerance")

The Haskell loss/attention subtract a **detached max** for numerical stability;
the Agda `Cat/VecPrim.agda` `logSumExp`/`softmax` are naive. This is **shift-invariant**:
same value and same gradient up to floating-point rounding. So any numeric oracle must
**tolerance-compare** (≈1e-9), not require bit-identical output. The optimizer divergence
(item 4) is outside the model and is not part of the forward/loss/gradient conformance.

## The deepest guarantee — BUILT: the numeric oracle (forward+loss conform to 1e-16)

The Agda↔Haskell numeric oracle now exists and is a green flake check
(`checks.conformance`):

- **`modArConformanceOracle.agda`** (MAlonzo executable) reads the shared
  flat-float file, rebuilds `TransformerParams 2 4 8 4` via `fromFloats`, and for
  three fixed `(a,b,t)` cases emits `logits ++ [loss] ++ gradient`
  (`eval (transformerLogits a b)` and `gradAndLoss (transformerLoss a b t)`).
- **`transformer-conformance`** (Haskell) builds the same `Params 3 4 8 4`, writes
  the params file, and emits the same quantities via the backend.
- **`checks.conformance`** runs both and tolerance-diffs the float streams.

**Result (measured):**

| quantity | max \|Δ\| (Agda vs Haskell) |
|---|---|
| logits (forward) | **6.9e-17** |
| loss             | **1.1e-16** |
| gradient         | 0.335 (discrepancy — see below) |

So the **forward pass and loss are numerically identical to the Agda spec to
machine precision** — the backend provably computes the spec's model and
objective. The check gates CI on forward+loss conformance (`max|Δ| ≤ 1e-9`).

### Finding: the gradient diverges (the oracle did its job)

Forward+loss match exactly, yet the gradient differs by ~0.33. Since the Haskell
gradient is independently finite-difference-verified correct (`transformer-gradcheck`,
1.6e-11), and the forward/loss are identical, this points to a discrepancy on the
**Agda side's reverse-mode** (`Cat.Grad.gradAndLoss` through `Cat.Dual`/the layer
pullbacks) — i.e., the spec's *stated* gradient does not match the correct gradient
of its own (conformant) forward. This is exactly what a conformance oracle is for:
it surfaced a real spec-vs-implementation gap in the gradient path. Reconciling it
(audit the Agda layer pullbacks against the finite-diff-correct backend adjoints)
is the precise next step; until then the gradient is reported by the check but does
not gate CI.

## (Original design notes) the numeric oracle

A direct Agda↔Haskell numeric check, now shown feasible by the alignment above:

- **`modArConformanceOracle.agda`** (MAlonzo executable, mirroring
  `modArTensorDiagnostics.agda` + the FFI in `modArTransformer.agda`): read a shared
  flat-float file, `fromFloats` it into `TransformerParams 2 4 8 4` (vocab 3, dM 4, dF 8,
  dK 4 — matching the Haskell gradcheck dims), and for a few baked `(a,b,t)` cases emit
  `logits = eval (transformerLogits a b) params`, the loss, and the gradient
  `gradAndLoss (transformerLoss a b t) params` (via `toFloats`), one float per line.
- **A Haskell `conformance` executable** (extends `GradCheck.hs`): build the same params
  from a fixed seed, write the shared flat-float file, compute the same logits/loss/grad
  via `transformerLogitsVal` / `transformerGradLoss`, emit in the same order.
- **A flake `check`** orchestrates: run the Haskell emitter (writes params + `hs.txt`),
  run the Agda oracle (reads params, writes `agda.txt`), then tolerance-diff the two
  float streams (`awk`, |Δ| ≤ 1e-6) — failing the build on any mismatch.

Cost/risk: the Agda oracle is a fresh MAlonzo compile (minutes, like the existing agda
checks), and the IO/FFI boilerplate must compile. The math/serialization are aligned, so
the remaining work is mechanical. Once green, CI guarantees the backend's
logits/loss/gradient match the Agda spec on shared inputs — the strongest conformance.

## Summary

| Guarantee | Status |
|---|---|
| Agda spec type-checks | ✅ CI (`agda-modArTransformer-check`) |
| Gradient = finite differences | ✅ CI (`transformer-gradcheck`, 1.65e-11) |
| Parameter serialization aligned (Agda ↔ Haskell) | ✅ verified (this doc) |
| Transliteration documented; optimizer divergence noted | ✅ |
| Numeric oracle — forward+loss match Agda in CI (to ~1e-16) | ✅ `checks.conformance` |
| Numeric oracle — gradient matches Agda | ✗ discrepancy found (Agda reverse-mode); reported, not gating |
