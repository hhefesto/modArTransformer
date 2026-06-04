# Conformance: how we guarantee the backend follows the Agda spec

The Haskell fast trainer (`backend/transformer/*`) is a hand transliteration of the
type-checked Agda specification (`ModArTransformer/**`). This document records the
guarantees that the implementation actually tracks the spec, from cheapest/strongest-
in-CI to the deepest (numeric) one.

## Scope of the guarantee — what "follows the Agda spec" means here

The conformance guarantee is about the **denotational model**, *not* the training
procedure. Be precise about the boundary:

- **Verified surface (what the oracle gates):** the model's forward pass, loss, and
  gradient — `transformerLogits`, `transformerLoss`, and `gradAndLoss` as pure functions
  of `(params, input)`.
  `checks.conformance` feeds *shared, fixed* params from a file and compares Agda vs
  Haskell on a few `(a,b,t)` cases; it never generates data, splits, shuffles, or runs
  the optimizer loop. Forward, loss, and gradient now match to ~1e-16 (below).

- **Out of scope (the training harness — intentionally NOT bit-faithful to the spec):**
  the surrounding *recipe* has deliberately drifted between Agda and Haskell, and
  nothing checks it. Known divergences:
  | aspect | Agda spec | Haskell backend |
  |---|---|---|
  | train/test split | even/odd by position — deterministic, no RNG (`Data.agda`, `modArTransformer.agda:169`) | shuffle, then take halves — RNG-dependent (`Optimizer.hs` `splitData`) |
  | shuffle | front-swap Fisher–Yates on `Vec` (`Data.agda`) | selection shuffle on a list (`Optimizer.hs`) |
  | RNG | legacy L'Ecuyer LCG (`Random.agda`) | **SplitMix** (`random-1.2.1.3`) — different stream entirely |
  | init values | Xavier `sqrt(6/(r+c))`, same leaf order — but different RNG ⇒ different numbers | same formula/order (`Main.hs` `initFlat`) |
  | LR schedule | keyed on **epoch**, warmup minLR→base (`Train.agda`, `Schedule.agda`) | keyed on **global step**, warmup 0→base (`Optimizer.hs`) |
  | optimizer (AdamW) | `Cat/Adamable.agda` | historical `Main.hs@62b0b4d` (eps-placement/warmup differ) |
  | checkpoint | params-only; resets Adam + epoch on load | header + params + Adam m/v; full resume |

This split is **by design**: the spec's denotational value is proving the *model + AD*
correct (Conal's AD-as-categories; Tai-Danae's enriched-copresheaf loss), which the
oracle covers. The training recipe (which examples are held out, batch order, RNG,
schedule constants) is engineering, not semantics, and is not claimed to be bit-faithful.
A fresh Agda run and a fresh Haskell run from the same seed are therefore *not* the same
experiment — they share the model, not the trajectory.

## Guarantees in place today (all in CI via `nix flake check`)

1. **The spec type-checks** — `agda-modArTransformer-check` type-checks the whole
   Agda development (the enriched-category semantics in `Semantics/*`, the AD category
   in `Cat/*`, and the layers in `Layers/*`). This guarantees the raw constructions are
   accepted by Agda; semantic laws and Float interval bounds that are recorded as future
   obligations are not thereby proved.

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

The backend and Agda both use max-stabilized two-class attention softmax; the backend
also uses a max-stabilized cross-entropy loss. These stabilizations are shift-invariant:
same value and same gradient up to floating-point rounding. So the numeric oracle
**tolerance-compares** (≈1e-9), not requiring bit-identical output. The optimizer
divergence (item 4) is outside the model and is not part of the forward/loss/gradient
conformance.

## The deepest guarantee — BUILT: the numeric oracle (forward+loss+grad conform to 1e-16)

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
| logits (forward) | **1.1e-16** |
| loss             | **1.1e-16** |
| gradient         | **2.3e-16** |

So the **forward pass, loss, and gradient are numerically identical to the Agda spec to
machine precision** — the backend provably computes the spec's model, objective, and
reverse-mode derivative on shared inputs. The check gates CI on all three streams
(`max|Δ| ≤ 1e-9`).

### Finding resolved: the gradient divergence was LayerNorm

The oracle originally found a ~0.33 gradient delta while forward+loss matched exactly.
The cause was the Agda `LayerNorm` pullback: the centered expression needed to be scaled
as a whole by `invStd`. After that fix, the Agda gradient agrees with the
finite-difference-correct backend to machine precision, and `checks.conformance` now gates
gradient agreement too.

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
  float streams (`awk`, |Δ| ≤ 1e-9) — failing the build on forward/loss/gradient
  mismatch.

Cost/risk: the Agda oracle is a fresh MAlonzo compile (minutes, like the existing Agda
checks), and the IO/FFI boilerplate must compile. The math/serialization are aligned, and
the current green check guarantees the backend's logits/loss/gradient match the Agda spec
on shared inputs.

## Summary

| Guarantee | Status |
|---|---|
| Scope: *model* forward/loss/gradient verified as pure fns of params+input; *training harness* out of scope | ℹ️ see "Scope of the guarantee" |
| Agda spec type-checks | ✅ CI (`agda-modArTransformer-check`) |
| Gradient = finite differences | ✅ CI (`transformer-gradcheck`, 1.65e-11) |
| Parameter serialization aligned (Agda ↔ Haskell) | ✅ verified (this doc) |
| Transliteration documented; optimizer divergence noted | ✅ |
| Numeric oracle — forward+loss match Agda in CI (to ~1e-16) | ✅ `checks.conformance` |
| Numeric oracle — gradient matches Agda in CI (to ~1e-16) | ✅ `checks.conformance` |
