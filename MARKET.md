● The market-prediction LLM — heads to toes

  Everything below exists in this repo today, is committed, and every claim marked verified is
  enforced by a green nix flake check gate. I'll walk the whole spine: from what the model means,
  down through what it computes, to the data it eats, how it trains, and what comes out the other
  end when a trade hits the exchange.

  ---
  0. The one-sentence picture

  ▎ A causal transformer, formally specified in Agda and numerically conformance-gated at every
  ▎ implementation tier, learns the "language of the market" — candle returns quantized into a
  ▎ finite token alphabet — by minimizing the relative entropy between its predicted next-token
  ▎ distribution (which is, literally, a Tai-Danae Bradley meaning-copresheaf) and what the market
  ▎ actually did next; its gradients are Conal Elliott's categorical adjoints end to end, and in
  ▎ live mode it takes one gradient step per closed candle while emitting a trading recommendation
  ▎ on every trade.

  ---
  1. The head: what the model means (the Agda semantics)

  The market is treated as a language. Following Bradley's 2025 magnitude paper, any
  autoregressive next-token model π over a finite vocabulary with begin/end tokens (⊥, †) induces
  a [0,1]-enriched category of texts:

  - Objects are token sequences (market histories).
  - Hom-objects L(x,y) = π(y|x) — the probability the model assigns to history x continuing as y,
  computed as the chain-rule product of next-token probabilities.
  - The meaning of a history x is its representable copresheaf よx = L(x,−): "everything this
  history can become, weighted by likelihood." The model's softmax at a prefix is this object —
  not an analogy, an identity.

  This lives in ModArTransformer/Semantics/MarketLanguage.agda, which also proves the identity law
  (π(x|x) = 1) and states the composition law π(z|y)·π(y|x) ≤ π(z|x) as an obligation.
  Semantics/Magnitude.agda specifies the evaluation invariant Mag(tM) (Tsallis entropy; its slope
  at t=1 is the model's total Shannon entropy).

  The training loss is the semantics. Cross-entropy against the actual next token = relative
  entropy between the model's copresheaf and the Dirac copresheaf of what really happened. The
  objective isn't bolted on; it's the semantic distance.

  2. The spine: what the gradient is (categorical AD)

  There is no hand-written backward pass and no autograd library anywhere in the training path.
  Following Elliott's Simple Essence of Automatic Differentiation:

  - The model is one composite morphism in the category D (Cat/D.agda): each primitive carries its
  forward value and its transposed derivative (the Dual pullback). The chain rule is morphism
  composition — Layers/SeqTransformer.agda builds the whole transformer this way, with exactly one
  new nonlinear primitive (the masked n-ary softmax in Cat/SeqPrim.agda).
  - The fast implementation (backend/transformer/Tape.hs) is a Wengert tape: every intermediate
  gets a mutable cotangent cell, consumers add into it, backprop fires each node once in reverse
  creation order. The local adjoints are the same categorical pullbacks — the tape only sequences
  them efficiently (~27× over the point-free form).

  3. The body: the architecture (Transformer.hs ParamsSeq)

  The verified model is deliberately exactly what the spec describes:

  tokEmbed (v×dM) + posEmbed (n×dM)
  → ONE block: 2 attention heads (Wq,Wk,Wv each dK×dM per head),
    CAUSAL (position i attends keys 0..i), head-concat → Wo (dM×2dK),
    + residual → LayerNorm → FFN (dM→dF, ReLU, →dM) + residual → LayerNorm
  → unembed (v×dM) at EVERY position
  loss = mean over the n−1 positions of next-token cross-entropy

  Two instantiations: mkt-small (128 bins + ⊥/† = vocab 130, context 32, ~20k params) and mkt-base
  (512 bins, context 64, ~113k params). Bigger configs wait for the GPU tier.

  4. The nervous system: the conformance chain (what "verified" means)

  Three independent implementations of the same denotation, tied together numerically:

  ┌───────────────────────────┬────────────────────────┬──────────────────────────────────────┐
  │           tier            │          what          │              agreement               │
  ├───────────────────────────┼────────────────────────┼──────────────────────────────────────┤
  │ Agda spec (MAlonzo)       │ the denotation itself  │ —                                    │
  ├───────────────────────────┼────────────────────────┼──────────────────────────────────────┤
  │ CPU Wengert tape          │ the reference &        │ max|Δ| = 4.4e-16 vs Agda (forward +  │
  │ (hmatrix)                 │ trainer                │ loss + gradient, checks.conformance) │
  ├───────────────────────────┼────────────────────────┼──────────────────────────────────────┤
  │ libtorch tier             │ same tape design over  │                                      │
  │ (gpu/ConformanceG.hs)     │ torch kernels, no      │ max|Δ| = 2.2e-16 vs the tape         │
  │                           │ autograd               │                                      │
  └───────────────────────────┴────────────────────────┴──────────────────────────────────────┘

  Plus a finite-difference gradcheck (1.4e-11) as a spec-independent sanity gate. Scope boundary
  (README §4): the model (forward/loss/gradient as a function of params+input) is verified; the
  training harness (splits, shuffles, schedules, checkpoint cadence, the live recommendation rule)
  is engineering and explicitly out of scope.

  5. The toes: data (market-backfill, vendored Hyperliquid client)

  - Source: Hyperliquid perp DEX, public API, no keys. backend/hyperliquid/ holds the vendored
  client.
  - market-backfill pulls OHLCV candles via REST candleSnapshot into data/<coin>-<interval>.csv,
  paginated, validated (OHLC coherence), idempotent — re-running appends only newer candles, so a
  cron of it is your durable archive.
  - Hard constraint discovered empirically: the API retains only ~5000 candles per interval (1m ≈
  3.6 days, 15m ≈ 52 days, 1h ≈ 208 days). Your 1m archive deepens over time only because the cron
  keeps accumulating past the API's horizon. Until then, 15m/1h are the trainable corpora.

  6. The tokenizer (backend/data/MarketTokenizer.hs) — where market becomes language

  1. closes → log returns
  2. z-scored with train-window statistics only
  3. binned by quantile edges fitted on the train window only (the no-lookahead rule — stated as a
  side condition in the Agda spec, enforced by the pipeline, and guarded by a leak-detector
  property test)
  4. vocabulary: ⊥=0, †=1, bin i = 2+i — byte-for-byte the layout in MarketLanguage.agda; the
  Haskell quantize is the Agda recursion verbatim
  5. decoding: each bin's representative is the median training return that fell in it
  6. the fitted artifact is saved as <checkpoint>.tok — the tokenizer is part of the model; prompt
  and live mode refuse to run without it

  Ten property checks (market-tokenizer-check): monotonicity, totality, decode coherence,
  no-lookahead (with a power check), sorted edges, artifact round-trip.

  7. Offline training (market-train)

  - Walk-forward discipline: train = first --train-frac of time, validation = the rest. Never
  shuffled across the boundary.
  - Each training sequence: ⊥ + (n−1) bin tokens, sliding stride-1 windows.
  - AdamW (matrix-only weight decay), warmup→cosine LR, global-norm clip 1.0, batch gradients in
  parallel (parMap, order-preserving so results are deterministic).
  - How to read the log: val CE vs base (the unconditional train-distribution entropy). The model
  knows something only when val CE < base. best tracks the best validation epoch, continuously
  saved to PATH.best — that's the deployable checkpoint, never the final epoch. Your first run was
  the textbook failure mode: 112k params on 1k windows memorized, val CE rose from epoch 4
  onward, and the harness correctly preserved epoch 4 — which still never beat baseline. More
  data, not different knobs.
  - Checkpoints are binary (CKPTB: header + raw float64; atomic write; legacy text still loads)
  and carry the full Adam state, so --resume continues the exact trajectory.
  - --prompt gives a REPL: type recent returns, get the next-bin distribution decoded back to
  returns.

  8. Live mode (market-live) — the part that meets the market


  Two clocks, per your design decisions:

  - The token clock (1m candle close): a completed candle is one new token of the language. One
  online gradient step on the newest window — the same verified seqGradLoss morphism — constant
  --lr 1e-4, clipped; checkpoint + state saved every --save-every candles. The model keeps
  learning forever.
  - The trade clock (every transaction): the in-progress candle's return-so-far becomes a
  provisional last token; one forward pass gives the copresheaf at that prefix → expected next
  return μ (probability-weighted bin decode), P(up), entropy. A FLAT/LONG/SHORT paper state
  machine maps μ against --threshold-bps (default 10bp, which must clear --fee-bps 3.5bp twice):
  OPEN_LONG / OPEN_SHORT / CLOSE / FLIP_* / HOLD, emitted as one JSON line per trade. PnL is
  tracked in log-return units net of fees; position survives restarts via <ckpt>.pos.
  - Plumbing: WS thread → STM queue; the worker folds every trade into the aggregator but
  evaluates only the newest pending one (backpressure); tid-monotonic dedup absorbs reconnect
  replays.
  - --replay CSV drives the identical pipeline from an archive: your verification path and a
  fee-aware paper backtest.
  - It only recommends — it never places orders (the signing path is deliberately unimplemented).

  9. The introspection organs (probes)

  market-seq-probe (read-only over any checkpoint): the magnitude function and Shannon-slope (the
  model's uncertainty landscape), Yoneda synonymy (do similar histories have close copresheaves? —
  33× collapse measured even on a barely-trained model), and a numeric check of the composition 
  law (holds at ~1e-61 — Bradley's point that autoregressive models are enriched categories by 
  construction).

  10. Honest current limits

  CPU-scale models (~113k params) until the GPU tier's batched trainer lands on the NVIDIA machine
  (its conformance gate already passes); archives capped by the API's retention until your cron
  deepens them; and — the big one — no demonstrated predictive edge yet: nothing has pushed val CE
  below the unconditional baseline on real data. The entire pipeline is built and verified;
  whether 1-bar BTC returns are predictable at this scale is now an empirical question the
  pipeline can finally answer cleanly.

  11. The whole thing, in order

  backfill (15m + 1m top-up) → market-train on data/BTC-15m.csv → market-live --replay (paper
  backtest) → market-live (online learning + per-trade recommendations)

  — the exact one-liner from last session, with checkpoint-*.ckpt.best + .tok as the deployable
  pair at the center of it all.

  Since this was a walkthrough rather than a code change, there's nothing to execute — if you'd
  like this written into the README as a §0 "guided tour" section, say the word.
