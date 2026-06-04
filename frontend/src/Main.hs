{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecursiveDo #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- Reflex-dom port of the old PureScript/Halogen diagram (recovered from git
-- d893ccd~1), rewritten to show the denotational stack: the Tai-Danae Bradley
-- enriched-category language model (softmax = copresheaf, loss = relative entropy
-- to the Dirac truth) and Conal Elliott's AD-as-categories (backward = transpose
-- of the forward morphism, run as a Wengert tape).
--
-- Build with reflex-platform (this directory is its own flake, like the user's
-- other reflex projects):  nix build .#frontend   (GHCJS -> static .jsexe), or
-- nix run .#dev  for a jsaddle-warp dev server.  See ../diagram/index.html for the
-- same diagram as standalone static SVG.
module Main where

import           Reflex.Dom
import           Data.Text (Text)
import qualified Data.Text as T

-- one stage of the pipeline (the data the old Block record carried, renewed)
data Stage = Stage
  { sId    :: Text
  , sTitle :: Text
  , sSub   :: Text
  , sWhat  :: Text
  , sWhy   :: Text
  }

stages :: [Stage]
stages =
  [ Stage "embed" "Embedding" "token + position → ℝ²ˣ⁶⁴"
      "Each token a,b ∈ {0..p-1} is looked up in a learned table and a positional vector is added."
      "Gives the two tokens geometry to compute with; learned end-to-end by the reverse pass."
  , Stage "attn" "Self-Attention (single head)" "Q·Kᵀ → softmax → ·V → Wₒ, + residual"
      "Both token positions attend over both keys/values; output projection then a residual add."
      "The core transformer mechanism. In a 2-layer model layer 2 attends over layer 1's outputs."
  , Stage "ln" "LayerNorm" "centre / scale"
      "Normalises the residual stream (mean 0, unit variance, then γ,β)."
      "Stabilises depth; in the backend its sqrt-based variance is one tape primitive."
  , Stage "ffn" "FFN  64 → 256 → ReLU → 64" "+ residual → LayerNorm"
      "Position-wise MLP expanding to 256, ReLU, back to 64, residual, LayerNorm."
      "Adds nonlinearity beyond attention; stacked ×N (the paper used 2 layers)."
  , Stage "unembed" "Unembed (position 0)" "ℝ⁶⁴ → ℝᵖ"
      "Reads the position-0 residual of the final layer and projects to p class logits."
      "Only position 0 feeds the readout: it predicts (a+b) mod p."
  , Stage "copresheaf" "softmax = π(· | [a,b])" "a [0,1]-enriched copresheaf"
      "The softmax output IS the representable copresheaf hˣ = L(x,−): the meaning of the context."
      "Tai-Danae Bradley: meaning lives in the [0,1]-copresheaf category; not an analogy — it is the hom-object."
  , Stage "loss" "loss = relative entropy(π ‖ δ)" "to the Dirac truth at (a+b) mod p"
      "Cross-entropy to the one-hot target = relative entropy of the meaning to the Dirac ground-truth meaning."
      "The Shannon (t→1) limit of Bradley's magnitude/Tsallis invariant — the loss IS the semantic quantity."
  ]

main :: IO ()
main = mainWidgetWithHead headW bodyW

headW :: MonadWidget t m => m ()
headW = do
  el "title" (text "Denotational Modular-Arithmetic Transformer")
  elAttr "meta" ("charset" =: "UTF-8") blank
  el "style" $ text css

css :: Text
css = T.unlines
  [ "body{margin:0;background:#0b1020;color:#e8eefc;font:14px/1.5 system-ui,sans-serif}"
  , "header{padding:16px 22px;border-bottom:1px solid #26344f}"
  , ".wrap{display:flex;gap:18px;padding:18px 22px;flex-wrap:wrap}"
  , ".col{flex:1 1 420px;min-width:320px}"
  , ".stage{background:#0f1830;border:1px solid #26344f;border-radius:8px;padding:10px 12px;margin:8px 0;cursor:pointer}"
  , ".stage:hover{border-color:#42c9ff}"
  , ".stage.sel{border-color:#c792ea;background:#16122a}"
  , ".panel{background:#0f1830;border:1px solid #26344f;border-radius:10px;padding:16px}"
  , ".sub{color:#9fb0d0;font-size:12px}"
  , ".arrow{color:#42c9ff;text-align:center;margin:-2px 0}"
  , "h1{font-size:19px;margin:0 0 4px} h2{font-size:15px;margin:0 0 6px}"
  , "code{background:#0a1226;border:1px solid #26344f;border-radius:5px;padding:1px 5px}"
  ]

bodyW :: MonadWidget t m => m ()
bodyW = do
  elClass "header" "" $ do
    el "h1" $ text "Denotational Modular-Arithmetic Transformer"
    elClass "p" "sub" $ text "meaning = Tai-Danae Bradley's [0,1]-enriched semantics · gradient = Conal Elliott's AD-as-categories (Wengert tape). Learns (a + b) mod p."
  elClass "div" "wrap" $ do
    -- left: clickable pipeline; emits the clicked stage
    selE <- elClass "div" "col" $ do
      clicks <- mapM stageRow stages
      pure (leftmost clicks)
    sel <- holdDyn (head stages) selE
    -- right: detail panel for the selected stage
    elClass "div" "col" $ elClass "div" "panel" $ do
      dyn_ $ ffor sel $ \s -> do
        el "h2" $ text (sTitle s)
        elClass "p" "sub" $ text (sSub s)
        el "p" $ do el "strong" (text "What. "); text (sWhat s)
        el "p" $ do el "strong" (text "Why. "); text (sWhy s)
      elClass "p" "sub" $ text "Backward pass = transpose of the forward morphism (Dual AddFun); no hand-written backward; run as a Wengert tape (~27× over the point-free form)."

stageRow :: MonadWidget t m => Stage -> m (Event t Stage)
stageRow s = do
  (e, _) <- elClass' "div" "stage" $ do
    el "div" $ text (sTitle s)
    elClass "div" "sub" $ text (sSub s)
  elClass "div" "arrow" $ text "↓"
  pure (s <$ domEvent Click e)
