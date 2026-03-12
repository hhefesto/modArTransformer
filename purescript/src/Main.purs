module Main where

import Prelude

import Data.Array (find, (:))
import Data.Maybe (Maybe(..), fromMaybe)
import Data.Int (toNumber)
import Data.Foldable (for_)
import Data.Tuple (Tuple(..))
import Effect (Effect)
import Effect.Aff.Class (class MonadAff)
import Halogen as H
import Halogen.Aff as HA
import Halogen.HTML as HH
import Halogen.HTML.Events as HE
import Halogen.HTML.Properties as HP
import Halogen.Svg.Elements as SE
import Halogen.Svg.Attributes as SA
import Halogen.Svg.Attributes.FontWeight as SFW
import Halogen.VDom.Driver (runUI)
import Web.DOM.NonElementParentNode (getElementById)
import Web.HTML (window)
import Web.HTML.HTMLDocument (toDocument)
import Web.HTML.HTMLElement (fromElement)
import Web.HTML.Window (document)
import Web.DOM.Document (toNonElementParentNode)
import Effect.Class (liftEffect)

-- ════════════════════════════════════════════════════════════════════════════
-- COLORS
-- ════════════════════════════════════════════════════════════════════════════

type Colors =
  { bg :: String
  , panel :: String
  , panelBdr :: String
  , card :: String
  , bdr :: String
  , txt :: String
  , dim :: String
  , bright :: String
  , accent :: String
  , green :: String
  , greenBg :: String
  , orange :: String
  , orangeBg :: String
  , pink :: String
  , pinkBg :: String
  , purple :: String
  , purpleBg :: String
  }

c :: Colors
c =
  { bg: "#06090f"
  , panel: "#0d1117"
  , panelBdr: "#1b2535"
  , card: "#111927"
  , bdr: "#1e2d4a"
  , txt: "#bfcee0"
  , dim: "#4e6280"
  , bright: "#e4edf8"
  , accent: "#58b4f6"
  , green: "#5ec26a"
  , greenBg: "#0f1f14"
  , orange: "#f0a346"
  , orangeBg: "#1f170d"
  , pink: "#e8628a"
  , pinkBg: "#1f0d15"
  , purple: "#a98be0"
  , purpleBg: "#150d1f"
  }

-- ════════════════════════════════════════════════════════════════════════════
-- BLOCK DATA
-- ════════════════════════════════════════════════════════════════════════════

type Block =
  { id :: String
  , label :: String
  , sub :: String
  , color :: String
  , bg :: String
  , what :: String
  , why :: String
  , params :: String
  , shapes :: String
  , hs :: String
  }

totalParams :: String
totalParams = "56,949"

paramBreakdown :: String
paramBreakdown = "Embed 3,520 · Attn 16,768 · MLP 33,216 · Unembed 3,445"

blocks :: Array Block
blocks =
  [ { id: "embed"
    , label: "Embedding"
    , sub: "Token + Position → ℝ²ˣ⁶⁴"
    , color: c.accent
    , bg: "#0d1a2a"
    , what: """Each integer a, b ∈ [0..52] is looked up in a learned embedding table (53 rows × 64 cols) to produce a dense 64-dim vector. A separate positional embedding (2 rows × 64 cols) is added so the model knows which operand is which.

Both tables are learned end-to-end via backpropagation — there is no hand-designed encoding."""
    , why: """Raw integers carry no geometric structure. The embedding gives each number a rich learned representation where modular-arithmetic patterns can emerge as directions in ℝ⁶⁴.

What the model learns: Research shows these models encode numbers as points on circles — the embeddings learn Fourier components of modular arithmetic. Addition mod 53 becomes rotation in this learned space.

Position embedding breaks symmetry — without it the model couldn't distinguish (a, b) from (b, a)."""
    , params: """tokEmbed  53 × 64  =  3,392
posEmbed   2 × 64  =    128
─────────────────────────────
Subtotal              3,520"""
    , hs: """embA = tokEmbed LA.! tokA        -- row lookup
embB = tokEmbed LA.! tokB
x0   = embA + (posEmbed LA.! 0)  -- add position
x1   = embB + (posEmbed LA.! 1)
-- embedded = [x0, x1]"""
    , shapes: """a, b  ∈  ℤ  (scalars in [0..52])
         ↓  lookup + elementwise add
[x₀, x₁]  ∈  ℝ²ˣ⁶⁴  (2 tokens, 64-dim each)"""
    }
  , { id: "attn"
    , label: "Attention"
    , sub: "Single-Head Self-Attention"
    , color: c.purple
    , bg: c.purpleBg
    , what: """Every token is linearly projected into a Query, Key, and Value:
  Q = Wq·x + bq,   K = Wk·x + bk,   V = Wv·x + bv

Dot-product scores QKᵀ/√64 form a 2×2 attention matrix:

        attends to:
            tok₀   tok₁
  tok₀ │  0.3    0.7  │  ← tok₀ gets 70% of tok₁'s info
  tok₁ │  0.6    0.4  │  ← tok₁ gets 60% of tok₀'s info

After softmax, each row sums to 1. Each token's output is a weighted sum of Values, projected through Wₒ.

Includes residual connection and LayerNorm (post-norm)."""
    , why: """The two numbers need to "talk" to compute their sum. Attention is the only place this happens — the MLP processes each token independently.

The 2×2 attention matrix controls how much of operand b's representation gets mixed into operand a's (and vice versa). No causal mask — both tokens see each other freely, appropriate since addition is commutative."""
    , params: """Wq  64×64 + bq 64  =  4,160
Wk  64×64 + bk 64  =  4,160
Wv  64×64 + bv 64  =  4,160
Wₒ  64×64 + bₒ 64  =  4,160
LayerNorm γ,β      =    128
────────────────────────────
Subtotal             16,768"""
    , hs: """qs = map (linearForward attnWq) xs
ks = map (linearForward attnWk) xs
vs = map (linearForward attnWv) xs

-- scale factor
sf = 1 / sqrt (fromIntegral dK)

-- 2×2 attention weights
weights = [ softmax $ scale sf $
              fromList [qi <.> kj | kj <- ks]
          | qi <- qs ]

-- weighted sum of values
attended = [ Σⱼ wᵢⱼ · vⱼ | wᵢ <- weights ]

-- output projection + residual + layernorm
outs = map (linearForward attnWo) attended
residual = zipWith (+) embedded outs
ln1Out = map (layerNormForward ln1) residual"""
    , shapes: """Q, K, V      ∈  ℝ²ˣ⁶⁴
raw scores   ∈  ℝ²ˣ²   (before softmax)
weights      ∈  ℝ²ˣ²   (after softmax, rows sum to 1)
attended     ∈  ℝ²ˣ⁶⁴
attnOut      ∈  ℝ²ˣ⁶⁴  (after Wₒ, residual, LayerNorm)"""
    }
  , { id: "mlp"
    , label: "MLP"
    , sub: "64 → 256 → ReLU → 64"
    , color: c.orange
    , bg: c.orangeBg
    , what: """Two linear layers with ReLU between them, applied independently to each token:
  hidden = ReLU(W₁ · x + b₁)      64 → 256  (expand)
  out    = W₂ · hidden + b₂       256 → 64   (compress)

The 4× expansion ratio (64 → 256) is standard. ReLU zeroes negative activations, providing the essential nonlinearity.

Includes residual connection and LayerNorm (post-norm)."""
    , why: """Attention mixes information across tokens; the MLP transforms each token's blended representation through a nonlinear bottleneck. The expansion to 256 dims gives the network enough parameters and capacity to learn the complex modular-arithmetic mapping.

This is where most of the model's parameters live (33K of ~57K total). The "grokking" phenomenon likely involves these weights slowly finding Fourier-like representations of mod-53 arithmetic."""
    , params: """W₁  256×64 + b₁ 256  =  16,640
W₂   64×256 + b₂  64  =  16,448
LayerNorm γ,β        =     128
──────────────────────────────
Subtotal                33,216"""
    , hs: """ffnForward FFNParams{..} x =
  let h   = linearForward ffnLinear1 x   -- 64 → 256
      act = relu h                        -- max(0, ·)
      out = linearForward ffnLinear2 act  -- 256 → 64
  in  (out, FFNCache h act)

-- with residual + layernorm
residual2 = zipWith (+) ln1Out ffnOut
ln2Out = map (layerNormForward ln2) residual2"""
    , shapes: """Per token:  ℝ⁶⁴ → ℝ²⁵⁶ → ℝ⁶⁴
(the 256-dim hidden is cached for backprop)"""
    }
  , { id: "unembed"
    , label: "Unembed"
    , sub: "Linear 64 → 53 + Softmax"
    , color: c.pink
    , bg: c.pinkBg
    , what: """First, take only token 0's representation (the "answer" position).

Then a single linear layer projects from ℝ⁶⁴ to ℝ⁵³ (one logit per possible answer class). Softmax converts raw logits to a valid probability distribution over [0..52].

  readout = head tokens           (take token 0)
  logits = W · readout + b        (64 → 53)
  probs  = softmax(logits)"""
    , why: """This "decodes" the model's internal 64-dim representation back into the answer space. Each of the 53 outputs corresponds to one residue class mod 53. The softmax ensures probabilities sum to 1, making the loss well-defined.

Token 0 is designated as the "answer" position — through attention, it has already gathered all the information it needs from token 1."""
    , params: """W  53×64 + b 53  =  3,445
────────────────────────────
Subtotal            3,445"""
    , hs: """readout = head ln2Out  -- first token only
logits = linearForward unembed readout
probs  = softmax logits"""
    , shapes: """ℝ²ˣ⁶⁴  →  ℝ⁶⁴ (token 0)  →  logits ∈ ℝ⁵³  →  probs ∈ Δ⁵²
(Δ⁵² = probability simplex, sums to 1)"""
    }
  , { id: "result"
    , label: "Result"
    , sub: "argmax(probs) → (a + b) mod 53"
    , color: c.green
    , bg: c.greenBg
    , what: """The predicted answer is simply the argmax of the probability distribution:

  prediction = argmax(probs)

During training, cross-entropy loss = −log(p_target).

THE GROKKING PHENOMENON:
The model first memorizes the training set:
  → ~95% train accuracy, ~0% test accuracy

Then, after many more epochs, it suddenly generalizes:
  → ~99% train accuracy, ~99% test accuracy

This phase transition requires weight decay — without it, the model stays stuck in memorization forever."""
    , why: """Cross-entropy penalises wrong predictions (p_target → 0 means loss → ∞).

Backpropagation: compute ∂loss/∂θ for every parameter θ by applying the chain rule backwards through the network — from loss → unembed → MLP → attention → embedding.

AdamW then uses these gradients to update parameters:
• Adaptive lr + momentum → faster, stable convergence
• Weight decay → critical for grokking; without it (wd ≈ 1e-3) the model memorizes but never generalizes"""
    , params: """0 learnable parameters (objective function)

AdamW: β₁=0.9, β₂=0.999, wd=1e-3"""
    , hs: """-- inference
prediction = maxIndex probs

-- training loss
crossEntropyLoss probs target =
  let p = max 1e-12 (probs ! target)
  in  negate (log p)

-- AdamW update (per parameter)
adamW m v g t θ =
  let m' = β1 * m + (1 - β1) * g
      v' = β2 * v + (1 - β2) * g * g
      mHat = m' / (1 - β1^t)
      vHat = v' / (1 - β2^t)
      θ'  = θ - lr * (mHat / (sqrt vHat + ε) + wd * θ)
  in  (θ', m', v')"""
    , shapes: """probs ∈ Δ⁵²  →  prediction ∈ [0..52]
              ↓  (training)
         loss ∈ ℝ  (scalar)
              ↓  backprop
       ∇θ loss ∈ ℝ⁵⁶ᐧ⁹⁴⁹  (gradients for all params)
              ↓  AdamW
          Δθ  ∈ ℝ⁵⁶ᐧ⁹⁴⁹  (parameter updates)"""
    }
  ]

findBlock :: String -> Maybe Block
findBlock id = find (\b -> b.id == id) blocks

-- ════════════════════════════════════════════════════════════════════════════
-- COMPONENT STATE & ACTIONS
-- ════════════════════════════════════════════════════════════════════════════

data Tab = What | Why | Params | Shapes | Hs

derive instance eqTab :: Eq Tab

type State =
  { selected :: Maybe String
  , activeTab :: Tab
  }

data Action
  = SelectBlock String
  | SetTab Tab

-- ════════════════════════════════════════════════════════════════════════════
-- COMPONENT
-- ════════════════════════════════════════════════════════════════════════════

component :: forall q i o m. MonadAff m => H.Component q i o m
component = H.mkComponent
  { initialState: \_ -> { selected: Nothing, activeTab: What }
  , render
  , eval: H.mkEval $ H.defaultEval { handleAction = handleAction }
  }

handleAction :: forall o m. MonadAff m => Action -> H.HalogenM State Action () o m Unit
handleAction = case _ of
  SelectBlock id -> H.modify_ \s -> s { selected = Just id, activeTab = What }
  SetTab tab -> H.modify_ \s -> s { activeTab = tab }

-- ════════════════════════════════════════════════════════════════════════════
-- RENDER
-- ════════════════════════════════════════════════════════════════════════════

render :: forall m. State -> H.ComponentHTML Action () m
render state =
  HH.div
    [ style $ "display: flex; flex-direction: column; height: 100vh; width: 100vw; " <>
        "background: " <> c.bg <> "; color: " <> c.txt <> "; overflow: hidden; " <>
        "font-family: 'Geist', 'IBM Plex Sans', system-ui, sans-serif;"
    ]
    [ renderHeader
    , HH.div
        [ style "display: flex; flex: 1; min-height: 0; overflow: hidden;" ]
        [ renderDiagramPane state
        , renderInfoPane state
        ]
    , renderFooter
    ]

renderHeader :: forall w i. HH.HTML w i
renderHeader =
  HH.div
    [ style $ "padding: 13px 22px; border-bottom: 1px solid " <> c.panelBdr <> "; " <>
        "display: flex; align-items: baseline; gap: 14px; flex-shrink: 0;"
    ]
    [ HH.span
        [ style $ "font-size: 15px; font-weight: 800; color: " <> c.bright <> "; letter-spacing: -0.02em;" ]
        [ HH.text "Modular Arithmetic Transformer" ]
    , HH.span
        [ style $ "font-size: 11px; color: " <> c.dim <> ";" ]
        [ HH.text "1 layer · 1 head · d=64 · (a+b) mod 53" ]
    , HH.span
        [ style $ "margin-left: auto; font-size: 10.5px; color: " <> c.dim <> "; " <>
            "font-family: 'JetBrains Mono', monospace;"
        ]
        [ HH.text $ totalParams <> " params" ]
    ]

renderFooter :: forall w i. HH.HTML w i
renderFooter =
  HH.div
    [ style $ "padding: 7px 22px; border-top: 1px solid " <> c.panelBdr <> "; " <>
        "font-size: 10px; color: " <> c.dim <> "; display: flex; gap: 20px; flex-shrink: 0; " <>
        "font-family: 'JetBrains Mono', monospace;"
    ]
    [ HH.span_ [ HH.text "post-norm · no causal mask · AdamW + cosine LR" ]
    , HH.span [ style "margin-left: auto;" ]
        [ HH.text "p=53 · dModel=64 · dFF=256 · dK=64 · batch=32" ]
    ]

-- ════════════════════════════════════════════════════════════════════════════
-- DIAGRAM PANE (SVG)
-- ════════════════════════════════════════════════════════════════════════════

-- Layout constants
svgWidth :: Number
svgWidth = 560.0

cxPos :: Number
cxPos = svgWidth / 2.0

blockWidth :: Number
blockWidth = 330.0

blockHeight :: Number
blockHeight = 54.0

gap :: Number
gap = 8.0

yEmbed :: Number
yEmbed = 40.0

yAttn :: Number
yAttn = 130.0

yMlp :: Number
yMlp = 220.0

yUnembed :: Number
yUnembed = 310.0

yResult :: Number
yResult = 400.0

svgHeight :: Number
svgHeight = yResult + blockHeight + 40.0

renderDiagramPane :: forall m. State -> H.ComponentHTML Action () m
renderDiagramPane state =
  HH.div
    [ style $ "flex: 0 0 52%; overflow-y: auto; padding: 14px 6px; " <>
        "border-right: 1px solid " <> c.panelBdr <> ";"
    ]
    [ renderSvg state ]

renderSvg :: forall m. State -> H.ComponentHTML Action () m
renderSvg state =
  SE.svg
    [ SA.viewBox 0.0 0.0 svgWidth svgHeight
    , style "display: block; width: 100%;"
    ]
    [ -- Arrow marker definition
      SE.defs []
        [ SE.marker
            [ SA.id "ah"
            , SA.markerWidth 8.0
            , SA.markerHeight 6.0
            , SA.refX 7.0
            , SA.refY 3.0
            , SA.orient SA.AutoOrient
            ]
            [ SE.polygon [ SA.points [ Tuple 0.0 0.0, Tuple 8.0 3.0, Tuple 0.0 6.0 ]
                         , style $ "fill: " <> c.dim <> ";" ] ]
        , SE.marker
            [ SA.id "ah-back"
            , SA.markerWidth 8.0
            , SA.markerHeight 6.0
            , SA.refX 1.0
            , SA.refY 3.0
            , SA.orient SA.AutoOrient
            ]
            [ SE.polygon [ SA.points [ Tuple 8.0 0.0, Tuple 0.0 3.0, Tuple 8.0 6.0 ]
                         , style $ "fill: " <> c.orange <> ";" ] ]
        ]
    -- Input label
    , SE.text
        [ SA.x cxPos, SA.y 24.0
        , SA.textAnchor SA.AnchorMiddle
        , SA.fill $ SA.Named c.bright
              , SA.fontWeight SFW.FWeightBold
        , style "font-size: 13px; font-family: 'JetBrains Mono', monospace;"
        ]
        [ HH.text "Inputs:  a, b  ∈ [0..52]" ]
    -- Blocks
    , renderBlock "embed" yEmbed state.selected
    , renderArrow (yEmbed + blockHeight + gap) (yAttn - gap)
    , renderBlock "attn" yAttn state.selected
    , renderArrow (yAttn + blockHeight + gap) (yMlp - gap)
    , renderBlock "mlp" yMlp state.selected
    , renderArrow (yMlp + blockHeight + gap) (yUnembed - gap)
    , renderBlock "unembed" yUnembed state.selected
    , renderArrow (yUnembed + blockHeight + gap) (yResult - gap)
    , renderBlock "result" yResult state.selected
    -- Dimension labels
    , renderDim (cxPos + blockWidth / 2.0 + 14.0) (yEmbed + 32.0) "ℝ²ˣ⁶⁴"
    , renderDim (cxPos + blockWidth / 2.0 + 14.0) (yAttn + 32.0) "ℝ²ˣ⁶⁴"
    , renderDim (cxPos + blockWidth / 2.0 + 14.0) (yMlp + 32.0) "ℝ²ˣ⁶⁴"
    , renderDim (cxPos + blockWidth / 2.0 + 14.0) (yUnembed + 32.0) "ℝ⁵³"
    , renderDim (cxPos + blockWidth / 2.0 + 14.0) (yResult + 32.0) "[0..52]"
    -- AdamW backprop arrow
    , SE.path
        [ SA.d [ SA.m SA.Abs (cxPos - blockWidth / 2.0 - 20.0) (yResult + blockHeight / 2.0)
               , SA.l SA.Abs (cxPos - blockWidth / 2.0 - 20.0) (yEmbed + blockHeight / 2.0)
               ]
        , SA.fill SA.NoColor
        , SA.stroke $ SA.Named c.orange
        , SA.strokeWidth 2.0
        , SA.strokeDashArray "6,4"
        , SA.markerEnd "url(#ah-back)"
        , style "opacity: 0.7;"
        ]
    , SE.text
        [ SA.transform [ SA.Translate (cxPos - blockWidth / 2.0 - 32.0) ((yEmbed + yResult + blockHeight) / 2.0)
                       , SA.Rotate (-90.0) 0.0 0.0
                       ]
        , SA.textAnchor SA.AnchorMiddle
        , SA.fill $ SA.Named c.orange
        , style "font-size: 10px; font-family: 'JetBrains Mono', monospace; opacity: 0.8;"
        ]
        [ HH.text "AdamW backprop" ]
    -- Bottom label
    , SE.text
        [ SA.x cxPos, SA.y (svgHeight - 8.0)
        , SA.textAnchor SA.AnchorMiddle
        , SA.fill $ SA.Named c.bright
        , style "font-size: 12px; font-family: 'JetBrains Mono', monospace; opacity: 0.6;"
        ]
        [ HH.text "prediction:  (a + b) mod 53" ]
    ]

renderBlock :: forall m. String -> Number -> Maybe String -> H.ComponentHTML Action () m
renderBlock id y selected =
  case findBlock id of
    Nothing -> SE.g [] []
    Just block ->
      let
        isSelected = selected == Just id
        fillColor = if isSelected then block.bg else c.card
        strokeColor = if isSelected then block.color else c.bdr
        strokeW = if isSelected then 2.5 else 1.5
      in
        SE.g
          [ HE.onClick \_ -> SelectBlock id
          , style "cursor: pointer;"
          ]
          [ SE.rect
              [ SA.x (cxPos - blockWidth / 2.0)
              , SA.y y
              , SA.width blockWidth
              , SA.height blockHeight
              , SA.rx 6.0
              , SA.fill $ SA.Named fillColor
              , SA.stroke $ SA.Named strokeColor
              , SA.strokeWidth strokeW
              , style "transition: all 0.15s;"
              ]
          , SE.text
              [ SA.x cxPos, SA.y (y + 22.0)
              , SA.textAnchor SA.AnchorMiddle
              , SA.fill $ SA.Named block.color
        , SA.fontWeight SFW.FWeightBold
              , style "font-size: 14px; font-family: 'JetBrains Mono', 'Fira Code', monospace;"
              ]
              [ HH.text block.label ]
          , SE.text
              [ SA.x cxPos, SA.y (y + 40.0)
              , SA.textAnchor SA.AnchorMiddle
              , SA.fill $ SA.Named c.dim
              , style "font-size: 10px; font-family: 'JetBrains Mono', monospace;"
              ]
              [ HH.text block.sub ]
          ]

renderArrow :: forall w i. Number -> Number -> HH.HTML w i
renderArrow y1 y2 =
  SE.line
    [ SA.x1 cxPos, SA.y1 y1
    , SA.x2 cxPos, SA.y2 y2
    , SA.stroke $ SA.Named c.dim
    , SA.strokeWidth 1.5
    , SA.markerEnd "url(#ah)"
    ]

renderDim :: forall w i. Number -> Number -> String -> HH.HTML w i
renderDim x y txt =
  SE.text
    [ SA.x x, SA.y y
    , SA.fill $ SA.Named c.dim
    , style "font-size: 10px; font-family: 'JetBrains Mono', monospace; opacity: 0.6;"
    ]
    [ HH.text txt ]

-- ════════════════════════════════════════════════════════════════════════════
-- INFO PANE
-- ════════════════════════════════════════════════════════════════════════════

renderInfoPane :: forall m. State -> H.ComponentHTML Action () m
renderInfoPane state =
  HH.div
    [ style $ "flex: 1; overflow-y: auto; background: " <> c.panel <> ";" ]
    [ case state.selected >>= findBlock of
        Nothing -> renderEmptyInfo
        Just block -> renderBlockInfo block state.activeTab
    ]

renderEmptyInfo :: forall w i. HH.HTML w i
renderEmptyInfo =
  HH.div
    [ style $ "padding: 40px 28px; color: " <> c.dim <> "; " <>
        "font-family: 'Geist', 'IBM Plex Sans', system-ui, sans-serif; " <>
        "text-align: center; line-height: 1.8;"
    ]
    [ HH.div [ style "font-size: 40px; margin-bottom: 16px; opacity: 0.18;" ] [ HH.text "◆" ]
    , HH.div [ style "font-size: 14px; font-weight: 500;" ] [ HH.text "Click any block to explore it" ]
    , HH.div
        [ style "font-size: 12px; margin-top: 10px; opacity: 0.55; line-height: 1.7;" ]
        [ HH.text "Each panel shows "
        , HH.em_ [ HH.text "what" ]
        , HH.text " the layer does, "
        , HH.em_ [ HH.text "why" ]
        , HH.text " it's there, parameter counts, tensor shapes, and the actual Haskell code."
        ]
    , HH.div
        [ style $ "margin-top: 28px; padding: 14px 18px; background: #0a0e18; border-radius: 6px; " <>
            "border: 1px solid " <> c.panelBdr <> "; font-size: 11.5px; text-align: left; line-height: 1.8; " <>
            "font-family: 'JetBrains Mono', monospace;"
        ]
        [ HH.div [ style $ "color: " <> c.dim <> "; margin-bottom: 4px;" ]
            [ HH.text "Total learnable parameters" ]
        , HH.div [ style $ "color: " <> c.accent <> "; font-size: 16px; font-weight: 700;" ]
            [ HH.text totalParams ]
        , HH.div [ style $ "color: " <> c.dim <> "; margin-top: 10px; font-size: 10.5px; line-height: 1.7;" ]
            [ HH.text paramBreakdown ]
        ]
    ]

renderBlockInfo :: forall m. Block -> Tab -> H.ComponentHTML Action () m
renderBlockInfo block activeTab =
  HH.div
    [ style "font-family: 'Geist', 'IBM Plex Sans', system-ui, sans-serif;" ]
    [ -- Header with color dot
      HH.div
        [ style $ "padding: 16px 22px 12px; border-bottom: 1px solid " <> c.panelBdr <> "; " <>
            "display: flex; align-items: center; gap: 10px;"
        ]
        [ HH.div
            [ style $ "width: 10px; height: 10px; border-radius: 50%; " <>
                "background: " <> block.color <> "; box-shadow: 0 0 10px " <> block.color <> "44; flex-shrink: 0;"
            ] []
        , HH.div
            [ style $ "font-size: 14px; font-weight: 700; color: " <> c.bright <> "; letter-spacing: -0.01em;" ]
            [ HH.text block.label ]
        ]
    -- Tabs
    , HH.div
        [ style $ "display: flex; border-bottom: 1px solid " <> c.panelBdr <> "; padding: 0 14px;" ]
        [ renderTab "What" What activeTab block.color
        , renderTab "Why" Why activeTab block.color
        , renderTab "Params" Params activeTab block.color
        , renderTab "Shapes" Shapes activeTab block.color
        , renderTab "Haskell" Hs activeTab block.color
        ]
    -- Content
    , renderTabContent block activeTab
    ]

renderTab :: forall m. String -> Tab -> Tab -> String -> H.ComponentHTML Action () m
renderTab label tab activeTab accentColor =
  HH.button
    [ HE.onClick \_ -> SetTab tab
    , style $ "background: none; border: none; cursor: pointer; " <>
        "padding: 10px 13px; font-size: 11.5px; font-weight: 600; " <>
        "font-family: 'Geist', 'IBM Plex Sans', system-ui, sans-serif; " <>
        "color: " <> (if tab == activeTab then accentColor else c.dim) <> "; " <>
        "border-bottom: 2px solid " <> (if tab == activeTab then accentColor else "transparent") <> "; " <>
        "margin-bottom: -1px; transition: color 0.15s;"
    ]
    [ HH.text label ]

renderTabContent :: forall w i. Block -> Tab -> HH.HTML w i
renderTabContent block tab =
  let
    content = case tab of
      What -> block.what
      Why -> block.why
      Params -> block.params
      Shapes -> block.shapes
      Hs -> block.hs
    isMono = tab == Hs || tab == Params || tab == Shapes
    isCode = tab == Hs
    baseStyle = "padding: 18px 22px; font-size: 13px; line-height: 1.8; " <>
                "color: " <> c.txt <> "; white-space: pre-wrap; "
    fontStyle = if isMono
                then "font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 12.5px; line-height: 1.7;"
                else ""
    codeStyle = if isCode
                then "font-size: 12px; line-height: 1.65; background: #060a12; padding: 18px 22px;"
                else ""
  in
    HH.div
      [ style $ baseStyle <> fontStyle <> codeStyle ]
      [ HH.text content ]

-- ════════════════════════════════════════════════════════════════════════════
-- HELPERS
-- ════════════════════════════════════════════════════════════════════════════

style :: forall r i. String -> HP.IProp ( style :: String | r ) i
style = HP.attr (H.AttrName "style")

-- ════════════════════════════════════════════════════════════════════════════
-- MAIN
-- ════════════════════════════════════════════════════════════════════════════

main :: Effect Unit
main = HA.runHalogenAff do
  mRoot <- liftEffect do
    win <- window
    doc <- document win
    let docNode = toNonElementParentNode (toDocument doc)
    getElementById "root" docNode
  case mRoot >>= fromElement of
    Nothing -> pure unit
    Just root -> void $ runUI component unit root
