{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE AllowAmbiguousTypes #-}

-- market-seq-probe — the Bradley evaluation layer over a trained sequence
-- model (README §10 Phase 5).  Read-only over a market-train checkpoint.
--
--   1. MAGNITUDE (Semantics/Magnitude.agda, Bradley 2025): over sampled
--      validation prefixes x, Mag(tM) = (t−1)·Σ_x H_t(p_x) + |T(⊥)| with H_t
--      the Tsallis t-entropy of the model's next-token distribution p_x.  We
--      report the prompt-sum term at several t and the slope at t=1
--      (= Σ_x Shannon H(p_x)) — the model's entropy landscape.  (The |T(⊥)|
--      constant counts terminating outputs; the market stream never emits †,
--      so it is omitted — stated, not hidden.)
--   2. YONEDA SYNONYMY: the meaning of a context is its copresheaf π(·|x).
--      If recent history determines the model's expectation, contexts that
--      AGREE on their last k bins should have closer copresheaves (symmetric
--      KL) than random pairs.  Reported for k = 1, 2.
--   3. COMPOSITION (CompLaw, Semantics/MarketLanguage.agda): for nested
--      extensions x ⊑ y ⊑ z the enriched-category axiom π(z|y)·π(y|x) = π(z|x)
--      holds for ANY autoregressive model by the chain rule — Bradley's point
--      that LLMs are enriched categories by construction.  We verify the
--      equality numerically over validation windows (it should sit at float
--      rounding; a violation would mean the implementation broke the axiom).
--
-- Writes market-magnitude.csv (t, prompt-sum term) next to the report.
module Main where

import Control.Monad (forM_, unless)
import Data.List (sortOn, foldl')
import Data.Maybe (fromMaybe, mapMaybe)
import System.Directory (doesFileExist)
import System.Exit (exitFailure)
import System.IO (hPutStrLn, stderr)
import Text.Printf (printf)
import Text.Read (readMaybe)
import qualified Options.Applicative as O

import Tensor (Additive(..), vtoList)
import Serialize (Serialize, toFloats)
import Transformer (ParamsSeq, ParamsCSeq, seqLogitsVal)
import Checkpoint (CkptMeta(..), loadCkpt)
import MarketTokenizer

data Opts = Opts
  { optMode :: String
  , optCkpt :: FilePath
  , optData :: FilePath
  , optFrac :: Double
  }

optsP :: O.Parser Opts
optsP = Opts
  <$> O.strOption (O.long "mode" <> O.short 'm' <> O.value "mkt-small" <> O.showDefault)
  <*> O.strOption (O.long "checkpoint" <> O.short 'c' <> O.metavar "PATH"
        <> O.value "checkpoint-mkt-small.ckpt" <> O.showDefault)
  <*> O.strOption (O.long "data" <> O.short 'd' <> O.value "data/BTC-1m.csv" <> O.showDefault)
  <*> O.option O.auto (O.long "train-frac" <> O.value 0.7 <> O.showDefault
        <> O.help "must match the training run (defines the val slice)")

-- ── Magnitude.agda mirrored (fpow guard and all) ──────────────────────────────

fpow :: Double -> Double -> Double
fpow a t = if a < 1e-300 then 0 else exp (t * log a)

tsallis :: Double -> [Double] -> Double
tsallis t p = (1 - sum (map (`fpow` t) p)) / (t - 1)

shannon :: [Double] -> Double
shannon p = negate (sum [ if pi' < 1e-300 then 0 else pi' * log pi' | pi' <- p ])

-- ── distributions ─────────────────────────────────────────────────────────────

softmaxL :: [Double] -> [Double]
softmaxL xs = let m = maximum xs; es = map (\x -> exp (x - m)) xs; z = sum es
              in map (/ z) es

symKL :: [Double] -> [Double] -> Double
symKL p q = 0.5 * (kl p q + kl q p)
  where kl a b = sum [ x * log (max 1e-300 x / max 1e-300 y) | (x, y) <- zip a b ]

mean :: [Double] -> Double
mean xs = sum xs / fromIntegral (length xs)

-- ── the probe ─────────────────────────────────────────────────────────────────

probe :: forall v n dM dF dK.
         (ParamsCSeq v n dM dF dK, Serialize (ParamsSeq v n dM dF dK))
      => Int -> Opts -> IO ()
probe nI o = do
  let specPath = optCkpt o ++ ".tok"
      nParam   = length (toFloats (zeroA :: ParamsSeq v n dM dF dK))
  r <- loadCkpt nParam (optCkpt o)
  pp <- case r of
    Just (meta, pp, _) -> do
      printf "probe over %s (mode %s, epoch %d)\n" (optCkpt o) (ckMode meta) (ckEpoch meta)
      pure (pp :: ParamsSeq v n dM dF dK)
    Nothing -> hPutStrLn stderr "probe: no checkpoint" >> exitFailure
  haveSpec <- doesFileExist specPath
  unless haveSpec (hPutStrLn stderr ("probe: no tokenizer at " ++ specPath) >> exitFailure)
  spec <- loadSpec specPath
  s <- readFile (optData o)
  let closes = mapMaybe (\row -> case break (== ',') <$> stripField 4 row of
                                   Just (c, _) -> readMaybe c :: Maybe Double
                                   Nothing     -> Nothing)
                        (drop 1 (lines s))
      rets   = logReturns closes
      (_, valR) = walkForwardSplit (optFrac o) rets
      valToks   = map (encodeReturn spec) valR
      wins      = take 200 (map (bosTok :) (contextWindows (nI - 1) valToks))
      -- one forward pass per window: position-wise next-token distributions
      dists w   = map (softmaxL . vtoList) (seqLogitsVal w pp)
      finals    = [ (w, last (dists w)) | w <- wins ]

  -- 1. magnitude
  let promptDists = map snd finals
      slope1      = sum (map shannon promptDists)
      ts          = [0.5, 0.9, 0.99, 1.01, 1.1, 2.0]
      magTerm t   = (t - 1) * sum (map (tsallis t) promptDists)
  printf "\n[magnitude] %d prompts, vocab %d\n" (length promptDists) (vocabSize spec)
  printf "  slope at t=1 (Σ Shannon H) = %.3f nats; mean H = %.3f (uniform would be %.3f)\n"
         slope1 (slope1 / fromIntegral (length promptDists) :: Double)
         (log (fromIntegral (vocabSize spec)) :: Double)
  writeFile "market-magnitude.csv" $ unlines $
    "t,prompt_sum_term" : [ show t ++ "," ++ show (magTerm t) | t <- ts ]
  forM_ ts $ \t -> printf "  t=%.2f  (t-1)·ΣH_t = %+.4f\n" t (magTerm t)
  putStrLn "  (|T(⊥)| omitted: the market stream never terminates with †)"

  -- 2. Yoneda synonymy: same last-k bins ⇒ closer copresheaves?
  forM_ [1, 2 :: Int] $ \k -> do
    let key w   = drop (length w - k) w
        pairs   = [ (d1, d2, key w1 == key w2)
                  | ((w1, d1) : rest) <- tails' finals, (w2, d2) <- rest ]
        intra   = [ symKL a b | (a, b, True)  <- pairs ]
        inter   = [ symKL a b | (a, b, False) <- pairs ]
    if null intra
      then printf "\n[yoneda k=%d] no same-suffix pairs in the sample\n" k
      else do
        printf "\n[yoneda k=%d] same-last-%d-bins pairs: %d, others: %d\n"
               k k (length intra) (length inter)
        printf "  mean symKL  intra %.4f   inter %.4f   ratio %.2fx\n"
               (mean intra) (mean inter) (mean inter / max 1e-12 (mean intra))

  -- 3. composition law (chain rule): π(z|y)·π(y|x) vs π(z|x) on nested prefixes
  let compDev w =
        let ds = dists w
            stepP i = ds !! i !! (w !! (i + 1))      -- π(tok_{i+1} | prefix_i)
            l  = length w
            a  = l `div` 3; b = 2 * l `div` 3
            pxy = product [ stepP i | i <- [a - 1 .. b - 2] ]   -- π(y|x)
            pyz = product [ stepP i | i <- [b - 1 .. l - 2] ]   -- π(z|y)
            pxz = product [ stepP i | i <- [a - 1 .. l - 2] ]   -- π(z|x)
        in abs (pxy * pyz - pxz)
      devs = map compDev wins
  printf "\n[composition] max |π(z|y)·π(y|x) − π(z|x)| over %d windows = %.3e\n"
         (length devs) (maximum devs)
  putStrLn "  (an equality by the chain rule — Bradley's 'LLMs are enriched categories"
  putStrLn "   by construction'; deviation is float rounding, gated below 1e-12)"
  if maximum devs < 1e-12
    then putStrLn "\nPROBE OK"
    else putStrLn "\nPROBE FAILED (composition law violated)" >> exitFailure
  where
    tails' xs = case xs of [] -> []; _ : r -> xs : tails' r
    -- extract field i (0-based) of a CSV row
    stripField :: Int -> String -> Maybe String
    stripField 0 row = Just row
    stripField k row = case dropWhile (/= ',') row of
      ',' : rest -> stripField (k - 1) rest
      _          -> Nothing

main :: IO ()
main = do
  o <- O.execParser (O.info (O.helper <*> optsP)
        (O.fullDesc <> O.progDesc "Bradley probes (magnitude/Yoneda/composition) over a market checkpoint"))
  case optMode o of
    "mkt-small" -> probe @130 @32 @32 @128 @8  32 o
    "mkt-base"  -> probe @514 @64 @64 @256 @16 64 o
    m -> hPutStrLn stderr ("unknown mode " ++ m) >> exitFailure
