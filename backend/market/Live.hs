{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE BangPatterns #-}

-- market-live — live online learning + per-trade trading recommendations
-- (README §10; approved plan 2026-06-09).
--
-- The model is the conformance-gated sequence transformer, warm-started from
-- a market-train checkpoint (+ its .tok tokenizer artifact, which defines the
-- language).  The live trade stream (vendored Hyperliquid client) drives two
-- cadences:
--
--   * EVERY TRADE: the in-progress candle's return-so-far becomes a
--     PROVISIONAL last token; the model's copresheaf at that prefix gives the
--     expected next return μ, P(up), and entropy; a FLAT/LONG/SHORT paper
--     position state machine maps μ against --threshold-bps (net of
--     --fee-bps) to OPEN_LONG / OPEN_SHORT / CLOSE / FLIP_* / HOLD.  One JSON
--     line per trade on stdout; humans get stderr lines on non-HOLD actions.
--
--   * EVERY CANDLE CLOSE (1m default): the completed candle is one new token
--     of the verified language — ONE online gradient step (the same verified
--     seqGradLoss morphism + AdamW, constant --lr, global-norm clip), and the
--     checkpoint + paper-position state are saved every --save-every candles.
--
-- The online training RECIPE is harness (out of conformance scope, README
-- §4); the model + gradient stay the verified morphism.  Recommendations
-- only — no order placement (the signing path is unimplemented); paper PnL
-- (log-return units, net of fees) is tracked so the rule's value is
-- measurable.  --replay drives the identical pipeline from a backfill CSV
-- (verification + fee-aware paper backtest, no network).
module Main where

import Control.Concurrent (forkIO)
import Control.Concurrent.STM
  (TQueue, atomically, newTQueueIO, writeTQueue, readTQueue, flushTQueue)
import Control.Monad (forM_, unless, when)
import qualified Data.Aeson as A
import Data.Aeson ((.=))
import qualified Data.ByteString.Lazy.Char8 as BL8
import Data.List (foldl', sortOn)
import Data.Maybe (fromMaybe, mapMaybe)
import qualified Data.Text as T
import System.Directory (doesFileExist)
import System.Exit (exitFailure)
import System.IO (hFlush, hPutStrLn, stdout, stderr)
import Text.Printf (printf)
import Text.Read (readMaybe)
import qualified Options.Applicative as O

import Tensor (Additive(..), Scale(..), vtoList)
import Serialize (Serialize, toFloats, fromFloats)
import Transformer (ParamsSeq, ParamsCSeq, seqLogitsVal, seqGradLoss)
import Optimizer (AdamConfig(..), AdamState(..), Adam, adamStep)
import Checkpoint (CkptMeta(..), saveCkpt, loadCkpt)
import MarketTokenizer
import Hyperliquid.Types (Trade(..), WsMessage(..))
import Hyperliquid.WebSocket (streamMarketData)

-- ── CLI ───────────────────────────────────────────────────────────────────────

data Opts = Opts
  { optMode      :: String
  , optCkpt      :: Maybe FilePath
  , optCoin      :: String
  , optData      :: Maybe FilePath   -- bootstrap context from a backfill CSV
  , optLr        :: Double
  , optSaveEvery :: Int
  , optThreshBps :: Double
  , optFeeBps    :: Double
  , optState     :: Maybe FilePath
  , optReplay    :: Maybe FilePath
  , optInterval  :: Int              -- seconds
  }

optsP :: O.Parser Opts
optsP = Opts
  <$> O.strOption (O.long "mode" <> O.short 'm' <> O.value "mkt-base" <> O.showDefault)
  <*> O.optional (O.strOption (O.long "checkpoint" <> O.short 'c' <> O.metavar "PATH"
        <> O.help "warm-start checkpoint (default checkpoint-<mode>.ckpt); its .tok must exist"))
  <*> O.strOption (O.long "coin" <> O.value "BTC" <> O.showDefault)
  <*> O.optional (O.strOption (O.long "data" <> O.short 'd' <> O.metavar "CSV"
        <> O.help "bootstrap the initial context from this backfill CSV's tail"))
  <*> O.option O.auto (O.long "lr" <> O.value 1.0e-4 <> O.showDefault
        <> O.help "constant online learning rate")
  <*> O.option O.auto (O.long "save-every" <> O.value 30 <> O.showDefault
        <> O.help "checkpoint/state save period, in candles")
  <*> O.option O.auto (O.long "threshold-bps" <> O.value 10 <> O.showDefault
        <> O.help "|expected return| needed to open/flip, basis points")
  <*> O.option O.auto (O.long "fee-bps" <> O.value 3.5 <> O.showDefault
        <> O.help "assumed taker fee per side, basis points")
  <*> O.optional (O.strOption (O.long "state" <> O.metavar "PATH"
        <> O.help "paper-position state file (default <checkpoint>.pos)"))
  <*> O.optional (O.strOption (O.long "replay" <> O.metavar "CSV"
        <> O.help "no network: drive the pipeline from a backfill CSV (paper backtest)"))
  <*> O.option O.auto (O.long "interval-secs" <> O.value 60 <> O.showDefault
        <> O.help "candle interval (the token clock); 60 matches the offline corpus")

-- ── paper position ────────────────────────────────────────────────────────────

data Side = Flat | Long | Short deriving (Show, Read, Eq)

data Pos = Pos
  { pSide    :: Side
  , pEntry   :: Double   -- entry price (0 when flat)
  , pPnl     :: Double   -- realized paper PnL, cumulative log-return net fees
  , pCandles :: Int      -- candles seen (online-training steps)
  } deriving (Show, Read)

pos0 :: Pos
pos0 = Pos Flat 0 0 0

data Action = OpenLong | OpenShort | Close | FlipLong | FlipShort | Hold
  deriving (Show, Eq)

-- the state machine: μ (expected next log-return) against θ
decide :: Double -> Double -> Side -> Action
decide th mu side = case side of
  Flat  | mu >  th  -> OpenLong
        | mu < -th  -> OpenShort
        | otherwise -> Hold
  Long  | mu < -th  -> FlipShort
        | mu <  0   -> Close
        | otherwise -> Hold
  Short | mu >  th  -> FlipLong
        | mu >  0   -> Close
        | otherwise -> Hold

-- apply an action at price px; fee charged per side, in log-return units
apply :: Double -> Double -> Action -> Pos -> Pos
apply fee px act p = case act of
  Hold      -> p
  OpenLong  -> p { pSide = Long,  pEntry = px, pPnl = pPnl p - fee }
  OpenShort -> p { pSide = Short, pEntry = px, pPnl = pPnl p - fee }
  Close     -> p { pSide = Flat,  pEntry = 0,  pPnl = pPnl p + settle - fee }
  FlipLong  -> apply fee px OpenLong  (apply fee px Close p)
  FlipShort -> apply fee px OpenShort (apply fee px Close p)
  where
    settle = case pSide p of
      Long  -> log (px / pEntry p)
      Short -> log (pEntry p / px)
      Flat  -> 0

unrealized :: Double -> Pos -> Double
unrealized px p = case pSide p of
  Long  -> log (px / pEntry p)
  Short -> log (pEntry p / px)
  Flat  -> 0

-- ── helpers ───────────────────────────────────────────────────────────────────

softmaxL :: [Double] -> [Double]
softmaxL xs = let m = maximum xs; es = map (\x -> exp (x - m)) xs; z = sum es
              in map (/ z) es

entropyL :: [Double] -> Double
entropyL ps = negate (sum [ p * log p | p <- ps, p > 1e-300 ])

clipGrad :: (Serialize p, Scale p) => Double -> p -> p
clipGrad c g = let n = sqrt (sum [ x * x | x <- toFloats g ])
               in if n > c then scaleA (c / n) g else g

takeLast :: Int -> [a] -> [a]
takeLast k xs = drop (length xs - k) xs

csvCloses :: FilePath -> IO [(Integer, Double)]   -- (open time ms, close)
csvCloses path = do
  s <- readFile path
  let row r = case splitOn ',' r of
        (t : _o : _h : _l : c : _) ->
          (,) <$> (readMaybe t :: Maybe Integer) <*> (readMaybe c :: Maybe Double)
        _ -> Nothing
  pure (mapMaybe row (drop 1 (lines s)))
  where splitOn c xs = case break (== c) xs of
          (a, [])    -> [a]
          (a, _ : b) -> a : splitOn c b

-- ── the engine (shared by live and replay) ───────────────────────────────────

data Live v n dM dF dK = Live
  { lvParams    :: !(ParamsSeq v n dM dF dK)
  , lvAdam      :: !(AdamState (ParamsSeq v n dM dF dK))
  , lvRing      :: ![Int]      -- last completed bin tokens, oldest first (≤ n−1)
  , lvPrevClose :: !Double     -- previous candle close
  , lvBucket    :: !Integer    -- current candle bucket (ts // interval)
  , lvLastPx    :: !Double     -- last trade price in the current bucket
  , lvPos       :: !Pos
  }

data Cfg = Cfg
  { cfgN        :: Int
  , cfgSpec     :: TokenizerSpec
  , cfgLr       :: Double
  , cfgFee      :: Double      -- per side, log-return units
  , cfgTh       :: Double      -- threshold, log-return units
  , cfgSaveEv   :: Int
  , cfgCkpt     :: FilePath
  , cfgStateF   :: FilePath
  , cfgMode     :: String
  , cfgIntMs    :: Integer
  }

adamCfg :: AdamConfig
adamCfg = AdamConfig 0.9 0.999 1.0e-8 1.0e-3

-- one trade: maybe close candle(s) (train), then recommend on the provisional window
onTrade :: forall v n dM dF dK.
           ( ParamsCSeq v n dM dF dK
           , Serialize (ParamsSeq v n dM dF dK)
           , Adam (ParamsSeq v n dM dF dK)
           , Scale (ParamsSeq v n dM dF dK) )
        => Cfg -> Bool -> Integer -> Double -> Live v n dM dF dK
        -> IO (Live v n dM dF dK)
onTrade cfg recommend ts px lv0 = do
  let bucket = ts `div` cfgIntMs cfg
  lv <- if bucket > lvBucket lv0 && lvBucket lv0 >= 0
          then closeCandle cfg lv0 >>= \l -> pure l { lvBucket = bucket, lvLastPx = px }
          else pure lv0 { lvBucket = max bucket (lvBucket lv0)
                        , lvLastPx = if bucket >= lvBucket lv0 then px else lvLastPx lv0 }
  if not recommend then pure lv else do
    let spec   = cfgSpec cfg
        prov   = encodeReturn spec (log (px / lvPrevClose lv))
        window = bosTok : takeLast (cfgN cfg - 1) (lvRing lv ++ [prov])
    if length window < 2 then pure lv else do
      let logits = vtoList (last (seqLogitsVal window (lvParams lv)))
          probs  = softmaxL logits
          binsP  = zip [0 ..] probs
          mu     = sum [ p * r | (t, p) <- binsP, Just r <- [decodeTok spec t] ]
          pUp    = sum [ p | (t, p) <- binsP, Just r <- [decodeTok spec t], r > 0 ]
          ent    = entropyL probs
          act    = decide (cfgTh cfg) mu (pSide (lvPos lv))
          pos'   = apply (cfgFee cfg) px act (lvPos lv)
      BL8.hPutStrLn stdout $ A.encode $ A.object
        [ "ts" .= ts, "px" .= px, "action" .= show act
        , "expRet" .= mu, "pUp" .= pUp, "entropy" .= ent
        , "position" .= show (pSide pos'), "entryPx" .= pEntry pos'
        , "paperPnl" .= pPnl pos', "unrealized" .= unrealized px pos'
        ]
      hFlush stdout
      when (act /= Hold) $ do
        hPutStrLn stderr $ printf "[action] %s %s @ %.1f  μ=%+.5f pUp=%.3f  realizedPnl=%+.5f"
          (show act) (cfgMode cfg) px mu pUp (pPnl pos')
        writeFile (cfgStateF cfg) (show pos')
      pure lv { lvPos = pos' }

-- candle close: the completed candle is one new token — train once, maybe save
closeCandle :: forall v n dM dF dK.
               ( ParamsCSeq v n dM dF dK
               , Serialize (ParamsSeq v n dM dF dK)
               , Adam (ParamsSeq v n dM dF dK)
               , Scale (ParamsSeq v n dM dF dK) )
            => Cfg -> Live v n dM dF dK -> IO (Live v n dM dF dK)
closeCandle cfg lv = do
  let spec  = cfgSpec cfg
      close = lvLastPx lv
      ret   = log (close / lvPrevClose lv)
      tok   = encodeReturn spec ret
      ring' = takeLast (cfgN cfg - 1) (lvRing lv ++ [tok])
      win   = bosTok : ring'
      pos'  = (lvPos lv) { pCandles = pCandles (lvPos lv) + 1 }
  (params', adam', mLoss) <-
    if length win >= 2
      then do
        let (g, l) = seqGradLoss win (lvParams lv)
            gC     = clipGrad 1.0 g
            (p', a') = adamStep adamCfg (cfgLr cfg) (lvParams lv) (lvAdam lv) gC
        pure (p', a', Just l)
      else pure (lvParams lv, lvAdam lv, Nothing)
  hPutStrLn stderr $ printf "[candle %d] close %.1f ret %+.5f tok %d  %s  pnl %+.5f"
    (pCandles pos') close ret tok
    (maybe "(warming up)" (printf "online loss %.4f") mLoss :: String)
    (pPnl pos')
  when (pCandles pos' `mod` cfgSaveEv cfg == 0) $ do
    saveCkpt (cfgCkpt cfg) (CkptMeta "market" (cfgMode cfg) (pCandles pos')) params' adam'
    writeFile (cfgStateF cfg) (show pos')
    hPutStrLn stderr $ "[save] " ++ cfgCkpt cfg ++ " @ candle " ++ show (pCandles pos')
  pure lv { lvParams = params', lvAdam = adam', lvRing = ring'
          , lvPrevClose = close, lvPos = pos' }

-- ── per-mode runner ───────────────────────────────────────────────────────────

run :: forall v n dM dF dK.
       ( ParamsCSeq v n dM dF dK
       , Serialize (ParamsSeq v n dM dF dK)
       , Adam (ParamsSeq v n dM dF dK)
       , Scale (ParamsSeq v n dM dF dK) )
    => Int -> Int -> Opts -> IO ()
run vI nI o = do
  let ckptPath = fromMaybe ("checkpoint-" ++ optMode o ++ ".ckpt") (optCkpt o)
      specPath = ckptPath ++ ".tok"
      stateF   = fromMaybe (ckptPath ++ ".pos") (optState o)
      nParam   = length (toFloats (zeroA :: ParamsSeq v n dM dF dK))
  haveSpec <- doesFileExist specPath
  unless haveSpec $ do
    hPutStrLn stderr ("market-live: no tokenizer artifact at " ++ specPath
      ++ " — train with market-train first (the .tok defines the language)")
    exitFailure
  spec <- loadSpec specPath
  when (vocabSize spec /= vI) $ do
    hPutStrLn stderr $ "market-live: tokenizer vocab " ++ show (vocabSize spec)
      ++ " /= mode vocab " ++ show vI
    exitFailure
  r <- loadCkpt nParam ckptPath
  (meta, params :: ParamsSeq v n dM dF dK, adam) <- case r of
    Just x  -> pure x
    Nothing -> do
      hPutStrLn stderr ("market-live: no checkpoint at " ++ ckptPath
        ++ " — warm-start from market-train is required")
      exitFailure
  when (ckMode meta /= optMode o) $
    hPutStrLn stderr ("WARNING: checkpoint mode " ++ ckMode meta
                      ++ " /= requested " ++ optMode o)
  pos <- do
    have <- doesFileExist stateF
    if have then fromMaybe pos0 . readMaybe <$> readFile stateF else pure pos0
  -- bootstrap context (ring + prevClose) from the backfill CSV's tail
  (ring0, prevClose0) <- case optData o of
    Nothing -> pure ([], 0)
    Just f  -> do
      have <- doesFileExist f
      if not have then pure ([], 0) else do
        closes <- map snd <$> csvCloses f
        let rets = logReturns closes
        pure ( takeLast (nI - 1) (map (encodeReturn spec) rets)
             , if null closes then 0 else last closes )
  let cfg = Cfg { cfgN = nI, cfgSpec = spec, cfgLr = optLr o
                , cfgFee = optFeeBps o * 1e-4, cfgTh = optThreshBps o * 1e-4
                , cfgSaveEv = optSaveEvery o, cfgCkpt = ckptPath
                , cfgStateF = stateF, cfgMode = optMode o
                , cfgIntMs = fromIntegral (optInterval o) * 1000 }
      lvInit pc = Live { lvParams = params, lvAdam = adam, lvRing = ring0
                       , lvPrevClose = pc, lvBucket = -1, lvLastPx = pc
                       , lvPos = pos }
  hPutStrLn stderr $ printf
    "market-live %s: warm start %s (epoch %d), context %d, %d bins, θ=%.1fbp fee=%.1fbp lr=%g"
    (optMode o) ckptPath (ckEpoch meta) nI (tsNBins spec)
    (optThreshBps o) (optFeeBps o) (optLr o)
  case optReplay o of
    Just f  -> replay cfg (lvInit prevClose0) f
    Nothing -> live cfg (lvInit prevClose0) (T.pack (optCoin o)) prevClose0

-- replay: each CSV candle becomes one trade at its close (same pipeline)
replay :: forall v n dM dF dK.
          ( ParamsCSeq v n dM dF dK
          , Serialize (ParamsSeq v n dM dF dK)
          , Adam (ParamsSeq v n dM dF dK)
          , Scale (ParamsSeq v n dM dF dK) )
       => Cfg -> Live v n dM dF dK -> FilePath -> IO ()
replay cfg lv0 f = do
  rows <- csvCloses f
  when (null rows) (hPutStrLn stderr "replay: empty CSV" >> exitFailure)
  let ((t0, c0) : rest) = sortOn fst rows
      start = lv0 { lvPrevClose = if lvPrevClose lv0 > 0 then lvPrevClose lv0 else c0
                  , lvBucket = t0 `div` cfgIntMs cfg, lvLastPx = c0 }
  lvF <- foldM' start rest
  let p = lvPos lvF
  hPutStrLn stderr $ printf
    "[replay done] %d candles | realized paper PnL %+.5f (log-return units, net %.1fbp/side) | final %s"
    (pCandles p) (pPnl p) (cfgFee cfg * 1e4) (show (pSide p))
  where
    foldM' lv []              = pure lv
    foldM' lv ((t, c) : rest) = do
      lv' <- onTrade cfg True (t + cfgIntMs cfg - 1) c lv
      foldM' lv' rest

-- live: WS thread feeds a queue; the worker folds every trade, recommends on
-- the newest pending one (one-slot semantics via drain)
live :: forall v n dM dF dK.
        ( ParamsCSeq v n dM dF dK
        , Serialize (ParamsSeq v n dM dF dK)
        , Adam (ParamsSeq v n dM dF dK)
        , Scale (ParamsSeq v n dM dF dK) )
     => Cfg -> Live v n dM dF dK -> T.Text -> Double -> IO ()
live cfg lv0 coin prevClose0 = do
  q <- newTQueueIO :: IO (TQueue (Integer, Integer, Double))   -- (tid, ts, px)
  _ <- forkIO $ streamMarketData [coin] $ \msg -> case msg of
        WsTrades ts -> forM_ ts $ \t ->
          forM_ (readMaybe (T.unpack (tradePx t)) :: Maybe Double) $ \px ->
            atomically (writeTQueue q (tradeTid t, tradeTime t, px))
        _ -> pure ()
  when (prevClose0 <= 0) $
    hPutStrLn stderr "NOTE: no --data bootstrap; the first candle only sets the price reference"
  let loop !maxTid !lv = do
        t1 <- atomically (readTQueue q)
        ts <- atomically (flushTQueue q)
        let fresh = [ tr | tr@(tid, _, _) <- t1 : ts, tid > maxTid ]
        if null fresh then loop maxTid lv else do
          let maxTid' = maximum [ tid | (tid, _, _) <- fresh ]
          -- bootstrap the price reference from the very first trade if needed
          lvA <- if lvPrevClose lv <= 0
                   then let (_, ts0, px0) = head fresh
                        in pure lv { lvPrevClose = px0, lvLastPx = px0
                                   , lvBucket = ts0 `div` cfgIntMs cfg }
                   else pure lv
          -- fold all but the newest silently (aggregation + any candle closes)
          lvB <- foldSilent lvA (init fresh)
          -- recommend on the newest
          let (_, tsN, pxN) = last fresh
          lvC <- onTrade cfg True tsN pxN lvB
          loop maxTid' lvC
      foldSilent lv []                 = pure lv
      foldSilent lv ((_, ts, px) : r)  = onTrade cfg False ts px lv >>= \lv' -> foldSilent lv' r
  loop (-1) lv0

-- ── dispatch ──────────────────────────────────────────────────────────────────

main :: IO ()
main = do
  o <- O.execParser (O.info (O.helper <*> optsP)
        (O.fullDesc <> O.progDesc "live online learning + per-trade recommendations (paper)"))
  case optMode o of
    "mkt-small" -> run @130 @32 @32 @128 @8  130 32 o
    "mkt-base"  -> run @514 @64 @64 @256 @16 514 64 o
    m -> hPutStrLn stderr ("unknown mode " ++ m) >> exitFailure
