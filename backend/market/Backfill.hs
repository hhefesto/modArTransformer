-- VENDORED from ~/src/chain-query/app/Backfill.hs (commit 38a2f7c): local
-- historical/top-up OHLCV pull (executable market-backfill).
--
-- | chain-backfill — historical/top-up OHLCV ingestion for the training
-- pipeline (modArTransformer README §10 Phase 3).
--
-- Pulls 1-minute (or coarser) candles from Hyperliquid's REST
-- @candleSnapshot@ into one CSV per market, paginated and idempotent:
-- re-running appends only candles newer than the file's last row, so a cron
-- of this binary IS the durable live-ingestion path for candles.
--
-- Configuration (env, matching the streamer's style):
--   HL_COINS     comma-separated markets or ALL     (default BTC,ETH,SOL,HYPE)
--   HL_INTERVAL  1m | 5m | 15m | 1h | 4h | 1d       (default 1m)
--   HL_DAYS      how many days back to start        (default 30)
--   HL_OUTDIR    output directory for the CSVs      (default data)
{-# LANGUAGE OverloadedStrings #-}
module Main where

import           Control.Monad         (forM_)
import           Data.Maybe            (fromMaybe)
import qualified Data.Text             as T
import           Data.Time.Clock.POSIX (getPOSIXTime)
import           System.Directory      (createDirectoryIfMissing)
import           System.Environment    (lookupEnv)
import           System.Exit           (die)
import           System.IO             (hPutStrLn, stderr)
import           Text.Read             (readMaybe)

import Hyperliquid.Candles (backfillCoin, intervalMs)
import Hyperliquid.Info    (fetchPerpCoins, newManager)

main :: IO ()
main = do
  coinsEnv  <- fromMaybe "BTC,ETH,SOL,HYPE" <$> lookupEnv "HL_COINS"
  interval  <- T.pack . fromMaybe "1m"      <$> lookupEnv "HL_INTERVAL"
  daysEnv   <- fromMaybe "30"               <$> lookupEnv "HL_DAYS"
  outDir    <- fromMaybe "data"             <$> lookupEnv "HL_OUTDIR"
  days <- maybe (die "HL_DAYS must be a number") pure (readMaybe daysEnv :: Maybe Double)
  _ <- maybe (die ("unknown HL_INTERVAL " <> T.unpack interval)) pure (intervalMs interval)
  mgr <- newManager
  coins <- if coinsEnv == "ALL"
    then fetchPerpCoins mgr >>= either (die . ("perp universe: " <>)) pure
    else pure (map T.strip (T.splitOn "," (T.pack coinsEnv)))
  nowS <- getPOSIXTime
  let nowMs   = floor (realToFrac nowS * 1000 :: Double) :: Integer
      startMs = nowMs - floor (days * 86_400_000)
  createDirectoryIfMissing True outDir
  hPutStrLn stderr $ "[backfill] " <> show (length coins) <> " coins, interval "
    <> T.unpack interval <> ", " <> show days <> " days, dir " <> outDir
  forM_ coins $ \c -> backfillCoin mgr outDir c interval startMs nowMs
