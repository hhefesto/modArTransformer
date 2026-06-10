-- VENDORED from ~/src/chain-query/src/Hyperliquid/Candles.hs (commit 38a2f7c) so the
-- market pipeline is fully local to this repo.  Keep edits upstream-first.
--
-- | Historical OHLCV backfill via the REST @/info@ @candleSnapshot@ endpoint.
--
-- @POST /info {"type":"candleSnapshot","req":{coin,interval,startTime,endTime}}@
-- returns at most ~5000 candles and, for a wide range, the LATEST ones in it —
-- so backfill paginates in fixed forward windows of @batch × interval@ and
-- de-duplicates by open time.  Re-running is idempotent: the writer reads the
-- last open time already in the CSV and appends only newer candles, which also
-- makes periodic "top-up" runs the durable live-ingestion path (no stateful
-- stream needed for candles).
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Hyperliquid.Candles
  ( Candle(..)
  , intervalMs
  , fetchCandles
  , backfillCoin
  ) where

import Control.Concurrent     (threadDelay)
import Control.Monad          (when)
import Data.Aeson
import Data.List              (sortOn)
import Data.Maybe             (mapMaybe)
import qualified Data.Text    as T
import Data.Text              (Text)
import Network.HTTP.Client    (Manager)
import System.Directory       (doesFileExist)
import System.IO              (IOMode (AppendMode), hPutStr, withFile, hPutStrLn, stderr)
import Text.Read              (readMaybe)

import Hyperliquid.Info  (postInfo)
import Hyperliquid.Types (Coin)

-- | One OHLCV candle.  Numeric fields are kept as the API's strings
-- (precision-preserving, same policy as 'LlmRecord'); validation parses them.
data Candle = Candle
  { caT :: Integer   -- ^ open time (epoch ms)
  , caO :: Text      -- ^ open
  , caH :: Text      -- ^ high
  , caL :: Text      -- ^ low
  , caC :: Text      -- ^ close
  , caV :: Text      -- ^ base volume
  , caN :: Integer   -- ^ number of trades
  } deriving (Show, Eq)

instance FromJSON Candle where
  parseJSON = withObject "candle" $ \o ->
    Candle <$> o .: "t" <*> o .: "o" <*> o .: "h" <*> o .: "l"
           <*> o .: "c" <*> o .: "v" <*> o .: "n"

-- | Interval string → milliseconds (the ones the API documents).
intervalMs :: Text -> Maybe Integer
intervalMs i = case i of
  "1m" -> Just 60_000 ; "5m"  -> Just 300_000 ; "15m" -> Just 900_000
  "1h" -> Just 3_600_000 ; "4h" -> Just 14_400_000 ; "1d" -> Just 86_400_000
  _    -> Nothing

-- | One snapshot request for a closed window.
fetchCandles :: Manager -> Coin -> Text -> Integer -> Integer -> IO (Either String [Candle])
fetchCandles mgr coin interval startMs endMs =
  postInfo mgr $ object
    [ "type" .= ("candleSnapshot" :: Text)
    , "req"  .= object
        [ "coin" .= coin, "interval" .= interval
        , "startTime" .= startMs, "endTime" .= endMs ]
    ]

-- | A candle is well-formed when its numeric strings parse and OHLC are
-- coherent (l ≤ o,c ≤ h).  Malformed rows are dropped LOUDLY (count reported).
validCandle :: Candle -> Bool
validCandle c =
  case traverse (readMaybe . T.unpack) [caO c, caH c, caL c, caC c, caV c] :: Maybe [Double] of
    Just [o', h, l, cl, v] -> l <= h && l <= o' && o' <= h && l <= cl && cl <= h && v >= 0
    _                      -> False

csvLine :: Candle -> String
csvLine c = concat
  [ show (caT c), ",", T.unpack (caO c), ",", T.unpack (caH c), ",", T.unpack (caL c)
  , ",", T.unpack (caC c), ",", T.unpack (caV c), ",", show (caN c) ]

-- last open-time already present in the CSV (skipping the header).
lastT :: FilePath -> IO (Maybe Integer)
lastT path = do
  ok <- doesFileExist path
  if not ok then pure Nothing else do
    s <- readFile path
    let ts = mapMaybe (readMaybe . takeWhile (/= ',')) (drop 1 (lines s)) :: [Integer]
    length ts `seq` pure (if null ts then Nothing else Just (maximum ts))

-- | Backfill (or top up) one coin into @<dir>/<coin>-<interval>.csv@.
-- Idempotent: starts after the newest candle already on disk.
backfillCoin :: Manager -> FilePath -> Coin -> Text -> Integer -> Integer -> IO ()
backfillCoin mgr dir coin interval startMs endMs = do
  step <- maybe (fail ("unknown interval " <> T.unpack interval)) pure (intervalMs interval)
  let path  = dir <> "/" <> T.unpack coin <> "-" <> T.unpack interval <> ".csv"
      batch = 4000 * step                       -- stay under the ~5000 cap
  prev <- lastT path
  exists <- doesFileExist path
  when (not exists) (writeFile path "t,o,h,l,c,v,n\n")
  let cur0 = maybe startMs (\t -> max startMs (t + step)) prev
  hPutStrLn stderr $ "[backfill] " <> T.unpack coin <> " " <> T.unpack interval
    <> " from " <> show cur0 <> " to " <> show endMs
  go path step batch cur0 (maybe (-1) id prev) (0 :: Int) (0 :: Int)
  where
    go path step batch cur seen total dropped
      | cur > endMs = hPutStrLn stderr $
          "[backfill] " <> T.unpack coin <> " done: " <> show total
          <> " candles (" <> show dropped <> " malformed dropped)"
      | otherwise = do
          let hi = min endMs (cur + batch - 1)
          r <- fetchCandles mgr coin interval cur hi
          case r of
            Left err -> do
              hPutStrLn stderr ("[backfill] " <> T.unpack coin <> " ERROR: " <> err
                                <> " (retrying window in 5s)")
              threadDelay 5_000_000
              go path step batch cur seen total dropped
            Right cs -> do
              let fresh   = sortOn caT [ c | c <- cs, caT c > seen ]
                  (good, bad) = (filter validCandle fresh, filter (not . validCandle) fresh)
              when (not (null good)) $
                withFile path AppendMode $ \h ->
                  hPutStr h (unlines (map csvLine good))
              threadDelay 300_000   -- be polite to the public endpoint
              let seen' = if null fresh then seen else maximum (map caT fresh)
              go path step batch (hi + 1) (max seen seen')
                 (total + length good) (dropped + length bad)
