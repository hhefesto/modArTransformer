-- VENDORED from ~/src/chain-query/src/Hyperliquid/Types.hs (commit 38a2f7c) so the
-- market pipeline is fully local to this repo.  Keep edits upstream-first.
--
-- | Core domain types for the Hyperliquid market-data feed.
--
-- The shapes here mirror the public Hyperliquid API
-- (<https://hyperliquid.gitbook.io/hyperliquid-docs>):
--
--   * 'Trade'     – one fill from the @trades@ WebSocket stream.
--   * 'AssetCtx'  – per-coin context (funding, open interest, mark/oracle
--                   price) from the @activeAssetCtx@ stream.
--   * 'LlmRecord' – the flattened, LLM-ready record we emit as JSONL.
--   * 'WsMessage' – a decoded WebSocket envelope.
{-# LANGUAGE OverloadedStrings #-}
module Hyperliquid.Types
  ( Coin
  , Trade (..)
  , AssetCtx (..)
  , LlmRecord (..)
  , WsMessage (..)
  ) where

import Data.Aeson
import Data.Text (Text)

-- | A Hyperliquid market symbol, e.g. @"BTC"@, @"ETH"@, @"HYPE"@.
type Coin = Text

-- ---------------------------------------------------------------------------
-- Trade (from the @trades@ subscription)
-- ---------------------------------------------------------------------------

-- | A single executed trade.
--
-- @users@ is the counterparty pair; Hyperliquid orders it @[buyer, seller]@.
-- @side@ is the aggressor side: @"B"@ (buy) or @"A"@ (sell).
data Trade = Trade
  { tradeCoin  :: Coin
  , tradeSide  :: Text
  , tradePx    :: Text
  , tradeSz    :: Text
  , tradeTime  :: Integer  -- ^ epoch milliseconds
  , tradeHash  :: Text
  , tradeTid   :: Integer  -- ^ unique, monotonically increasing trade id
  , tradeUsers :: [Text]   -- ^ @[buyer, seller]@
  } deriving (Show, Eq)

instance FromJSON Trade where
  parseJSON = withObject "Trade" $ \o ->
    Trade
      <$> o .: "coin"
      <*> o .: "side"
      <*> o .: "px"
      <*> o .: "sz"
      <*> o .: "time"
      <*> o .: "hash"
      <*> o .: "tid"
      <*> o .: "users"

-- ---------------------------------------------------------------------------
-- Asset context (from the @activeAssetCtx@ subscription)
-- ---------------------------------------------------------------------------

-- | Per-coin market context. All numeric fields arrive as strings and are
-- kept verbatim so no precision is lost before the LLM sees them.
data AssetCtx = AssetCtx
  { ctxFunding      :: Maybe Text
  , ctxOpenInterest :: Maybe Text
  , ctxOraclePx     :: Maybe Text
  , ctxMarkPx       :: Maybe Text
  , ctxMidPx        :: Maybe Text
  } deriving (Show, Eq)

instance FromJSON AssetCtx where
  parseJSON = withObject "AssetCtx" $ \o ->
    AssetCtx
      <$> o .:? "funding"
      <*> o .:? "openInterest"
      <*> o .:? "oraclePx"
      <*> o .:? "markPx"
      <*> o .:? "midPx"

-- ---------------------------------------------------------------------------
-- LLM-ready record (our JSONL output)
-- ---------------------------------------------------------------------------

-- | One trade, flattened and joined with the latest 'AssetCtx' for its coin.
-- This is the record an LLM market-prediction pipeline consumes.
--
--   * @who@   → 'lrBuyer' \/ 'lrSeller'
--   * @what@  → 'lrCoin'
--   * @how much@ → 'lrSz' (+ 'lrNotional')
data LlmRecord = LlmRecord
  { lrTs           :: Integer
  , lrCoin         :: Coin
  , lrSide         :: Text       -- ^ "buy" or "sell" (aggressor)
  , lrPx           :: Text
  , lrSz           :: Text
  , lrNotional     :: Double
  , lrBuyer        :: Maybe Text
  , lrSeller       :: Maybe Text
  , lrTid          :: Integer
  , lrHash         :: Text
  , lrMarkPx       :: Maybe Text
  , lrOraclePx     :: Maybe Text
  , lrFunding      :: Maybe Text
  , lrOpenInterest :: Maybe Text
  } deriving (Show, Eq)

instance ToJSON LlmRecord where
  toJSON r = object
    [ "ts"           .= lrTs r
    , "coin"         .= lrCoin r
    , "side"         .= lrSide r
    , "px"           .= lrPx r
    , "sz"           .= lrSz r
    , "notional"     .= lrNotional r
    , "buyer"        .= lrBuyer r
    , "seller"       .= lrSeller r
    , "tid"          .= lrTid r
    , "hash"         .= lrHash r
    , "markPx"       .= lrMarkPx r
    , "oraclePx"     .= lrOraclePx r
    , "funding"      .= lrFunding r
    , "openInterest" .= lrOpenInterest r
    ]

-- ---------------------------------------------------------------------------
-- WebSocket envelope
-- ---------------------------------------------------------------------------

-- | A decoded message from the Hyperliquid WebSocket. Channels we don't act
-- on (subscription acks, errors) collapse into 'WsOther'.
data WsMessage
  = WsTrades [Trade]      -- ^ @channel: "trades"@
  | WsCtx Coin AssetCtx   -- ^ @channel: "activeAssetCtx"@
  | WsPong                -- ^ @channel: "pong"@
  | WsOther Text          -- ^ any other channel, carrying its name
  deriving (Show)

instance FromJSON WsMessage where
  parseJSON = withObject "WsMessage" $ \o -> do
    channel <- o .: "channel"
    case (channel :: Text) of
      "trades"         -> WsTrades <$> o .: "data"
      "activeAssetCtx" -> do
        d <- o .: "data"
        WsCtx <$> d .: "coin" <*> d .: "ctx"
      "pong"           -> pure WsPong
      other            -> pure (WsOther other)
