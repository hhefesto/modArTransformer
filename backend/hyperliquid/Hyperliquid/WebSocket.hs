-- VENDORED from ~/src/chain-query/src/Hyperliquid/WebSocket.hs (commit 38a2f7c) so the
-- market pipeline is fully local to this repo.  Keep edits upstream-first.
--
-- | Hyperliquid WebSocket client.
--
-- Connects to @wss://api.hyperliquid.xyz/ws@, subscribes to the @trades@ and
-- @activeAssetCtx@ streams for a set of coins, keeps the connection alive with
-- periodic pings, and hands every decoded 'WsMessage' to a callback.
--
-- The connection self-heals: on any disconnect it waits briefly and
-- reconnects, resubscribing to every coin. Snapshot replays after a reconnect
-- are de-duplicated downstream by trade id.
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Hyperliquid.WebSocket
  ( streamMarketData
  ) where

import Control.Concurrent      (forkIO, killThread, threadDelay)
import Control.Exception       (SomeException, finally, try)
import Control.Monad           (forM_, forever)
import Data.Aeson              (Value, eitherDecode, encode, object, (.=))
import Data.Text               (Text)
import qualified Data.ByteString.Lazy as BL
import qualified Network.WebSockets   as WS
import System.IO               (hPutStrLn, stderr)
import Wuss                    (runSecureClient)

import Hyperliquid.Types (Coin, WsMessage)

wsHost :: String
wsHost = "api.hyperliquid.xyz"

wsPath :: String
wsPath = "/ws"

-- | Seconds between keepalive pings. The server drops idle connections after
-- 60s, so we stay well under that.
pingIntervalSec :: Int
pingIntervalSec = 20

-- | Seconds to wait before reconnecting after a dropped connection.
reconnectDelaySec :: Int
reconnectDelaySec = 3

-- | Stream trades and asset context for @coins@ forever, invoking @handler@
-- on each decoded message. Reconnects automatically on failure.
streamMarketData :: [Coin] -> (WsMessage -> IO ()) -> IO ()
streamMarketData coins handler = forever $ do
  result <- try (runSecureClient wsHost 443 wsPath (app coins handler))
  case result of
    Left (e :: SomeException) ->
      hPutStrLn stderr $
        "[WARN] WebSocket disconnected (" <> show e <> "); reconnecting in "
          <> show reconnectDelaySec <> "s"
    Right () -> pure ()
  threadDelay (reconnectDelaySec * 1_000_000)

-- | One connection's lifecycle: subscribe, ping, receive until it drops.
app :: [Coin] -> (WsMessage -> IO ()) -> WS.ClientApp ()
app coins handler conn = do
  forM_ coins $ \c -> do
    WS.sendTextData conn (encode (subscribe "trades" c))
    WS.sendTextData conn (encode (subscribe "activeAssetCtx" c))

  pinger <- forkIO $ forever $ do
    threadDelay (pingIntervalSec * 1_000_000)
    WS.sendTextData conn (encode pingMessage)

  flip finally (killThread pinger) $ forever $ do
    raw <- WS.receiveData conn :: IO BL.ByteString
    case eitherDecode raw of
      Right msg -> handler msg
      Left _    -> pure ()  -- ignore acks / unrecognised frames

-- | A @subscribe@ request for one channel + coin.
subscribe :: Text -> Coin -> Value
subscribe channel coin = object
  [ "method"       .= ("subscribe" :: Text)
  , "subscription" .= object [ "type" .= channel, "coin" .= coin ]
  ]

pingMessage :: Value
pingMessage = object [ "method" .= ("ping" :: Text) ]
