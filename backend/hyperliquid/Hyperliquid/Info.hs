-- VENDORED from ~/src/chain-query/src/Hyperliquid/Info.hs (commit 38a2f7c) so the
-- market pipeline is fully local to this repo.  Keep edits upstream-first.
--
-- | Thin client for the Hyperliquid REST @/info@ endpoint.
--
-- Public market data needs no authentication. We only use this to discover
-- the perp universe (the list of tradeable coins) so the streamer can be
-- pointed at "all coins" instead of a fixed shortlist.
--
-- Endpoint: @POST https://api.hyperliquid.xyz/info@
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Hyperliquid.Info
  ( newManager
  , fetchPerpCoins
  , postInfo
  ) where

import Control.Exception      (SomeException, try)
import Data.Aeson
import Network.HTTP.Client
  ( Manager, RequestBody (..), httpLbs, parseRequest, requestBody
  , requestHeaders, responseBody, method
  )
import Network.HTTP.Client.TLS (newTlsManager)

import Hyperliquid.Types (Coin)

-- | A TLS-capable HTTP manager, reusable across requests.
newManager :: IO Manager
newManager = newTlsManager

-- | The perp universe returned by @{"type":"meta"}@: just the coin names.
newtype Meta = Meta [Coin]

instance FromJSON Meta where
  parseJSON = withObject "meta" $ \o -> do
    universe <- o .: "universe"
    Meta <$> mapM (withObject "asset" (.: "name")) universe

-- | Fetch the full list of perpetual coins from @/info@.
fetchPerpCoins :: Manager -> IO (Either String [Coin])
fetchPerpCoins mgr =
  fmap (fmap (\(Meta cs) -> cs)) (postInfo mgr (object ["type" .= ("meta" :: String)]))

-- | POST a JSON body to @/info@ and decode the response.
postInfo :: FromJSON a => Manager -> Value -> IO (Either String a)
postInfo mgr body = do
  result <- try $ do
    req0 <- parseRequest "https://api.hyperliquid.xyz/info"
    let req = req0
          { method         = "POST"
          , requestBody    = RequestBodyLBS (encode body)
          , requestHeaders = [("Content-Type", "application/json")]
          }
    responseBody <$> httpLbs req mgr
  pure $ case result of
    Left (e :: SomeException) -> Left (show e)
    Right lbs                 -> eitherDecode lbs
