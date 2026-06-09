{-# LANGUAGE ScopedTypeVariables #-}

-- Shared, robust checkpoint I/O for every trainer (modular, Dyck, synth, market).
--
--  * Atomic save: write `<path>.tmp` then rename into place, so a save interrupted
--    mid-write never corrupts the resumable checkpoint.
--  * Validated load: the body must be exactly 3*nParam floats (params + Adam m + v);
--    otherwise fail loudly instead of crashing deep in `fromFloats`.
--  * BINARY format (new saves): a one-line text header
--    `CKPTB <task> <mode> <epoch> <t> <b1pow> <b2pow> <nParam>` followed by
--    3*nParam raw little-endian Float64s — ~4× smaller and ~50× faster to parse
--    than the text format at production parameter counts.
--  * The legacy TEXT formats are still read: `CKPT ...` (one float per line) and
--    the original modular `CHECKPOINT_ADAMW <epoch> <t> <b1pow> <b2pow>`.
module Checkpoint
  ( CkptMeta(..)
  , saveCkpt
  , loadCkpt
  ) where

import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BC
import qualified Data.ByteString.Builder as BB
import qualified Data.ByteString.Lazy as BL
import Data.Binary.Get (runGetOrFail, getDoublele)
import Control.Monad (replicateM)
import System.Directory (doesFileExist, renameFile)
import System.IO (hPutStrLn, stderr)
import System.Exit (exitFailure)
import Text.Read (readMaybe)

import Optimizer (AdamState(..))
import Serialize (Serialize, toFloats, fromFloats)

data CkptMeta = CkptMeta
  { ckTask  :: String      -- "modarith" | "dyck" | "synth" | "market"
  , ckMode  :: String      -- the -m mode string (e.g. "p97l2", "mkt-small")
  , ckEpoch :: Int         -- training epoch this checkpoint was taken at
  } deriving (Show)

-- | Atomic, tagged BINARY save of params + AdamW state.
saveCkpt :: Serialize p => FilePath -> CkptMeta -> p -> AdamState p -> IO ()
saveCkpt path meta ps st = do
  let tmp    = path ++ ".tmp"
      nParam = length (toFloats ps)
      header = unwords
        [ "CKPTB", sanitize (ckTask meta), sanitize (ckMode meta), show (ckEpoch meta)
        , show (asT st), show (asB1Pow st), show (asB2Pow st), show nParam ]
      body   = toFloats ps ++ toFloats (asM st) ++ toFloats (asV st)
  BL.writeFile tmp $ BB.toLazyByteString $
    BB.stringUtf8 (header ++ "\n") <> foldMap BB.doubleLE body
  renameFile tmp path
  where sanitize = map (\c -> if c == ' ' then '_' else c)

-- | Load + validate (binary or legacy text).  Returns the metadata, params, and
-- AdamW state, or Nothing if the file does not exist.  A malformed/incomplete
-- file aborts with a clear message.
loadCkpt :: forall p. Serialize p => Int -> FilePath -> IO (Maybe (CkptMeta, p, AdamState p))
loadCkpt nParam path = do
  exists <- doesFileExist path
  if not exists then pure Nothing else do
    raw <- BS.readFile path
    let (headerB, restB) = BC.break (== '\n') raw
    case words (BC.unpack headerB) of
      ("CKPTB" : task : mode : eS : tsS : b1S : b2S : nS) ->
        case (,,,,) <$> readMaybe eS <*> readMaybe tsS <*> readMaybe b1S
                    <*> readMaybe b2S <*> declared nS of
          Nothing -> bad "unparseable CKPTB header"
          Just (e, ts, b1, b2, nDecl)
            | nDecl /= nParam -> bad $ "checkpoint is for a model with "
                ++ show nDecl ++ " params, expected " ++ show nParam
            | otherwise ->
                let bytes = BS.drop 1 restB
                    want  = 3 * nParam
                in if BS.length bytes /= 8 * want
                     then bad $ "binary body is " ++ show (BS.length bytes)
                          ++ " bytes, expected " ++ show (8 * want)
                          ++ " (interrupted or wrong-model save)"
                     else case runGetOrFail (replicateM want getDoublele)
                                            (BL.fromStrict bytes) of
                       Left  (_, _, msg) -> bad ("binary decode: " ++ msg)
                       Right (_, _, nums) -> pure (Just (assemble task mode e ts b1 b2 nums))
      _ -> loadText nParam path raw   -- legacy text formats
  where
    declared [nS] = readMaybe nS
    declared _    = Nothing
    bad = badWith path

-- legacy text loader: `CKPT ...` / `CHECKPOINT_ADAMW ...` headers, one float per line.
loadText :: forall p. Serialize p
         => Int -> FilePath -> BS.ByteString -> IO (Maybe (CkptMeta, p, AdamState p))
loadText nParam path raw =
  case lines (BC.unpack raw) of
    [] -> bad "empty checkpoint file"
    (header : rest) -> case parseHeader (words header) of
      Nothing -> bad "unrecognized checkpoint header"
      Just (task, mode, e, ts, b1, b2) ->
        case traverse readMaybe rest :: Maybe [Double] of
          Nothing -> bad "non-numeric body"
          Just nums
            | length nums == 3 * nParam ->
                pure (Just (assemble task mode e ts b1 b2 nums))
            | otherwise -> bad $ "wrong float count: " ++ show (length nums)
                ++ ", expected " ++ show (3 * nParam)
                ++ " (interrupted or wrong-model save)"
  where
    parseHeader ("CKPT" : task : mode : eS : tsS : b1S : b2S : _) =
      (\e ts b1 b2 -> (task, mode, e, ts, b1, b2))
        <$> readMaybe eS <*> readMaybe tsS <*> readMaybe b1S <*> readMaybe b2S
    parseHeader ["CHECKPOINT_ADAMW", eS, tsS, b1S, b2S] =
      (\e ts b1 b2 -> ("modarith", "?", e, ts, b1, b2))
        <$> readMaybe eS <*> readMaybe tsS <*> readMaybe b1S <*> readMaybe b2S
    parseHeader _ = Nothing
    bad = badWith path

assemble :: Serialize p
         => String -> String -> Int -> Int -> Double -> Double -> [Double]
         -> (CkptMeta, p, AdamState p)
assemble task mode e ts b1 b2 nums =
  let (ps, r1) = fromFloats nums
      (mm, r2) = fromFloats r1
      (vv, _)  = fromFloats r2
  in (CkptMeta task mode e, ps, AdamState ts b1 b2 mm vv)

badWith :: FilePath -> String -> IO a
badWith path msg = do
  hPutStrLn stderr $ "loadCkpt: " ++ path ++ " is invalid: " ++ msg
    ++ ". Delete it to start fresh, or pass --checkpoint PATH for another file."
  exitFailure
