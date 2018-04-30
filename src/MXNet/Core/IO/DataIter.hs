module MXNet.Core.IO.DataIter where

import Data.IORef
import Streaming
import Streaming.Prelude (Of(..), yield)
import MXNet.Core.Base
import MXNet.Core.Base.NDArray (NDArray(..))
import MXNet.Core.Base.Internal
import qualified MXNet.Core.IO.Internal as I

type StreamData a = Stream (Of (NDArray a, NDArray a)) IO Int

imageRecordIter :: (MatchKVList kvs I.ImageRecordIter_Args, ShowKV kvs, DType a) => 
                   HMap kvs -> StreamData a
imageRecordIter args = do
    cnt  <- liftIO (newIORef 0)
    iter <- liftIO (I.imageRecordIter args)
    let loop = do valid <- liftIO $ do 
                      modifyIORef cnt (+1)
                      checked $ mxDataIterNext iter
                  if valid == 0
                  then liftIO (readIORef cnt)
                  else do
                      item <- liftIO $ do 
                          dat <- checked $ mxDataIterGetData  iter
                          lbl <- checked $ mxDataIterGetLabel iter
                          return (NDArray dat, NDArray lbl)
                      yield item
                      loop
    loop
