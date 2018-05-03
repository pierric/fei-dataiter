{-# Language TypeFamilies #-}
{-# Language FlexibleInstances #-}
module MXNet.Core.IO.DataIter.Streaming (
    StreamData,
    Dataset(..),
    imageRecordIter, mnistIter, csvIter, libSVMIter
) where

import Data.IORef
import Streaming
import Streaming.Prelude (Of(..), yield, length_, toList_)
import qualified Streaming.Prelude as S
import MXNet.Core.Base
import MXNet.Core.Base.NDArray (NDArray(..))
import MXNet.Core.Base.Internal
import qualified MXNet.Core.IO.Internal as I

import MXNet.NN.Types (TrainM)
import MXNet.NN.DataIter.Class

newtype StreamData m a = StreamData { getStream :: Stream (Of a) m ()}

imageRecordIter :: (MatchKVList kvs I.ImageRecordIter_Args, ShowKV kvs, DType a, MonadIO m) => 
                   HMap kvs -> StreamData m (NDArray a, NDArray a)
imageRecordIter = makeIter I.imageRecordIter

mnistIter :: (MatchKVList kvs I.MNISTIter_Args, ShowKV kvs, DType a, MonadIO m) => 
             HMap kvs -> StreamData m (NDArray a, NDArray a)
mnistIter = makeIter I.mNISTIter

csvIter :: (MatchKVList kvs I.CSVIter_Args, ShowKV kvs, DType a, MonadIO m) => 
             HMap kvs -> StreamData m (NDArray a, NDArray a)
csvIter = makeIter I.cSVIter

libSVMIter :: (MatchKVList kvs I.LibSVMIter_Args, ShowKV kvs, DType a, MonadIO m) => 
              HMap kvs -> StreamData m (NDArray a, NDArray a)
libSVMIter = makeIter I.libSVMIter

makeIter creator args = StreamData $ do
    cnt  <- liftIO (newIORef 0)
    iter <- liftIO (creator args)
    let loop = do valid <- liftIO $ do 
                      modifyIORef cnt (+1)
                      checked $ mxDataIterNext iter
                  if valid == 0
                  then liftIO (checked $ mxDataIterFree iter)
                  else do
                      item <- liftIO $ do 
                          dat <- checked $ mxDataIterGetData  iter
                          lbl <- checked $ mxDataIterGetLabel iter
                          return (NDArray dat, NDArray lbl)
                      yield item
                      loop
    loop

type instance DatasetConstraint (StreamData m1) m2 = m1 ~ m2

instance Monad m => Dataset (StreamData m) where
    fromListD = StreamData . S.each
    zipD s1 s2 = StreamData $ S.zip (getStream s1) (getStream s2)
    sizeD = length_ . getStream
    forEachD dat proc = toList_ $ void $ S.mapM proc (getStream dat)
