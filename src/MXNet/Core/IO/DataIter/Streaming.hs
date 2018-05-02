{-# Language TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
module MXNet.Core.IO.DataIter.Streaming where

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

type StreamData m a = Stream (Of (NDArray a, NDArray a)) m ()

imageRecordIter :: (MatchKVList kvs I.ImageRecordIter_Args, ShowKV kvs, DType a, MonadIO m) => 
                   HMap kvs -> StreamData m a
imageRecordIter args = do
    cnt  <- liftIO (newIORef 0)
    iter <- liftIO (I.imageRecordIter args)
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

type instance DatasetConstraint (StreamData (TrainM t m1) t) m2 = m1 ~ m2

instance Monad m => Dataset (StreamData (TrainM t m) t) where
    type DatType (StreamData (TrainM t m) t) = t
    size = length_
    forEach  dat proc = do
        let index = S.enumFrom (1 :: Int)
        toList_ $ void $ S.mapM (\(i,(x,y)) -> proc i x y) (S.zip index dat)    
    forEach' dat proc = do
        t <- size dat
        let index = S.enumFrom (1 :: Int)
        toList_ $ void $ S.mapM (\(i,(x,y)) -> proc (i,t) x y) (S.zip index dat)
