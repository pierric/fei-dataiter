{-# Language TypeFamilies #-}
{-# Language FlexibleInstances #-}
module MXNet.NN.DataIter.Streaming (
    StreamData(..),
    Dataset(..),
    imageRecordIter, mnistIter, csvIter, libSVMIter
) where

import Streaming
import Streaming.Prelude (Of(..), yield, length_, toList_)
import qualified Streaming.Prelude as S

import MXNet.Base
import qualified MXNet.NN.DataIter.Raw as I
import MXNet.NN.DataIter.Class

newtype StreamData m a = StreamData { getStream :: Stream (Of a) m ()}

imageRecordIter :: (Fullfilled "ImageRecordIter" args, DType a, MonadIO m) 
    => ArgsHMap "ImageRecordIter" args -> StreamData m (NDArray a, NDArray a)
imageRecordIter = makeIter I._ImageRecordIter

mnistIter :: (Fullfilled "MNISTIter" args, DType a, MonadIO m) 
    => ArgsHMap "MNISTIter" args -> StreamData m (NDArray a, NDArray a)
mnistIter = makeIter I._MNISTIter

csvIter :: (Fullfilled "CSVIter" args, DType a, MonadIO m) 
    => ArgsHMap "CSVIter" args -> StreamData m (NDArray a, NDArray a)
csvIter = makeIter I._CSVIter

libSVMIter :: (Fullfilled "LibSVMIter" args, DType a, MonadIO m) 
    => ArgsHMap "LibSVMIter" args -> StreamData m (NDArray a, NDArray a)
libSVMIter = makeIter I._LibSVMIter

makeIter creator args = StreamData $ do
    iter <- liftIO (creator args)
    let loop = do valid <- liftIO $ mxDataIterNext iter
                  if valid == 0
                  then liftIO (finalizeDataIterHandle iter)
                  else do
                      item <- liftIO $ do 
                          dat <- mxDataIterGetData  iter
                          lbl <- mxDataIterGetLabel iter
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
    foldD dat elem proc = S.foldM_ proc (return elem) return (getStream dat)