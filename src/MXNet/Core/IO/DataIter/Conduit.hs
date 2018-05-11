{-# Language TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
module MXNet.Core.IO.DataIter.Conduit (
    ConduitData(..),
    Dataset(..),
    imageRecordIter, mnistIter, csvIter, libSVMIter
) where

import Data.IORef
import Data.Conduit
import qualified Data.Conduit.Combinators as C
import qualified Data.Conduit.List as CL
import Control.Monad.IO.Class
import MXNet.Core.Base
import MXNet.Core.Base.NDArray (NDArray(..))
import MXNet.Core.Base.Internal
import qualified MXNet.Core.IO.Internal as I

import MXNet.NN.Types (TrainM)
import MXNet.NN.DataIter.Class

newtype ConduitData m a = ConduitData { getConduit :: ConduitM () a m () }

imageRecordIter :: (MatchKVList kvs I.ImageRecordIter_Args, ShowKV kvs, DType a, MonadIO m) => 
                   HMap kvs -> ConduitData m (NDArray a, NDArray a)
imageRecordIter = makeIter I.imageRecordIter

mnistIter :: (MatchKVList kvs I.MNISTIter_Args, ShowKV kvs, DType a, MonadIO m) => 
             HMap kvs -> ConduitData m (NDArray a, NDArray a)
mnistIter = makeIter I.mNISTIter

csvIter :: (MatchKVList kvs I.CSVIter_Args, ShowKV kvs, DType a, MonadIO m) => 
             HMap kvs -> ConduitData m (NDArray a, NDArray a)
csvIter = makeIter I.cSVIter

libSVMIter :: (MatchKVList kvs I.LibSVMIter_Args, ShowKV kvs, DType a, MonadIO m) => 
              HMap kvs -> ConduitData m (NDArray a, NDArray a)
libSVMIter = makeIter I.libSVMIter

makeIter creator args = ConduitData $ do
    iter <- liftIO (creator args)
    let loop = do valid <- liftIO $ checked $ mxDataIterNext iter
                  if valid == 0
                  then liftIO (checked $ mxDataIterFree iter)
                  else do
                      yieldM $ liftIO $ do 
                          dat <- checked $ mxDataIterGetData  iter
                          lbl <- checked $ mxDataIterGetLabel iter
                          return (NDArray dat, NDArray lbl)
                      loop
    loop

type instance DatasetConstraint (ConduitData m1) m2 = m1 ~ m2

instance Monad m => Dataset (ConduitData m) where
    fromListD = ConduitData . CL.sourceList 
    zipD (ConduitData d1) (ConduitData d2) = ConduitData $ getZipSource $ (,) <$> ZipSource d1 <*> ZipSource d2
    sizeD (ConduitData dat) = runConduit (dat .| C.length)
    forEachD (ConduitData dat) proc = sourceToList $ dat .| CL.mapM proc
