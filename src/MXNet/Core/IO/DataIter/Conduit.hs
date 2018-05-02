{-# Language TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
module MXNet.Core.IO.DataIter.Conduit where

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

type ConduitData m a = ConduitM () (NDArray a, NDArray a) m ()

imageRecordIter :: (MatchKVList kvs I.ImageRecordIter_Args, ShowKV kvs, DType a, MonadIO m) => 
                   HMap kvs -> ConduitData m a
imageRecordIter args = do
    iter <- liftIO (I.imageRecordIter args)
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

type instance DatasetConstraint (ConduitData (TrainM t m1) t) m2 = m1 ~ m2

instance Monad m => Dataset (ConduitData (TrainM t m) t) where
    type DatType (ConduitData (TrainM t m) t) = t
    size dat = runConduit (dat .| C.length)
    forEach  dat proc = do
        let index = CL.sourceList [1..]
        sourceToList $ (getZipSource $ (,) <$> ZipSource index <*> ZipSource dat) .| CL.mapM (\(i,(x,y)) -> proc i x y)
    forEach' dat proc = do
        t <- size dat
        let index = CL.sourceList [1..]
        sourceToList $ (getZipSource $ (,) <$> ZipSource index <*> ZipSource dat) .| CL.mapM (\(i,(x,y)) -> proc (i,t) x y)