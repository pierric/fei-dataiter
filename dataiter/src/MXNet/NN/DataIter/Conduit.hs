{-# Language TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
module MXNet.NN.DataIter.Conduit (
    ConduitData(..),
    Dataset(..),
    imageRecordIter, mnistIter, csvIter, libSVMIter
) where

import Data.Conduit
import qualified Data.Conduit.Combinators as C
import qualified Data.Conduit.List as CL
import Control.Monad.IO.Class

import MXNet.Base
import qualified MXNet.NN.DataIter.Raw as I
import MXNet.NN.DataIter.Class

newtype ConduitData m a = ConduitData { getConduit :: ConduitM () a m () }

imageRecordIter :: (Fullfilled "ImageRecordIter" args, DType a, MonadIO m) 
    => ArgsHMap "ImageRecordIter" args -> ConduitData m (NDArray a, NDArray a)
imageRecordIter = makeIter I._ImageRecordIter

mnistIter :: (Fullfilled "MNISTIter" args, DType a, MonadIO m) 
    => ArgsHMap "MNISTIter" args -> ConduitData m (NDArray a, NDArray a)
mnistIter = makeIter I._MNISTIter

csvIter :: (Fullfilled "CSVIter" args, DType a, MonadIO m) 
    => ArgsHMap "CSVIter" args -> ConduitData m (NDArray a, NDArray a)
csvIter = makeIter I._CSVIter

libSVMIter :: (Fullfilled "LibSVMIter" args, DType a, MonadIO m) 
    => ArgsHMap "LibSVMIter" args -> ConduitData m (NDArray a, NDArray a)
libSVMIter = makeIter I._LibSVMIter

makeIter creator args = ConduitData $ do
    iter <- liftIO (creator args)
    let loop = do valid <- liftIO $ mxDataIterNext iter
                  if valid == 0
                  then liftIO (finalizeDataIterHandle iter)
                  else do
                      yieldM $ liftIO $ do 
                          dat <- mxDataIterGetData  iter
                          lbl <- mxDataIterGetLabel iter
                          return (NDArray dat, NDArray lbl)
                      loop
    loop

type instance DatasetConstraint (ConduitData m1) m2 = m1 ~ m2

instance Monad m => Dataset (ConduitData m) where
    fromListD = ConduitData . CL.sourceList 
    zipD (ConduitData d1) (ConduitData d2) = ConduitData $ getZipSource $ (,) <$> ZipSource d1 <*> ZipSource d2
    sizeD (ConduitData dat) = runConduit (dat .| C.length)
    forEachD (ConduitData dat) proc = sourceToList $ dat .| CL.mapM proc
