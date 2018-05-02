module Main where

import Test.Hspec
import MXNet.Core.Base

import MXNet.Core.IO.DataIter.Conduit

main :: IO ()
main = hspec $ do
  describe "ImageRecordIter" $ do
    it "batch-size = 32" $ do
      let sr = imageRecordIter (add @"path_imgrec" "test/data/cifar10_val.rec" $ 
                                add @"data_shape" [3,28,28] $
                                add @"batch_size" 32 nil) :: ConduitData Float
      size sr `shouldReturn` 313
    it "batch-size = 128" $ do
      let sr = imageRecordIter (add @"path_imgrec" "test/data/cifar10_val.rec" $ 
                                add @"data_shape" [3,28,28] $
                                add @"batch_size" 128 nil) :: ConduitData Float
      size sr `shouldReturn` 79
