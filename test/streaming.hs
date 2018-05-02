module Main where

import Test.Hspec
import Streaming.Prelude
import MXNet.Core.Base

import MXNet.Core.IO.DataIter.Streaming

main :: IO ()
main = hspec $ do
  describe "ImageRecordIter" $ do
    it "batch-size = 32" $ do
      let sr = imageRecordIter (add @"path_imgrec" "test/data/cifar10_val.rec" $ 
                                add @"data_shape" [3,28,28] $
                                add @"batch_size" 32 nil) :: StreamData Float
      effects sr `shouldReturn` 314
    it "batch-size = 128" $ do
      let sr = imageRecordIter (add @"path_imgrec" "test/data/cifar10_val.rec" $ 
                                add @"data_shape" [3,28,28] $
                                add @"batch_size" 128 nil) :: StreamData Float
      effects sr `shouldReturn` 80
