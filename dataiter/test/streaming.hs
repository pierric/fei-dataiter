module Main where

import Test.Hspec
import Streaming.Prelude
import MXNet.Core.Base

import MXNet.Core.IO.DataIter.Streaming

type DS = StreamData IO (NDArray Float, NDArray Float)

main :: IO ()
main = hspec $ do
  describe "MNISTIter" $ do
    it "batch-size = 1" $ do
      let sr = mnistIter (#image := "dataiter/test/data/train-images-idx3-ubyte" .&
                          #label := "dataiter/test/data/train-labels-idx1-ubyte" .&
                          #batch_size := 1 .& Nil) :: DS
      sizeD sr `shouldReturn` 60000
    it "batch-size = 32" $ do
      let sr = mnistIter (#image := "dataiter/test/data/train-images-idx3-ubyte" .&
                          #label := "dataiter/test/data/train-labels-idx1-ubyte" .&
                          #batch_size := 32 .& Nil) :: DS
      sizeD sr `shouldReturn` 1875
    it "batch-size = 128" $ do
      let sr = mnistIter (#image := "dataiter/test/data/train-images-idx3-ubyte" .&
                          #label := "dataiter/test/data/train-labels-idx1-ubyte" .&
                          #batch_size := 128 .& Nil) :: DS
      sizeD sr `shouldReturn` 468
  describe "ImageRecordIter" $ do
    it "batch-size = 32" $ do
      let sr = imageRecordIter (#path_imgrec := "dataiter/test/data/cifar10_val.rec" .&
                                #data_shape  := [3,28,28] .&
                                #batch_size  := 32 .& Nil) :: DS
      sizeD sr `shouldReturn` 313
    it "batch-size = 128" $ do
      let sr = imageRecordIter (#path_imgrec := "dataiter/test/data/cifar10_val.rec" .&
                                #data_shape  := [3,28,28] .&
                                #batch_size  := 128 .& Nil) :: DS
      sizeD sr `shouldReturn` 79
