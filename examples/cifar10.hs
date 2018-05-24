{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
module Main where

import MXNet.Core.Base (DType, NDArray, Symbol, contextCPU, contextGPU, mxListAllOpNames)
import MXNet.Core.Base.HMap
import qualified MXNet.Core.Base.NDArray as A
import qualified MXNet.Core.Base.Internal.TH.NDArray as A
import qualified MXNet.Core.Base.Symbol as S
import qualified Data.HashMap.Strict as M
import Control.Monad (forM_, void, foldM)
import qualified Data.Vector.Storable as SV
import Control.Monad.IO.Class
import System.IO (hFlush, stdout)
import MXNet.NN
import MXNet.NN.Layer
import MXNet.NN.EvalMetric
import MXNet.NN.Initializer
import MXNet.NN.DataIter.Class
import MXNet.Core.IO.DataIter.Conduit

type ArrayF = NDArray Float
type SymbolF = Symbol Float
type DS = ConduitData (TrainM Float IO) (ArrayF, ArrayF)

neural :: IO SymbolF
neural = do
    x  <- variable "x"
    y  <- variable "y"
    bnx <- batchnorm "bn-x" x (add @"eps" eps $ add @"momentum" bn_mom $ add @"fix_gamma" True nil)
    cvx <- convolution "conv-bn-x" bnx [3,3] 16 (add @"stride" "[1,1]" $ add @"pad" "[1,1]" $ add @"workspace" conv_workspace $ add @"no_bias" True nil)
    bdy <- foldM (\layer (num_filter, stride, dim_match, name) -> residual name layer num_filter stride dim_match resargs) cvx residual'parms
    bn1 <- batchnorm "bn-1" bdy (add @"eps" eps $ add @"momentum" bn_mom $ add @"fix_gamma" False nil)
    ac1 <- activation "relu-1" bn1 Relu
    pl1 <- pooling "pool-1" ac1 [7,7] PoolingAvg (add @"global_pool" True nil)
    flt <- flatten "flt-1" pl1
    fc1 <- fullyConnected "fc-1" flt 10 nil
    S.Symbol <$> softmaxoutput "softmax" fc1 y nil
  where
    bn_mom = 0.9 :: Float
    conv_workspace = 256 :: Int
    eps = 2e-5 :: Double
    residual'parms = [ (16, [1,1], False, "stage1-unit1"), (16, [1,1], True, "stage1-unit2"), (16, [1,1], True, "stage1-unit3")
                     , (32, [2,2], False, "stage2-unit1"), (32, [1,1], True, "stage2-unit2"), (32, [1,1], True, "stage2-unit3")
                     , (64, [2,2], False, "stage3-unit1"), (64, [1,1], True, "stage3-unit2"), (64, [1,1], True, "stage3-unit3") ]
    resargs = add @"bottle_neck" False $ add @"workspace" conv_workspace $ add @"memonger" False nil

range :: Int -> [Int]
range = enumFromTo 1

default_initializer :: Initializer Float
default_initializer shp@[_]   = zeros shp
default_initializer shp@[_,_] = xavier 2.0 XavierGaussian XavierIn shp
default_initializer shp = normal 0.1 shp
    
main :: IO ()
main = do
    -- call mxListAllOpNames can ensure the MXNet itself is properly initialized
    -- i.e. MXNet operators are registered in the NNVM
    _    <- mxListAllOpNames
    net  <- neural
    sess <- initialize net $ Config { 
                _cfg_placeholders = M.singleton "x" [1,1,28,28],
                _cfg_initializers = M.empty,
                _cfg_default_initializer = default_initializer,
                _cfg_context = contextCPU
            }
    optimizer <- makeOptimizer (SGD'Mom 0.0002) nil

    train sess $ do 

        let trainingData = imageRecordIter (add @"path_imgrec" "test/data/cifar10_train.rec" $ 
                                            add @"data_shape" ([3,28,28] :: [Int]) $
                                            add @"batch_size" (32 :: Int) nil) :: DS
        let testingData  = imageRecordIter (add @"path_imgrec" "test/data/cifar10_val.rec" $ 
                                            add @"data_shape" ([3,28,28] :: [Int]) $
                                            add @"batch_size" (32 :: Int) nil) :: DS
        total1 <- sizeD trainingData
        liftIO $ putStrLn $ "[Train] "
        forM_ (range 1) $ \ind -> do
            liftIO $ putStrLn $ "iteration " ++ show ind
            metric <- newMetric CrossEntropy "CrossEntropy" ["y"]
            void $ forEachD_i trainingData $ \(i, (x, y)) -> do
                liftIO $ do
                   eval <- formatMetric metric
                   putStr $ "\r\ESC[K" ++ show i ++ "/" ++ show total1 ++ " " ++ eval
                   hFlush stdout
                fitAndEval optimizer net (M.fromList [("x", x), ("y", y)]) metric
            liftIO $ putStrLn ""
        
        liftIO $ putStrLn $ "[Test] "

        total2 <- sizeD testingData
        result <- forEachD_i testingData $ \(i, (x, y)) -> do 
            liftIO $ do 
                putStr $ "\r\ESC[K" ++ show i ++ "/" ++ show total2
                hFlush stdout
            [y'] <- forwardOnly net (M.fromList [("x", Just x), ("y", Nothing)])
            ind1 <- liftIO $ A.items y
            ind2 <- liftIO $ argmax y' >>= A.items
            return (ind1, ind2)
        liftIO $ putStr "\r\ESC[K"

        let (ls,ps) = unzip result
            ls_unbatched = mconcat ls
            ps_unbatched = mconcat ps
            total_test_items = SV.length ls_unbatched
            correct = SV.length $ SV.filter id $ SV.zipWith (==) ls_unbatched ps_unbatched
        liftIO $ putStrLn $ "Accuracy: " ++ show correct ++ "/" ++ show total_test_items
  
  where
    argmax :: ArrayF -> IO ArrayF
    argmax ys = A.NDArray <$> A.argmax (A.getHandle ys) (add @"axis" (1 :: Int) nil)
