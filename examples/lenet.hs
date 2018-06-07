{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
module Main where

import MXNet.Core.Base (DType, NDArray, Symbol, contextCPU, contextGPU, mxListAllOpNames)
import MXNet.Core.Base.HMap
import qualified MXNet.Core.Base.NDArray as A
import qualified MXNet.Core.Base.Internal.TH.NDArray as A
import qualified Data.HashMap.Strict as M
import Control.Monad (forM_, void)
import qualified Data.Vector.Storable as SV
import Control.Monad.IO.Class
import System.IO (hFlush, stdout)
import MXNet.NN
import MXNet.NN.Utils.HMap
import MXNet.NN.EvalMetric
import MXNet.NN.Initializer
import MXNet.NN.DataIter.Class
import MXNet.Core.IO.DataIter.Conduit
import qualified Model.Lenet as Model

type ArrayF = NDArray Float
type SymbolF = Symbol Float
type DS = ConduitData (TrainM Float IO) (ArrayF, ArrayF)

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
    net  <- Model.symbol
    sess <- initialize net $ Config { 
                _cfg_placeholders = M.singleton "x" [1,1,28,28],
                _cfg_initializers = M.empty,
                _cfg_default_initializer = default_initializer,
                _cfg_context = contextCPU
            }
    optimizer <- makeOptimizer (SGD'Mom 0.0002) nil

    train sess $ do 

        let trainingData = mnistIter [α| image := "test/data/train-images-idx3-ubyte",
                                         label := "test/data/train-labels-idx1-ubyte",
                                         batch_size := 128 :: Int |] :: DS
        let testingData  = mnistIter [α| image := "test/data/t10k-images-idx3-ubyte",
                                         label := "test/data/t10k-labels-idx1-ubyte",
                                         batch_size := 16 :: Int  |] :: DS

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
    argmax ys = A.NDArray <$> A.argmax (A.getHandle ys) [α| axis := 1 :: Int |]
