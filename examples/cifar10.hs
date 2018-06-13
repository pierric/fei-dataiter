{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
module Main where

import MXNet.Core.Base (NDArray, contextCPU, contextGPU, mxListAllOpNames)
import MXNet.Core.Base.HMap
import qualified MXNet.Core.Base.NDArray as A
import qualified MXNet.Core.Base.Internal.TH.NDArray as A
import qualified Data.HashMap.Strict as M
import Control.Monad (forM_, void)
import qualified Data.Vector.Storable as SV
import Control.Monad.IO.Class
import System.IO (hFlush, stdout)
import Options.Applicative (Parser, execParser, header, info, fullDesc, helper, value, option, auto, metavar, short, showDefault, (<**>))
import Data.Semigroup ((<>))
import MXNet.NN
import MXNet.NN.Utils
import MXNet.NN.Utils.HMap
import MXNet.NN.DataIter.Class
import MXNet.Core.IO.DataIter.Conduit
import qualified Model.Resnet as Resnet
import qualified Model.Resnext as Resnext

type ArrayF = NDArray Float
type DS = ConduitData (TrainM Float IO) (ArrayF, ArrayF)

data Model   = Resnet | Resnext deriving (Show, Read)
data ProgArg = ProgArg Model
cmdArgParser :: Parser ProgArg
cmdArgParser = ProgArg <$> (option auto $ short 'm' <> metavar "MODEL" <> showDefault <> value Resnet)

range :: Int -> [Int]
range = enumFromTo 1

default_initializer :: Initializer Float
default_initializer name shp
    | endsWith "-bias"  name = zeros name shp
    | endsWith "-beta"  name = zeros name shp
    | endsWith "-gamma" name = ones  name shp
    | endsWith "-moving-mean" name = zeros name shp
    | endsWith "-moving-var"  name = ones  name shp
    | otherwise = case shp of 
                    [_,_] -> xavier 2.0 XavierGaussian XavierIn name shp
                    _ -> normal 0.1 name shp

main :: IO ()
main = do
    ProgArg model <- execParser $ info (cmdArgParser <**> helper) (fullDesc <> header "CIFAR-10 solver")
    -- call mxListAllOpNames can ensure the MXNet itself is properly initialized
    -- i.e. MXNet operators are registered in the NNVM
    _    <- mxListAllOpNames
    net  <- case model of 
              Resnet  -> Resnet.symbol
              Resnext -> Resnext.symbol
    sess <- initialize net $ Config { 
                _cfg_placeholders = M.singleton "x" [1,3,28,28],
                _cfg_initializers = M.empty,
                _cfg_default_initializer = default_initializer,
                _cfg_context = contextCPU
            }
    optimizer <- makeOptimizer (SGD'Mom $ lrOfMultifactor 0.2 [200,500] nil) nil

    train sess $ do 

        let trainingData = imageRecordIter [α| path_imgrec := "test/data/cifar10_train.rec",
                                               data_shape  := [3,28,28] :: [Int],
                                               batch_size  := 128 :: Int |]
        let testingData  = imageRecordIter [α| path_imgrec := "test/data/cifar10_val.rec",
                                               data_shape  := [3,28,28] :: [Int],
                                               batch_size  := 32 :: Int |]
        total1 <- sizeD trainingData
        liftIO $ putStrLn $ "[Train] "
        forM_ (range 1) $ \ind -> do
            liftIO $ putStrLn $ "iteration " ++ show ind
            metric <- metricCE ["y"] ## metricLR
            void $ forEachD_i trainingData $ \(i, (x, y)) -> do
                fitAndEval optimizer net (M.fromList [("x", x), ("y", y)]) metric
                eval <- format metric
                liftIO $ do
                   putStr $ "\r\ESC[K" ++ show i ++ "/" ++ show total1 ++ " " ++ eval
                   hFlush stdout
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
