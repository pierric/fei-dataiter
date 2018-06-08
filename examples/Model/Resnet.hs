{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}

module Model.Resnet (symbol) where

import Control.Monad (foldM, when, void)
import MXNet.Core.Base.HMap
import MXNet.Core.Types.Internal (SymbolHandle)
import qualified MXNet.Core.Base.Symbol as S
import qualified MXNet.Core.Base.Internal as I
import MXNet.Core.Base (DType, Symbol)
import MXNet.NN.Layer
import MXNet.NN.Utils.HMap

-------------------------------------------------------------------------------
-- ResNet
-- #layer: 110
-- #stage: 3
-- #layer per stage: 18
-- #filter of stage 1: 16
-- #filter of stage 2: 32
-- #filter of stage 3: 64
symbol :: DType a => IO (Symbol a)
symbol = do
    x  <- variable "x"
    y  <- variable "y"

    xcp <- identity "id" x

    bnx <- batchnorm "bn-x" xcp  [α| eps := eps, momentum := bn_mom, fix_gamma := True |]
    cvx <- convolution "conv-bn-x" bnx [3,3] 16 [α| stride := "[1,1]",  pad := "[1,1]", workspace := conv_workspace, no_bias := True |]
    
    bdy <- foldM (\layer (num_filter, stride, dim_match, name) -> residual name layer num_filter stride dim_match resargs) cvx residual'parms
    
    bn1 <- batchnorm "bn-1" bdy [α| eps := eps, momentum := bn_mom, fix_gamma := False |]
    ac1 <- activation "relu-1" bn1 Relu
    pl1 <- pooling "pool-1" ac1 [7,7] PoolingAvg [α| global_pool := True |]
    
    flt <- flatten "flt-1" pl1
    fc1 <- fullyConnected "fc-1" flt 10 [α| |]
    
    S.Symbol <$> softmaxoutput "softmax" fc1 y [α| |]
  where
    bn_mom = 0.9 :: Float
    conv_workspace = 256 :: Int
    eps = 2e-5 :: Double
    residual'parms =  [ (16, [1,1], False, "stage1-unit1") ] ++ map (\i -> (16, [1,1], True, "stage1-unit" ++ show i)) [2..18 :: Int]
                   ++ [ (32, [2,2], False, "stage2-unit1") ] ++ map (\i -> (32, [1,1], True, "stage2-unit" ++ show i)) [2..18 :: Int]
                   ++ [ (64, [2,2], False, "stage3-unit1") ] ++ map (\i -> (64, [1,1], True, "stage3-unit" ++ show i)) [2..18 :: Int]
    resargs = [α| bottle_neck := False, workspace := conv_workspace, memonger := False |]

type ResidualOptArgs = '["bottle_neck" ':= Bool, "bn_mom" ':= Float, "workspace" ':= Int, "memonger" ':= Bool]
residual :: (MatchKVList kvs ResidualOptArgs, ShowKV kvs) 
         => String -> SymbolHandle -> Int -> [Int] -> Bool -> HMap kvs -> IO SymbolHandle
residual name dat num_filter stride dim_match oargs = do
    let args = mergeTo oargs (True .+. 0.9 .+. 256 .+. False .+. nil) :: HMap ResidualOptArgs
        workspace = get @"workspace" args :: Int
        bn_mom = get @"bn_mom" args :: Float
        eps = 2e-5 :: Double
    if get @"bottle_neck" args 
      then do
        bn1 <- batchnorm (name ++ "-bn1") dat 
                     [α| eps       := eps
                       , momentum  := bn_mom
                       , fix_gamma := False |]
        act1 <- activation (name ++ "-relu1") bn1 Relu
        conv1 <- convolution (name ++ "-conv1") act1 [1,1] (num_filter `div` 4) 
                     [α| stride    := "[1,1]"
                       , pad       := "[0,0]"
                       , workspace := workspace
                       , no_bias   := True |]
        bn2 <- batchnorm (name ++ "-bn2") conv1 
                     [α| eps       := eps
                       , momentum  := bn_mom
                       , fix_gamma := False |]
        act2 <- activation (name ++ "-relu2") bn2 Relu
        conv2 <- convolution (name ++ "-conv2") act2 [3,3] (num_filter `div` 4) 
                     [α| stride    := (show stride)
                       , pad       := "[1,1]"
                       , workspace := workspace
                       , no_bias   := True |]
        bn3 <- batchnorm (name ++ "-bn3") conv2
                     [α| eps       := eps
                       , momentum  := bn_mom
                       , fix_gamma := False |]
        act3 <- activation (name ++ "-relu3") bn3 Relu
        conv3 <- convolution (name ++ "-conv3") act3 [1,1] num_filter 
                     [α| stride    := "[1,1]"
                       , pad       := "[0,0]"
                       , workspace := workspace
                       , no_bias   := True |]
        shortcut <- if dim_match
                    then return dat
                    else convolution (name ++ "-sc") act1 [1,1] num_filter 
                               [α| stride    := (show stride)
                                 , workspace := workspace
                                 , no_bias   := True |]
        when (get @"memonger" args) $ void $ I.mxSymbolSetAttr shortcut "mirror_stage" "true"
        plus name conv3 shortcut
      else do
        bn1 <- batchnorm (name ++ "-bn1") dat 
                     [α| eps       := eps
                       , momentum  := bn_mom
                       , fix_gamma := False |]
        act1 <- activation (name ++ "-relu1") bn1 Relu
        conv1 <- convolution (name ++ "-conv1") act1 [3,3] num_filter 
                     [α| stride    := (show stride)
                       , pad       :="[1,1]"
                       , workspace := workspace
                       , no_bias   := True |]
        bn2 <- batchnorm (name ++ "-bn2") conv1 
                     [α| eps       := eps
                       , momentum  := bn_mom
                       , fix_gamma := False |]
        act2 <- activation (name ++ "-relu2") bn2 Relu
        conv2 <- convolution (name ++ "-conv2") act2 [3,3] num_filter 
                     [α| stride    := "[1,1]"
                       , pad       := "[1,1]"
                       , workspace := workspace
                       , no_bias   := True |]
        shortcut <- if dim_match
                    then return dat
                    else convolution (name ++ "-sc") act1 [1,1] num_filter 
                               [α| stride    := (show stride)
                                 , workspace := workspace
                                 , no_bias   := True |]
        when (get @"memonger" args) $ void $ I.mxSymbolSetAttr shortcut "mirror_stage" "true"
        plus name conv2 shortcut