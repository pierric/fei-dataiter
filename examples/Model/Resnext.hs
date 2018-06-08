{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
module Model.Resnext where

import Control.Monad (foldM, when, void)
import MXNet.Core.Base.HMap
import MXNet.Core.Types.Internal (SymbolHandle)
import qualified MXNet.Core.Base.Symbol as S
import qualified MXNet.Core.Base.Internal as I
import MXNet.Core.Base (DType, Symbol)
import MXNet.NN.Layer
import MXNet.NN.Utils.HMap

-- ResNet
-- #layer: 164
-- #stage: 3
-- #layer per stage: 18
-- #filter of stage 1: 64
-- #filter of stage 2: 128
-- #filter of stage 3: 256

symbol :: DType a => IO (Symbol a)
symbol = do
    x  <- variable "x"
    y  <- variable "y"

    xcp <- identity "id" x

    bnx <- batchnorm "bn-x" xcp  [α| eps := eps, momentum := bn_mom, fix_gamma := True |]
    cvx <- convolution "conv-bn-x" bnx [3,3] 16 [α| stride := "[1,1]",  pad := "[1,1]", workspace := conv_workspace, no_bias := True |]
    
    bdy <- foldM (\layer (num_filter, stride, dim_match, name) -> residual name layer num_filter stride dim_match resargs) cvx residual'parms
    
    pool1 <- pooling "pool1" bdy [7,7] PoolingAvg [α| global_pool := True |]
    flat  <- flatten "flat-1" pool1
    fc1   <- fullyConnected "fc-1" flat 10 [α| |]    
    S.Symbol <$> softmaxoutput "softmax" fc1 y [α| |]
  where
    bn_mom = 0.9 :: Float
    conv_workspace = 256 :: Int
    eps = 2e-5 :: Double
    residual'parms =  [ (64,  [1,1], False, "stage1-unit1") ] ++ map (\i -> (64,  [1,1], True, "stage1-unit" ++ show i)) [2..18 :: Int]
                   ++ [ (128, [2,2], False, "stage2-unit1") ] ++ map (\i -> (128, [1,1], True, "stage2-unit" ++ show i)) [2..18 :: Int]
                   ++ [ (256, [2,2], False, "stage3-unit1") ] ++ map (\i -> (256, [1,1], True, "stage3-unit" ++ show i)) [2..18 :: Int]
    resargs = [α| bottle_neck := True, workspace := conv_workspace, memonger := False |]

type ResidualOptArgs = '["bottle_neck" ':= Bool, "num_group" ':= Int, "bn_mom" ':= Float, "workspace" ':= Int, "memonger" ':= Bool]
residual :: (MatchKVList kvs ResidualOptArgs, ShowKV kvs) 
         => String -> SymbolHandle -> Int -> [Int] -> Bool -> HMap kvs -> IO SymbolHandle
residual name dat num_filter stride dim_match oargs = do
    let args = mergeTo oargs (True .+. 32 .+. 0.9 .+. 256 .+. False .+. nil) :: HMap ResidualOptArgs
        workspace = get @"workspace" args :: Int
        bn_mom = get @"bn_mom" args :: Float
        num_group = get @"num_group" args :: Int
        eps = 2e-5 :: Double
    if get @"bottle_neck" args
      then do
        conv1 <- convolution (name ++ "-conv1") dat [1,1] (num_filter `div` 2) 
                     [α| stride    := "[1,1]"
                       , pad       := "[0,0]"
                       , workspace := workspace
                       , no_bias   := True |]
        bn1 <- batchnorm (name ++ "-bn1") conv1 
                     [α| eps       := eps
                       , momentum  := bn_mom
                       , fix_gamma := False |]
        act1 <- activation (name ++ "-relu1") bn1 Relu
        conv2 <- convolution (name ++ "-conv2") act1 [3,3] (num_filter `div` 2) 
                     [α| stride    := (show stride)
                       , pad       := "[1,1]"
                       , num_group := num_group
                       , workspace := workspace
                       , no_bias   := True |]        
        bn2 <- batchnorm (name ++ "-bn2") conv2 
                     [α| eps       := eps
                       , momentum  := bn_mom
                       , fix_gamma := False |]
        act2 <- activation (name ++ "-relu2") bn2 Relu
        conv3 <- convolution (name ++ "-conv3") act2 [1,1] num_filter 
                     [α| stride    := "[1,1]"
                       , pad       := "[0,0]"
                       , workspace := workspace
                       , no_bias   := True |]
        bn3 <- batchnorm (name ++ "-bn3") conv3
                     [α| eps       := eps
                       , momentum  := bn_mom
                       , fix_gamma := False |]
        shortcut <- if dim_match
                    then return dat
                    else do
                        shortcut_conv <- convolution (name ++ "-sc") dat [1,1] num_filter 
                                [α| stride    := (show stride)
                                  , workspace := workspace
                                  , no_bias   := True |]
                        batchnorm (name ++ "-sc-bn") shortcut_conv 
                                [α| eps       := eps
                                  , momentum  := bn_mom
                                  , fix_gamma := False |]
        when (get @"memonger" args) $ void $ I.mxSymbolSetAttr shortcut "mirror_stage" "true"
        eltwise <- plus name bn3 shortcut
        activation (name ++ "-relu") eltwise Relu
      else do
        conv1 <- convolution (name ++ "-conv1") dat [3,3] num_filter 
                     [α| stride    := (show stride)
                       , pad       :="[1,1]"
                       , workspace := workspace
                       , no_bias   := True |]
        bn1 <- batchnorm (name ++ "-bn1") conv1 
                     [α| eps       := eps
                       , momentum  := bn_mom
                       , fix_gamma := False |]
        act1 <- activation (name ++ "-relu1") bn1 Relu
        conv2 <- convolution (name ++ "-conv2") act1 [3,3] num_filter 
                     [α| stride    := "[1,1]"
                       , pad       := "[1,1]"
                       , workspace := workspace
                       , no_bias   := True |]
        bn2 <- batchnorm (name ++ "-bn2") conv2 
                     [α| eps       := eps
                       , momentum  := bn_mom
                       , fix_gamma := False |]
        shortcut <- if dim_match
                    then return dat
                    else do
                        shortcut_conv <- convolution (name ++ "-sc") act1 [1,1] num_filter 
                               [α| stride    := (show stride)
                                 , workspace := workspace
                                 , no_bias   := True |]
                        batchnorm (name ++ "-sc-bn") shortcut_conv 
                                [α| eps       := eps
                                  , momentum  := bn_mom
                                  , fix_gamma := False |]
        when (get @"memonger" args) $ void $ I.mxSymbolSetAttr shortcut "mirror_stage" "true"
        eltwise <- plus name bn2 shortcut
        activation (name ++ "-relu") eltwise Relu