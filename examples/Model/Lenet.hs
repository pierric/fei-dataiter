{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Model.Lenet (symbol) where

import qualified MXNet.Core.Base.Symbol as S
import MXNet.Core.Base (DType, Symbol)
import MXNet.NN.Layer
import MXNet.NN.Utils.HMap

-- # first conv
-- conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
-- tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
-- pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
-- # second conv
-- conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
-- tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
-- pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
-- # first fullc
-- flatten = mx.symbol.Flatten(data=pool2)
-- fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
-- tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
-- # second fullc
-- fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=num_classes)
-- # loss
-- lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

symbol :: DType a => IO (Symbol a)
symbol = do
    x  <- variable "x"
    y  <- variable "y"

    v1 <- convolution "conv1" x [5,5] 20 [α||]
    a1 <- activation "conv1-a" v1 Tanh
    p1 <- pooling "conv1-p" a1 [2,2] PoolingMax [α||]

    v2 <- convolution "conv2" p1 [5,5] 50 [α||]
    a2 <- activation "conv2-a" v2 Tanh
    p2 <- pooling "conv2-p" a2 [2,2] PoolingMax [α||]

    fl <- flatten "flatten" p2

    v3 <- fullyConnected "fc1" fl 500 [α||]
    a3 <- activation "fc1-a" v3 Tanh

    v4 <- fullyConnected "fc2" a3 10 [α||]
    a4 <- softmaxoutput "softmax" v4 y [α||]
    return $ S.Symbol a4