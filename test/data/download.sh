#! /bin/bash

# CIFAR-10 test
curl http://data.dmlc.ml/data/cifar10/cifar10_val.rec -o cifar10_val.rec

# MNIST
MNIST="train-images-idx3-ubyte train-labels-idx1-ubyte"
for fn in $MNIST; do
    curl http://yann.lecun.com/exdb/mnist/$fn.gz | gunzip - > $fn
done
