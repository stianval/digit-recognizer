#!/usr/bin/env bash

set -e
set -x

data_path=$(dirname "${BASH_SOURCE[0]}")

cd ${data_path}
mkdir train; cd train
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz

cd ${data_path}
mkdir test; cd test
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
