# Digit recognizer

This repo is no more than just another "Hello world" neural network
implementation. It is implemented in C++23 without any external dependencies.
The neural network is trainable to recognize digits from the MNIST dataset:

- Input layer is 28 * 28.
- Has two hidden layers of 16 neurons each.
- Output layer is 10 neurons, one neuron corresponding to one digit.
- Activtion function is ReLU.
- All data (input, weights, biases) are 32 bit floating point numbers (MNIST
  image data is converted from 8 bit integer to the range 0.0-1.0 when loaded).


## Requirements

* g++ 13.3.0 or higher (or possibly another compiler that supports C++23)
* gnu make


## Usage

1. Download the MNIST training and test data. A bash script is provided for
   doing that, in the `data` folder. The steps in the script should also be
   possible to perform manually if need be. The rest of the instructions will
   assume the training data is stored under `data/train` and the testing data
   is stored under `data/test`.
2. Make the relevant targets:
```
make RELEASE=1 src/train
make RELEASE=1 src/modelstats
```
3. Make a folder to store weight data, e.g. from the project root directory do
   `mkdir weights`.
4. Initialize a weight file by running `src/train - weights/start.dat`. This
   will use He initialization.
5. Train one epoch by running `src/train weights/start.dat weights/epoch1.dat
   data/train/train-images-idx3-ubyte data/train/train-labels-idx1-ubyte`.
6. Test the model by running `src/modelstats weights/epoch1.dat
   data/test/t10k-images-idx3-ubyte data/test/t10k-labels-idx1-ubyte`.

After this you can train further epochs by running
```
src/train weights/epoch<n>.dat weights/epoch<n+1>.dat data/train/train-images-idx3-ubyte data/train/train-labels-idx1-ubyte
```
And test the new models by running
```
src/modelstats weights/epoch<n>.dat data/test/t10k-images-idx3-ubyte data/test/t10k-labels-idx1-ubyte
```
Note: In both commands, manually replace `<n>` and `<n+1>` with appropriate values.

## Background

This project started after being inspired by a series on neural networks by 3
blue 1 brown (3b1b) on
[youtube](https://m.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi).
The first four videos should explain most of the background required to
understand the implementation here. The network topology is the same as
explained in the video, but this implementation use ReLU rather than sigmoid as
activation function.

Some parts are however not thoroughly explained by the videos:
1. Proper network initialization. For networks with ReLU activations He
   initialization seems to be a popular choice.
2. Proper step size. After finding a gradient it is common to scale it by a
   number commonly referred to as learning rate, to prevent overshooting a
   minima.
3. Proper mini batch size. In this implementation the default mini batch size
   is 100. This type of optimization is a variation of what is known as
   Stochastic Gradient Descent or SGD for short.

All these properties seems to be subject to choice, the first point often
guided by some mathematics, the latter two (to my knowledge) often guided by
trial and error.
