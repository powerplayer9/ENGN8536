import mnist
import os
import numpy as np

import sigmoid
import softmax
import forward
import backward
import gradientTest
import dataNormalizer
import train
import grad_check



# Check if dataset exists else download
if (os.path.isfile('mnist.pkl') == False):
    mnist.init()

# Load the dataset
X_trainData, y_trainData, X_testData, y_testData = mnist.load()

# Parameters
Epoch = 10
numHiddenLayers = 100
numClasses = np.max(y_trainData) + 1
num_training = 59000
num_validation = 1000
num_test = 1000
learningRate = 1e-2
reg = 2.5e4
batch_size = 10

# Split & Normalize data set
X_train, y_train, X_val, y_val, X_test, y_test = dataNormalizer.datasplit(X_trainData, y_trainData,
                                                                          X_testData, y_testData,
                                                                          num_training, num_validation, num_test)

# Getting Data stats
num_data_train, data_size = np.shape(X_train)

# Initializing Weights
weight_inputToHidden = np.random.random((data_size, numHiddenLayers))
weight_hiddenToOutput = np.random.random((numHiddenLayers, numClasses))
biasHidden = np.random.random(numHiddenLayers)
biasOutput = np.random.random(numClasses)

print(np.shape(X_train))

# train
#
print('Training...')
loss, weight_inputToHidden, weight_hiddenToOutput, biasHidden, biasOutput =\
    train.train(X_train, y_train, weight_inputToHidden, weight_hiddenToOutput,
                biasHidden, biasOutput, learningRate, reg, Epoch, verbose=True)


# Forward / Prediction
print('')
output, classOutput = forward.forward(X_train, weight_inputToHidden, weight_hiddenToOutput, biasHidden, biasOutput)

print('training accuracy: {:f}'.format(np.mean(y_train == classOutput)))

print(classOutput)
print(output)
print(y_train)
