import numpy as np


def softmax (inputValue):

    exp = np.exp(inputValue)
    #print(exp.shape)

    # Sum along classes
    exp_sum = np.sum(exp,axis=1)
    #print(exp_sum.shape)

    # Softmax calc.
    output = exp.T / exp_sum
    output = output.T

    return output