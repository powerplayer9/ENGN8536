import numpy as np


def sigmoid (inputValue):
    output = 1 / (1 + np.exp(-inputValue))

    return output