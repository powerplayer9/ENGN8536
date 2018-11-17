import numpy as np

import sigmoid
import softmax


def forward (data, weight_inputToHidden, weight_hiddenToOutput, biasHidden, biasOutput):

    # hidden = sigmoid( w1 * x + b1 )
    hidden = np.dot(data, weight_inputToHidden)
    hidden_withBias = hidden + biasHidden
    hidden_sigmoid = sigmoid.sigmoid(hidden_withBias)

    # print(np.shape(hidden))
    # print(np.shape(hidden_witBias))
    # print(hidden_sigmoid)

    # y = softmax( w2 * hidden + b2 )
    preSoftmax = np.dot(hidden_sigmoid, weight_hiddenToOutput)
    preSoftmax_withBias = preSoftmax + biasOutput

    expPre = np.exp(preSoftmax_withBias)
    #print(exp.shape)

    # Sum along classes
    exp_sumPre = np.sum(expPre,axis=0)
    #print(exp_sum.shape)

    # Softmax calc.
    output = expPre / exp_sumPre
    output = output.T

    # output = softmax.softmax(preSoftmax_withBias)

    classOutput = np.argmax(output, axis=0)

    # print(np.shape(classOutput))

    return output, classOutput

