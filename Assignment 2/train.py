import backward
import forward
import numpy as np
import timeit


def train(X_train, y_train, weight_inputToHidden, weight_hiddenToOutput, biasHidden, biasOutput,
          learningRate=1e-3, reg=1e-5, num_iters=100, verbose=False):

    # Run stochastic gradient descent to optimize W
    loss_history = []
    num_training, dataSize = np.shape(X_train)
    # print(num_training)

    # batch_size = 500

    for iterNum in range(num_iters):

        startTime = timeit.default_timer()

        for i in range(num_training):

            loss, grad_hiddenToOutput, grad_inputToHidden, grad_biasOutput, grad_biasHidden = backward.backward(X_train[i],
                                                                                                                y_train[i],
                                                                                                            weight_inputToHidden,
                                                                                                            weight_hiddenToOutput,
                                                                                                            biasHidden,
                                                                                                            biasOutput,
                                                                                                            reg)

            # update loss
            loss_history.append(loss)

            # update parameters
            weight_inputToHidden -= learningRate * grad_inputToHidden
            weight_hiddenToOutput -= learningRate * grad_hiddenToOutput
            biasHidden -= learningRate * grad_biasHidden
            biasOutput -= learningRate * grad_biasOutput

        stopTime = timeit.default_timer()

        if verbose and iterNum % 5 == 0:
            print('iteration %d / %d: loss %f' % (iterNum, num_iters, loss))

            print('Time Taken : ', stopTime - startTime)

            output, classOutput = forward.forward(X_train, weight_inputToHidden, weight_hiddenToOutput, biasHidden,
                                                  biasOutput)

            # print('Actual :   ', y_train [1 : 10])
            # print('Predicted :', classOutput [1:10])
            print('Training accuracy: {:f}'.format(np.mean(y_train == classOutput)))
            # print('Iter ', iterNum, ' -- training accuracy: {:f}'.format(np.mean(y_train == classOutput)))

    return loss_history, weight_inputToHidden, weight_hiddenToOutput, biasHidden, biasOutput