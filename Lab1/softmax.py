import numpy as np
from random import shuffle
import pdb


def softmax_loss(W, b, X, y, reg):
    """
    Softmax loss function.

    Inputs have dimension D, there are C classes, and we operate on mini batches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - b: A numpy array of shape (C,) containing biases.
    - X: A numpy array of shape (N, D) containing a mini batch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    - gradient with respect to weights b; an array of same shape as b

    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no for loops.       #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    # Finding Y = W X + b
    y_calc = np.dot(X, W)
    y_calc = np.add(y_calc, b)
    y_calc = y_calc.T

    # To solve numerical instability
    y_calc -= np.max(y_calc, axis=0)

    # Calc Softmax val
    y_exp = np.exp(y_calc)
    y_sum_exp = np.sum(y_exp, axis=0)  # along classes [cloumn]

    # Geting dims of matrix
    (D, C) = W.shape
    N = int(y.shape[0])
    # print(D)
    # print(C)
    # print(N)

    # Making matrix with acutal class assignment
    y_correct = np.zeros(shape=(C, N))
    y_correct[y, np.arange(N)] = 1

    # Calculating Loss
    # L = - y_calc(of actual class) + log(sum(exp(y_calc)))
    Loss_log_term = np.log(y_sum_exp)
    y_actual = y_calc[y, np.arange(N)]
    loss_per_image = Loss_log_term - y_actual

    # Overall Loss with regularization
    lossBeforeReg = np.sum(loss_per_image) / float(N)
    RegPenalty = np.sum(W * W)
    regLoss = 0.5 * reg * RegPenalty
    loss = lossBeforeReg + regLoss

    # Calculating dW
    dw1 = y_exp / (y_sum_exp)
    dw2 = dw1 - y_correct
    dw3 = np.dot(dw2, X)
    dW = dw3.T / N + reg * W

    # Calculating db
    db1 = dw2
    db2 = np.sum(db1, axis=1)
    db = db2.T / N + np.sum(W * W)

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW, db