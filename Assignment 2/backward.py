import numpy as np

import sigmoid
import softmax


def backward (data, labels, weight_inputToHidden, weight_hiddenToOutput, biasHidden, biasOutput, reg = 1e-5):
	
    # hidden = w1 * x + b1
    # hidden_hat = sigmoid( hidden )
    hidden = np.dot(data, weight_inputToHidden)
    hidden_withBias = hidden + biasHidden
    hidden_sigmoid = sigmoid.sigmoid(hidden_withBias)

    # print(np.shape(hidden))
    # print(np.shape(hidden_witBias))
    # print('Hidden sigmoid', np.shape(hidden_sigmoid))

    # y = w2 * hidden + b2
    # y_hat = softmax( y )
    preSoftmax = np.dot(hidden_sigmoid, weight_hiddenToOutput)
    preSoftmax_withBias = preSoftmax + biasOutput
    #print('pre softmax ', preSoftmax_withBias.shape)

    # To solve numerical instability
    preSoftmax_withBiasStable = preSoftmax_withBias.T
    preSoftmax_withBiasStable -= np.max(preSoftmax_withBiasStable, axis=0)
    preSoftmax_withBiasStable = preSoftmax_withBiasStable.T
    # print('Stabilised ', preSoftmax_withBiasStable.shape)
    # print('value ', preSoftmax_withBiasStable)

    expPre = np.exp(preSoftmax_withBiasStable)
    #print(exp.shape)

    # Sum along classes
    exp_sumPre = np.sum(expPre,axis=0)
    #print(exp_sum.shape)

    # Softmax calc.
    output = expPre / exp_sumPre
    output = output

    # output = softmax.softmax(preSoftmax_withBiasStable)


    # print(output[3])

    # Pre-process necessary matrices

    # Geting dims of matrix
    (D, C) = weight_hiddenToOutput.shape
    N = 1
    # print('Hidden units', D)
    # print('Classes', C)
    # print('Num Images', N)

    # Making matrix with acutal class assignment
    labelActual = np.zeros(shape=(N, C))
    labelActual[np.arange(N), labels] = 1
    # print(labelActual)

    # print(labelActual[5000])

    # Loss [ Cross Entropy ]
    exp = np.exp(output)
    exp_sum = np.sum(exp, axis=0)
    Loss_log_term = np.log(exp_sum)
    # print(Loss_log_term)
    output_actual = output[labels]
    # print(output_actual)
    loss_per_image = Loss_log_term - output_actual

    # Overall Loss with regularization
    lossBeforeReg = np.sum(loss_per_image)/float(N)
    RegPenalty = np.sum(weight_inputToHidden * weight_inputToHidden) + \
                 np.sum(weight_hiddenToOutput * weight_hiddenToOutput)
    regLoss = 0.5 * reg * RegPenalty
    loss = lossBeforeReg + regLoss

    # print('Loss ', loss)

    # print('hidden to op', weight_hiddenToOutput.shape)
    # print('ip ot hidden', weight_inputToHidden.shape)

    # Gradient Calculations
    # print(req_variable[5000])
    # print('req_variable', req_variable.shape)

    # dy_hat_dy = output * (labelActual - output)
    # print(dy_hat_dy)
    #
    # dL_dy_hat = - 1 / output
    #
    # dy_dv = hidden_sigmoid
    #
    dy_dh_hat = weight_hiddenToOutput
    #
    # dh_dw = data

    dh_hat_dh = hidden_sigmoid * (1 - hidden_sigmoid)
    # print(dh_hat_dh)
    dL_dy
    # dL_dy = dL_dy_hat * dy_hat_dy
     = (output - labelActual)
    # print(dL_dy.shape)

    hidden_sigmoid = np.reshape(hidden_sigmoid,(-1,D))

    dL_dv = np.dot(hidden_sigmoid.T, dL_dy)
    # print(dL_dv.shape)
    # print(hidden_sigmoid[2])
    # dL_dv = dL_dv / N  + (reg * weight_hiddenToOutput)

    # dL_dh_hat = np.dot(dL_dy, weight_hiddenToOutput.T)
    #
    # dL_dh = dL_dh_hat * dh_hat_dh
    # print(dL_dh.shape)
    # print(dL_dh[2])

    # print(data.shape)
    # print(data[2])

    dy_dh = dy_dh_hat.T * dh_hat_dh
    # print(np.shape(dy_dh))

    dL_dh = np.dot(dL_dy, dy_dh)
    # print(dL_dh.shape)

    data = np.reshape(data, (-1,784))

    dL_dw = np.dot(data.T, dL_dh)
    # print(dL_dw.shape)
    # dL_dw = dL_dw / N + (reg * weight_inputToHidden)

    # dL_dbO =  np.sum(dL_dy,axis=0)
    dL_dbO = dL_dy
    # print(dL_dbO.shape)
    dL_dbO = dL_dbO.reshape(-1)
    # print(dL_dbO.shape)

    # dL_dbH = np.sum(dL_dh,axis=0)
    dL_dbH = dL_dh
    dL_dbH = dL_dbH.reshape(-1)

    return loss, dL_dv, dL_dw, dL_dbO, dL_dbH

