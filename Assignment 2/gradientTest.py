import grad_check
import backward


def gradTest(X_train, y_train, weight_inputToHidden, weight_hiddenToOutput, biasHidden, biasOutput, grad_hiddenToOutput,
             grad_inputToHidden, grad_biasOutput, grad_biasHidden, reg = 0.0):

    # Your analytical gradient should be close to numerical gradient, error less than 1e-5.

    print('dV ')
    f = lambda weight_hiddenToOutput: backward.backward(X_train, y_train,
                                            weight_inputToHidden, weight_hiddenToOutput,
                                            biasHidden, biasOutput,
                                    reg)[0]
    grad_numerical_w = grad_check.grad_check(f, weight_hiddenToOutput, grad_hiddenToOutput, 5)

    print('dw ')
    f = lambda weight_inputToHidden: backward.backward(X_train, y_train,
                                            weight_inputToHidden, weight_hiddenToOutput,
                                            biasHidden, biasOutput,
                                    reg)[0]
    grad_numerical_w = grad_check.grad_check(f, weight_inputToHidden, grad_inputToHidden, 5)

    print('dbO ')
    f = lambda biasOutput: backward.backward(X_train, y_train,
                                            weight_inputToHidden, weight_hiddenToOutput,
                                            biasHidden, biasOutput,
                                    reg)[0]
    grad_numerical_w = grad_check.grad_check(f, biasOutput, grad_biasOutput, 5)

    print('dbH ')
    f = lambda biasHidden: backward.backward(X_train, y_train,
                                            weight_inputToHidden, weight_hiddenToOutput,
                                            biasHidden, biasOutput,
                                    reg)[0]
    grad_numerical_w = grad_check.grad_check(f, biasHidden, grad_biasHidden, 5)