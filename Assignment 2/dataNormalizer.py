def datasplit(X_train, y_train, X_test, y_test, num_training, num_validation, num_test):

    # subsample the data
    # Validation set
    maskVal = list(range(num_training, num_training + num_validation))
    X_val = X_train[maskVal]
    y_val = y_train[maskVal]

    # Training set
    maskTrain = list(range(num_training))
    X_train = X_train[maskTrain]
    y_train = y_train[maskTrain]

    # Subsample of test set
    maskTest = list(range(num_test))
    X_test = X_test[maskTest]
    y_test = y_test[maskTest]

    # # Normalize image
    # X_train = X_train / 255
    # X_val = X_val / 255
    # X_test = X_test / 255
    #
    # mean_image = np.mean(X_train, axis=0)
    # print(mean_image.shape)
    # X_train = X_train - mean_image
    # X_val = X_val - mean_image
    # X_test = X_test - mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test