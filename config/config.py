class model_config:
    batch_size = 64
    time_seq = 20
    n_features = 30
    learning_rate = 1e-1

    # CNN
    filters = 50
    kernel_size = 10
    P = 5   # pooling size, upsampling size

    # LSTM
    units1 = 50
    units2 = 1

    # clustering
    k = 10
    alpha = 1.
