from ...config.config import model_config

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense, Conv1D, LeakyReLU, MaxPool1D, Bidirectional, LSTM


class Encoder(Layer):
    def __init__(self, config=model_config, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.config = config
        self.conv1 = Conv1D(filters=self.config.filters, kernel_size=self.config.kernel_size, activation=LeakyReLU())
        self.max_pooling = MaxPool1D(pool_size=self.config.P)
        self.bi_lstm1 = Bidirectional(LSTM(units=self.config.units1, return_sequences=True))
        self.bi_lstm2 = Bidirectional(LSTM(units=self.config.units2, return_sequences=True))
    
    def call(self, input, training=None):
        # CNN
        x = self.conv1(input)
        x = self.max_pooling(x)

        # LSTM
        x = self.bi_lstm1(x)
        output = self.bi_lstm2(x)

        return output


def build_encoder(config=model_config):
    input = Input(shape=(config.time_seq, config.n_features))
    output = Encoder(config)(input)

    return Model(input, output)
