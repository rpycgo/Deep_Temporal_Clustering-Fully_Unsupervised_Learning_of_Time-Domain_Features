from ...config.config import model_config

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, UpSampling1D, Conv1DTranspose


class Decoder(Layer):
    def __init__(self, config=model_config, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.config = config
        self.upsampling = UpSampling1D(size=config.P)
        self.deconvolution = Conv1DTranspose(filters=config.n_features, kernel_size=config.kernel_size+1)
    
    def call(self, inputs, training=None):
        x = self.upsampling(inputs)        
        output = self.deconvolution(x)
        
        return output


def build_decoder(config=model_config):
    input = Input(shape=((config.time_seq-config.kernel_size+1)//config.P, 2))
    output = Decoder(config)(input)

    return Model(input, output)
