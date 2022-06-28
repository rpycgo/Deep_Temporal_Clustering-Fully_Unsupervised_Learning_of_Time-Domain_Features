from ...config.config import model_config
from ..layers.decoder import Decoder
from ..layers.encoder import Encoder

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
    

class TemporalAutoencoder(Model):
    def __init__(self, config=model_config, **kwargs):
        super(TemporalAutoencoder, self).__init__(**kwargs)
        self.config = config
        self.encoder = Encoder(config, name='encoder')
        self.decoder = Decoder(config, name='decoder')
    
    def call(self, inputs, training=None):
        x = self.encoder(inputs)
        
        return self.decoder(x)


def build_temporal_autoencoder(config=model_config):
    inputs = Input(shape=(config.time_seq, config.n_features), name='input')
    encoder_output = Encoder(config, name='latent_vector')(inputs)
    output = TemporalAutoencoder(config, name='temporal_autoencoder')(inputs)

    return Model(inputs, [encoder_output, output])
