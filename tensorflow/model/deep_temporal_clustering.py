from ...config.config import model_config
from ..model import build_temporal_autoencoder
from ..layers.temporal_clustering import TemporalClustering

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.losses import KLDivergence


class DeepTemporalClustering(Model):
    def __init__(self, config=model_config, **kwargs):
        super(DeepTemporalClustering, self).__init__()
        self.temporal_autoencoder = build_temporal_autoencoder(config)
        self.temporal_clustering = TemporalClustering(dist='cid', config=config)
        self.decoder = self.temporal_autoencoder.get_layer('temporal_autoencoder').get_layer('decoder')
        
        self.reconstruction_loss = MeanSquaredError(name='reconstruction_loss')
        self.kl_loss = KLDivergence()
        
    @property
    def metrics(self):
        return [self.reconstruction_loss]

    def train_step(self, inputs):
        # autoencoder
        with tf.GradientTape as tape:
            latent_vector, reconstructed_data = self.temporal_autoencoder(inputs)
            reconstruction_loss = tf.sqrt(tf.reduce_sum(tf.square((input - reconstructed_data))))
            kl_loss = self.kl_loss(input, reconstructed_data)
            loss = reconstruction_loss + kl_loss

        grads = tape.gradient(loss, self.temporal_autoencoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.temporal_autoencoder.trainable_weights))

        # clustering
        siml = self.temporal_clustering(latent_vector)
