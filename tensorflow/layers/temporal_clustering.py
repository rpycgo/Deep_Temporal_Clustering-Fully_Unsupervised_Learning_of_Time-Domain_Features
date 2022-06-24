from sklearn.metrics import euclidean_distances
from ...config.config import model_config
from ..metrics.complexity_invariant_similarity import ComplexityInvariantSimilarity
from ..metrics.correlation_based_similarity import CorrelationBasedSimilarity
from ..metrics.euclidean_distance import EuclideanDistance

import tensorflow as tf
from tensorflow.keras.layers import Layer


class TemporalClustering(Layer):
    def __init__(self, dist='cid', config=model_config, **kwargs):
        super(TemporalClustering, self).__init__(**kwargs)        
        self.config = model_config
        self.dist = dist.lower()

    def build(self, input_shape):        
        self.centroids = tf.random.normal(shape=(self.config.batch_size, ) + input_shape[1:] + (self.config.k, ))

    def _get_distance(self, inputs):
        if self.dist == 'cid':
            self.cid = ComplexityInvariantSimilarity()
            self.cid.update_state(inputs, self.centroids)
        elif self.dist == 'cor':
            self.cor = CorrelationBasedSimilarity()
            self.cor.update_state(inputs, self.centroids)
        elif self.dist == 'acf':
            raise NotImplementedError
        elif self.dist == 'eucl':
            self.eucl = EuclideanDistance()
            self.cor.update_state(inputs, self.centroids)
        else:
            raise ValueError('metrics are among cid, cor, acf and eucl')
        
        return self.__dict__.get(self.dist).distance

    
    def call(self, inputs, training=None):
        distance = self._get_distance(inputs)

        return distance
