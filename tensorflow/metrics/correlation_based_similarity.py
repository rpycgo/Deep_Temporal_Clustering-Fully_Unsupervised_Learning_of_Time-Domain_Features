import tensorflow as tf
from tensorflow.keras.metrics import Metric


class CorrelationBasedSimilarity(Metric):
    def __init__(self, name='correlation_based_similarity', **kwargs):
        super(CorrelationBasedSimilarity, self).__init__(**kwargs)
        self.distance = None
    
    def _error(self, inputs):
        return inputs - tf.math.reduce_mean(inputs, axis=1, keepdims=True)
    
    def _std(self, inputs):
        return tf.math.reduce_std(inputs, axis=2, keepdims=True)
        
    def _normalize(self, inputs):
        return self._error(inputs) / tf.math.reduce_std(inputs, axis=1, keepdims=True)

    def update_state(self, inputs, centroids):
        x = self._error(inputs) # batch_size, time_seq, n_features
        x = x[:, :, :, tf.newaxis]  # batch_size, time_seq, n_features, 1
        centers = self._error(centroids)    # batch_size, time_seq, n_features, k

        p = tf.reduce_mean(x * centers / (self._std(x) * self._std(centers)), axis=1)   # batch_size, n_features, k
        self.distance = tf.reduce_sum(tf.math.sqrt(2 * (1-p)), axis=1) # batch_size, k

    def result(self):
        return self.distance
