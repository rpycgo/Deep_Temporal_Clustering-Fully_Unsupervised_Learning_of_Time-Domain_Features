import tensorflow as tf
from tensorflow.keras.metrics import Metric


class ComplexityInvariantSimilarity(Metric):
    def __init__(self, name='complexity_invariant_similarity', **kwargs):
        super(ComplexityInvariantSimilarity, self).__init__()

    def _euclidean_distance(self, inputs):
        error = inputs - tf.reduce_mean(inputs, axis=1, keepdims=True)
        
        return tf.math.reduce_sum(tf.math.square(error), axis=1, keepdims=True)

    def _complexity_estimation(self, inputs):
        inputs_rolled = tf.roll(inputs, shift=1, axis=1)
        diff = (inputs_rolled - inputs)[:, 1:]
        
        return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(diff), axis=1, keepdims=True))
    
    def update_state(self, inputs, centroids):
        ce_x = self._complexity_estimation(inputs)  # batch_size, time_seq, n_features
        ce_x = ce_x[:, :, :, tf.newaxis]    # batch_size, time_seq, n_features, k
        ce_centers = self._complexity_estimation(centroids) # batch_size, time_seq, n_features, k

        # cf
        cf = tf.math.maximum(ce_x, ce_centers) / tf.math.minimum(ce_x, ce_centers)  # batch_size, time_seq, n_features, k
        cf = tf.squeeze(cf, axis=1) # batch_size, n_features, k

        # ed
        ed =  self._euclidean_distance(cf)  # batch_size, n_features, k

        self.distance = tf.reduce_sum(ed*cf, axis=1)    # batch_size, k


    def result(self):
        return self.distance
