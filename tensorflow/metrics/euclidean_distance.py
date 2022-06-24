import tensorflow as tf
from tensorflow.keras.metrics import Metric


class EuclideanDistance(Metric):
    def __init__(self, name='euclidean_distannce', **kwargs):
        super(EuclideanDistance, self).__init__(**kwargs)
        self.distance = None

    def update_state(self, inputs, centroids):
        x = tf.expand_dims(inputs, axis=1)  # batch_size, 1, time_seq, n_features
        self.distance = tf.math.reduce_sum(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(x - centroids), axis=2)), axis=-1)    # batch_size, k, time_seq, n_features

    def result(self):
        return self.distance
