import tensorflow as tf
from tensorflow.keras.metrics import Metric


class EuclideanDistance(Metric):
    def __init__(self, name='euclidean_distannce', **kwargs):
        super(EuclideanDistance, self).__init__(**kwargs)
        self.distance = None

    def update_state(self, inputs, centroids):
        x = x[:, :, :, tf.newaxis]  # batch_size, time_seq, n_features, 1
        self.distance = tf.math.reduce_sum(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(x - centroids), axis=1)), axis=1)    # batch_size, time_seq, n_features, k -> # batch_size, k

    def result(self):
        return self.distance
