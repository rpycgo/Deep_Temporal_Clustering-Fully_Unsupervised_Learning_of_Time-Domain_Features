import tensorflow as tf
from tensorflow.keras.metrics import Metric


class AotoCorrelationBasedSimilarity(Metric):
    def __init__(self, name='auto_correlation_based_similarity', **kwargs):
        super(AotoCorrelationBasedSimilarity, self).__init__(**kwargs)
        self.distance = None
    
    def update_state(self, inputs, centroids):
        pass

    def result(self):
        return self.distance
