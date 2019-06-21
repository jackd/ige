from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class FinalStepMetric(tf.keras.metrics.Metric):  # Metrics?
    def __init__(self, base_metric, base_ndims, name=None):
        self._base_metric = tf.keras.metrics.get(base_metric)
        if name is None:
            name = 'final_%s' % base_metric.name
        self._base_ndims = base_ndims
        super(FinalStepMetric, self).__init__(name=name)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if y_pred.shape.ndims == self._base_ndims:
            return self._base_metric.update_state(
                y_true, y_pred, sample_weight)
        y_pred = y_pred[-1]
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        # if sample_weight is not None:
        #     sample_weight = tf.reshape(sample_weight, (-1,))
        return self._base_metric.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        return self._base_metric.result()
    
    def reset_states(self):
        return self._base_metric.reset_states()
    
    def get_config(self):
        return dict(base_metric=self._base_metric.get_config())

    
