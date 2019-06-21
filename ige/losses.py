from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from ige.tf_compat import dim_value


class MultiStepLoss(object):  # Loss class?
    def __init__(self, base_loss, base_ndims, loss_decay=0.9):
        self._base_loss = tf.keras.losses.get(base_loss)
        self._loss_decay = loss_decay
        self._base_ndims = base_ndims
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        if y_pred.shape.ndims == self._base_ndims:
            return self._base_loss(y_true, y_pred, sample_weight)
        shape = y_pred.shape
        num_steps = dim_value(shape[0])
        weights = tf.convert_to_tensor(
            self._loss_decay)**(-tf.range(num_steps, dtype=tf.float32))
        weights = weights / tf.reduce_sum(weights)
        label_shape = [-1 if s is None else s for s in shape[1:].as_list()]
        y_true = tf.reshape(y_true, label_shape)
        losses = tf.map_fn(
            lambda yp: self._base_loss(y_true, yp, sample_weight), y_pred)
        return tf.reduce_sum(losses*weights)
    
    def get_config(self):
        return dict(
            base_loss=self._base_loss.get_config(),
            loss_decay=self._loss_decay,
            base_ndims = self._base_ndims,
        )
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
