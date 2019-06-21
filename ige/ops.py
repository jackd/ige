from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin

softplus = tf.keras.activations.softplus


@gin.configurable(blacklist=['x'])
def softabs(x):
    with tf.name_scope('softabs'):
        return (softplus(x) + softplus(-x)) / 2
