from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
import gin


@gin.configurable
def mobilenet_features(inputs, output_spec, training=None, alpha=1.0):
    model = tf.keras.applications.MobileNetV2(
        include_top=False, input_tensor=inputs, alpha=alpha)
    for layer in model.layers:
        print(layer.output_shape, layer.name)
    raise NotImplementedError
    image_features = [inputs] + [layer.output for layer in model.layers]
    image_features = [
        f for f in image_features if f.shape[1] <= output_spec.shape[0]]
    return image_features
