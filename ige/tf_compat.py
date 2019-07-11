"""
Various work-arounds for compatibility with tensorflow(_datasets) versioning.


tfds:
Hacky work-around to https://github.com/tensorflow/datasets/issues/580

Resolved in master branch, though required for earlier versions.

Note there's a @memoize on `tfds.core.download.checksums._checksum_paths()`,
so this should be imported before any other `tfds` usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import distutils.version
import tensorflow as tf
import tensorflow_datasets as tfds

# tfds checksums
try:
    import os
    tfds.core.download.checksums._CHECKSUM_DIRS.append(os.path.realpath(
        os.path.join(os.path.dirname(__file__), 'url_checksums')))
    tfds.core.download.checksums._checksum_paths.cache_clear()
except AttributeError:
    # later versions of tfds don't have tfds.core.download.checksums
    # bug seems fixed in these?
    pass


# clean up namespace
del tfds
del os

tf_version = distutils.version.LooseVersion(tf.__version__)
is_v1 = tf_version.version[0] == 1
is_v1_13 = is_v1 and tf_version.version[1] == 13

if is_v1_13:
    from tensorflow.python.keras import losses  # pylint: disable=no-name-in-module
    from tensorflow.python.keras import metrics # pylint: disable=no-name-in-module
    tf.keras.metrics.Metric = metrics.Metric

    class LossFunctionWrapper(object):
        def __init__(
                self,
                fn,
                reduction='sum_over_batch_size',
                name=None,
                **kwargs):
            if reduction not in ('sum', 'sum_over_batch_size'):
                raise NotImplementedError('In v1')
            self.fn = fn
            self._fn_kwargs = kwargs
            self.reduction = reduction
            self.name = name

        def __call__(self, y_true, y_pred, sample_weight=None):
            loss = self.call(y_true, y_pred)
            if loss.shape.ndims != 1:
                raise NotImplementedError('In v1')
            if sample_weight is not None:
                loss = loss * sample_weight
            if self.reduction == 'sum_over_batch_size':
                return tf.reduce_sum(loss) / tf.cast(
                    tf.shape(y_true)[0], loss.dtype)
            elif self.reduction == 'sum':
                return tf.reduce_sum(loss)
            else:
                raise NotImplementedError()

        def call(self, y_true, y_pred):
            return self.fn(y_true, y_pred, **self._fn_kwargs)

        def get_config(self):
            config = {}
            for k, v in six.iteritems(self._fn_kwargs):
                try:
                    v = tf.keras.backend.eval(v)
                except Exception:
                    pass
                config[k] = v
            config['reduction'] = self.reduction
            config['name'] = self.name
            config['fn'] = self.fn
            return config

        @classmethod
        def from_config(self, config):
            return LossFunctionWrapper(**config)

    losses.LossFunctionWrapper = LossFunctionWrapper


def dim_value(dimension):
    return getattr(dimension, 'value', dimension)
