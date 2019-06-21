from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import tensorflow as tf
import tensorflow_datasets as tfds

import gin

InputSpec = tf.keras.layers.InputSpec


def dataset_spec(dataset, has_batch_dim=False):
    if has_batch_dim:
        def f(shape, dtype):
            return InputSpec(shape=shape[1:], dtype=dtype)
    else:
        def f(shape, dtype):
            return InputSpec(shape=shape, dtype=dtype)
    return tf.nest.map_structure(
        f, dataset.output_shapes, dataset.output_types)


class Problem(object):
    @abc.abstractmethod
    def get_dataset(self, split, batch_size=None):
        raise NotImplementedError

    @abc.abstractmethod
    def examples_per_epoch(self, split=tfds.Split.TRAIN):
        raise NotImplementedError

    @abc.abstractproperty
    def loss(self):
        raise NotImplementedError

    # -------------------------------------------------
    # Base implementations: these make possibly wrong assumptions
    @property
    def metrics(self):
        return None

    def output_spec(self):
        """Assumed to be the same as target_spec, but not necessarily."""
        return self.target_spec()

    # -------------------------------------------------
    # Base implementations: these are possibly inefficient
    def dataset_spec(self):
        return dataset_spec(self.get_dataset('train'))

    def input_spec(self):
        return self.dataset_spec()[0]

    def target_spec(self):
        """`target` means label."""
        return self.dataset_spec()[1]


@gin.configurable
class TfdsProblem(Problem):
    def __init__(
            self, builder, loss, metrics=None, output_spec=None, map_fn=None,
            as_supervised=True, shuffle_buffer=1024,
            download_and_prepare=True):
        if map_fn is not None:
            assert(callable(map_fn) or
                   isinstance(map_fn, dict) and
                   all(v is None or callable(v) for v in map_fn.values()))
        if isinstance(builder, six.string_types):
            builder = tfds.builder(builder)
        self._builder = builder
        self._loss = loss
        self._metrics = metrics
        self._output_spec = output_spec
        self._map_fn = map_fn
        self._as_supervised = as_supervised
        self._download_and_prepare = download_and_prepare
        if shuffle_buffer is None:
            shuffle_buffer = self.examples_per_epoch('train')
        self._shuffle_buffer = shuffle_buffer

    def _supervised_feature(self, index):
        info = self.builder.info
        keys = info.supervised_keys
        if keys is None:
            return None
        else:
            return info.features[keys[index]]

    def output_spec(self):
        if self._output_spec is not None:
            return self._output_spec

        # attempt to handle supervised problems by default
        feature = self._supervised_feature(1)
        num_classes = getattr(feature, 'num_classes', None)
        if num_classes is not None:
            return InputSpec(shape=(num_classes,), dtype=tf.float32)
        return super(TfdsProblem, self).output_spec()

    @property
    def builder(self):
        return self._builder

    def get_dataset(self, split, batch_size=None, prefetch=True):
        if self._download_and_prepare:
            self._builder.download_and_prepare()
        dataset = self.builder.as_dataset(
            batch_size=1, split=self._split(split),
            as_supervised=self._as_supervised, shuffle_files=True)
        return self.data_pipeline(
            dataset, split, batch_size, prefetch=prefetch)

    def examples_per_epoch(self, split='train'):
        return int(self.builder.info.splits[self._split(split)].num_examples)

    @property
    def loss(self):
        return self._loss

    @property
    def metrics(self):
        return list(self._metrics)

    def _split(self, split):
        if (split == 'validation' and
                tfds.Split.VALIDATION not in self.builder.info.splits):
            split = 'test'
        return split

    def data_pipeline(self, dataset, split, batch_size, prefetch=True):
        map_fn = self._map_fn
        if isinstance(map_fn, dict):
            map_fn = map_fn[split]

        if split == tfds.Split.TRAIN:
            dataset = dataset.shuffle(self._shuffle_buffer)

        if map_fn is not None:
            dataset = dataset.map(
                map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if batch_size is not None:
            dataset = self._batch(dataset, batch_size)
        if prefetch:
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def _batch(self, dataset, batch_size):
        return dataset.batch(batch_size)
