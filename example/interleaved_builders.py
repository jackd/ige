from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from ige.vox.problem import _get_tfrecord_paths
from ige.vox.problem import _read_data
tf.compat.v1.enable_eager_execution()


class TestConfig(tfds.core.BuilderConfig):
    def __init__(self, value):
        self.value = value
        super(TestConfig, self).__init__(
            name='test_config%d' % value, version='0.0.1')


class TestBuilder(tfds.core.GeneratorBasedBuilder):
    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.core.features.FeaturesDict({
                'value': tfds.core.features.Tensor(
                    dtype=tf.int64, shape=()),
                'index': tfds.core.features.Tensor(
                    dtype=tf.int64, shape=())}
            ))


    def _generate_examples(self):
        for i in range(100):
            # yield i, dict(value=self.builder_config.value, index=i)
            yield i, dict(value=self.builder_config.value, index=i)

    def _split_generators(self, dl_manager):
        return [
            tfds.core.SplitGenerator(name=split, num_shards=10) for split in (
                'train', 'test', 'validation')]


builders = [TestBuilder(config=TestConfig(i)) for i in range(3)]
for b in builders:
    b.download_and_prepare()

split = 'validation'
paths = np.concatenate([_get_tfrecord_paths(b, split) for b in builders])

perm = np.random.permutation(len(paths))
paths = np.array(paths)[perm]
ds0 = tf.data.TFRecordDataset(paths, num_parallel_reads=len(paths))
ds1 = tf.data.TFRecordDataset(paths, num_parallel_reads=len(paths))
ds = tf.data.Dataset.zip((ds0, ds1))
ds = ds.map(lambda *data: tuple(_read_data(d, builders[0]) for d in data))

for examples in ds:
    for example in examples:
        print(example['value'].numpy(), example['index'].numpy())
    print('---')
