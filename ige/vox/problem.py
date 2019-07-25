from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import gin
import random
from ige.problem import Problem
from ige.vox.metrics import IoU
from shape_tfds.shape.shapenet import core


SYNSET_13 = (
    "bench",
    "cabinet",
    "car",
    "chair",
    "lamp",
    "display", # monitor
    "plane",
    "rifle",
    "sofa",
    "speaker",
    "table",
    "telephone",
    "watercraft",
)

SEEDS_24 = tuple(range(24))


def _read_data(data, builder):
    return builder.info.features.decode_example(
        builder._file_format_adapter._parser.parse_example(data))


def _get_tfrecord_paths(builder, split):
    prefix = '%s-%s' % (builder.name.split('/')[0], split)
    data_dir = builder.data_dir
    record_fns = tuple(
        os.path.join(data_dir, fn) for fn in sorted(os.listdir(data_dir))
        if fn.startswith(prefix))
    return record_fns


class VoxProblem(Problem):
    def __init__(
            self,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=(IoU,), synsets=SYNSET_13, num_views=24,
            image_resolution=(128, 128), voxel_resolution=32,
            use_frustum_voxels=False, shuffle_buffer=512):
        all_ids, all_names = core.load_synset_ids()
        synsets = tuple(all_ids.get(synset, synset) for synset in synsets)
        for synset in synsets:
            if synset not in all_names:
                raise ValueError('Invalid synset %s' % synset)

        # image_builders
        self._image_builders = []
        import tensorflow_datasets as tfds
        dl_config = tfds.core.download.DownloadConfig(register_checksums=True)
        # dl_config = None
        for view in range(num_views):
            for synset in synsets:
                builder = core.ShapenetCore(
                    config=core.ShapenetCoreRenderingsConfig(
                        synset, resolution=image_resolution, seed=view))
                builder.download_and_prepare(download_config=dl_config)
                self._image_builders.append(builder)

        # vox_builders
        self._vox_builders = []
        if use_frustum_voxels:
            for view in range(num_views):
                for synset in synsets:
                    builder = core.ShapenetCore(
                        config=core.ShapenetCoreFrustumVoxelConfig(
                            synset, resolution=voxel_resolution, seed=view))
                    builder.download_and_prepare(download_config=dl_config)
                    self._vox_builders.append(builder)
        else:
            for synset in synsets:
                builder = core.ShapenetCore(
                    config=core.ShapenetCoreVoxelConfig(
                        synset, resolution=voxel_resolution,
                        from_file_mapping=True))
                builder.download_and_prepare(download_config=dl_config)
                self._vox_builders.append(builder)
            self._vox_builders = self._vox_builders * num_views

        self._metrics = tuple(metrics)
        self._loss = loss
        self._num_views = num_views
        self._num_synsets = len(synsets)
        self._shuffle_buffer = shuffle_buffer

    def _get_builders(self, split):
        if split == tfds.Split.TRAIN:
            return self._image_builders, self._vox_builders
        else:
            return (
                self._image_builders[:self._num_synsets],
                self._vox_builders[:self._num_synsets])

    def get_dataset(self, split, batch_size=None):
        image_builders, vox_builders = self._get_builders(split)
        image_paths = [_get_tfrecord_paths(b, split) for b in image_builders]
        vox_paths =  [_get_tfrecord_paths(b, split) for b in vox_builders]
        num_paths = len(image_paths)
        assert(len(vox_paths) == num_paths)
        if split == tfds.Split.TRAIN:
            perm = np.random.permutation(len(image_paths))
            image_paths = np.array(image_paths)[perm]
            vox_paths = np.array(vox_paths)[perm]

        image_dataset = tf.data.TFRecordDataset(
            image_paths, num_parallel_reads=num_paths)
        vox_dataset = tf.data.TFRecordDataset(
            vox_paths, num_parallel_reads=num_paths)

        dataset = tf.data.Dataset.zip(
            dict(image=image_dataset, voxels=vox_dataset))
        if split == tfds.Split.TRAIN:
            dataset = dataset.shuffle(self._shuffle_buffer).repeat()

        def map_fn(data):
            image = _read_data(data['image'], image_builders[0])['image']
            voxels = _read_data(data['voxels'], vox_builders[0])['voxels']
            image = tf.cast(image, tf.float32)
            image = tf.image.per_image_standardization(image)
            return image, voxels

        dataset = dataset.map(
            map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if batch_size is not None:
            dataset = dataset.batch(batch_size)
        return dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def examples_per_epoch(self, split=tfds.Split.TRAIN):
        image_builders, _ = self._get_builders(split)
        return sum(
            int(b.info.splits[split].num_examples) for b in image_builders)

    @property
    def loss(self):
        return self._loss

    @property
    def metrics(self):
        return self._metrics
