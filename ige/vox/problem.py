from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds
import gin
import random
from ige.problem import Problem
from ige.vox.metrics import IoU
from shape_tfds.shape.shapenet import core
from shape_tfds.shape.shapenet.core.render import ShapenetCoreRenderConfig
from shape_tfds.shape.shapenet.core.voxel import ShapenetCoreVoxelConfig
from shape_tfds.shape.shapenet.core.frustum_voxel import \
    ShapenetCoreFrustumVoxelConfig


SYNSET_13 = (
    "bench",
    "cabinet",
    "car",
    "chair",
    "lamp",
    "monitor",
    "plane",
    "rifle",
    "sofa",
    "speaker",
    "table",
    "telephone",
    "watercraft",
)

SEEDS_24 = tuple(range(24))


class VoxProblem(Problem):
    def _get_render_builders(self, image_resolution, synset_id, mutators):
        return tuple(
            core.ShapenetCore(config=ShapenetCoreRenderConfig(
            synset_id, image_resolution, mutator)) for mutator in mutators) 

    def _get_voxel_builder(self, voxel_resolution, synset_id, mutators):
        return core.ShapenetCore(
            config=ShapenetCoreVoxelConfig(synset_id, voxel_resolution))
            
    def __init__(
            self,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=(IoU,), synsets=SYNSET_13, num_views=24,
            image_resolution=(128, 128), voxel_resolution=32):
        all_ids, all_names = core.load_synset_ids()
        indices = {id_: i for i, id_ in enumerate(sorted(all_names))}
        synsets = tuple(all_ids.get(synset, synset) for synset in synsets)
        for synset in synsets:
            if synset not in all_names:
                raise ValueError('Invalid synset %s' % synset)

        self._builders = {}
        self._voxel_builders = {}
        for synset in synsets:
            index = indices[synset]
            seeds = (100*index + i for i in range(num_views))
            mutators = tuple(
                core.SceneMutator(name='base%03d' % seed, seed=seed)
                for seed in seeds)
            self._builders[synset] = (
                self._get_render_builders(image_resolution, synset, mutators),
                self._get_voxel_builder(voxel_resolution, synset, mutators))
        self._metrics = tuple(metrics)
        self._loss = loss
        self._num_views = num_views
    
    def _get_zipped_datasets(
            self, synset_id, render_builders, voxel_builder, split):
        vox_dataset = voxel_builder.as_dataset(
            split=split, shuffle_files=False)
        datasets = [tf.data.Dataset.zip(
            (b.as_dataset(split=split, shuffle_files=False), vox_dataset))
            for b in render_builders]
        
        return datasets

    def _get_all_zipped_datasets(self, split):
        import numpy as np
        datasets = []
        for k, v in self._builders.items():
            ds = list(self._get_zipped_datasets(k, *v, split=split))
            if split == tfds.Split.TRAIN:
                random.shuffle(ds)
            datasets.append(ds)
        # order such that we iterate over synsets first, then view index
        datasets = np.array(datasets).T  # out shape (num_views, num_synsets)
        return datasets
    
    def get_dataset(self, split, batch_size=None):
        datasets = self._get_all_zipped_datasets(split)

        def map_fn(rendering_data, voxel_data):
            image = rendering_data['image']
            voxels = voxel_data['voxels']
            voxels = tf.expand_dims(voxels, axis=-1)
            image = tf.image.per_image_standardization(image)
            return image, voxels
        

        
        datasets = tuple(ds.map(map_fn) for ds in datasets)
        dataset = datasets[0]
        for ds in datasets[1:]:
            dataset = dataset.concatenate(ds)
        
        if split == tfds.Split.TRAIN:
            dataset = dataset.shuffle(256).batch(batch_size)
        dataset = dataset.prefetch(
            tf.data.experimental.AUTOTUNE)
        
     

    def examples_per_epoch(self, split=tfds.Split.TRAIN):
        num_examples = 0
        for render_builders, _ in self._builders.items():
            if split == tfds.Split.TRAIN:
                for render_builder in render_builders:
                    num_examples += int(render_builder.info.splits[
                        split].num_examples)
            else:
                nu_examples += int(render_builders[0].info.splits[
                    split].num_examples)
        return num_examples

    
    @property
    def loss(self):
        return self._loss
    
    @property
    def metrics(self):
        return self._metrics


class FrustumVoxProblem(VoxProblem):
        
    def _get_voxel_builder(self, voxel_resolution, synset_id, mutators):
        return tuple(
            core.ShapenetCore(config=ShapenetCoreFrustumVoxelConfig(
            synset_id, voxel_resolution, mutator.copy()))
            for mutator in mutators) 
