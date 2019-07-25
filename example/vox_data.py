from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app


def main(_):
    import numpy as np
    import tensorflow as tf
    from ige.vox.problem import VoxProblem
    import matplotlib.pyplot as plt
    tf.compat.v1.enable_eager_execution()

    # synsets = ('rifle',)
    synsets = ('display', 'suitcase')
    # synsets = ('suitcase',)
    # synsets = ('display',)
    num_views = 2
    problem = VoxProblem(
        use_frustum_voxels=True, num_views=num_views, synsets=synsets)
    print('num examples')
    for split in ('train', 'validation', 'test'):
        print('%s: %d' % (split, problem.examples_per_epoch(split)))

    dataset = problem.get_dataset(split='train')
    for image, voxels in dataset:
        image -= tf.reduce_min(image)
        image /= tf.reduce_max(image)
        image = image.numpy()
        voxels = tf.reduce_any(voxels, axis=-1)
        voxels = tf.image.resize(
            tf.expand_dims(tf.cast(voxels, tf.uint8), axis=-1),
            image.shape[:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        voxels = tf.cast(tf.squeeze(voxels, axis=-1), tf.bool).numpy()
        image[np.logical_not(voxels)] = 0
        plt.imshow(image)
        plt.show()


if __name__ == '__main__':
    app.run(main)
