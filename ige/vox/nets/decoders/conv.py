from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin


def _add(args):
    """Add tensors using `+` which allows array broadcasting."""
    # UPDATE IN 2.0: no need for `Lambda`s so move inline
    assert(isinstance(args, list))
    x = args[0]
    for a in args[1:]:
        x = x + a
    return x


@gin.configurable(
    blacklist=['image_features', 'training', 'kernel_regularizer',
               'resolution'])
def broadcast_initial_block(
        image_features, training=None, kernel_regularizer=None,
        resolution=(4, 4, 4), filters=128):
    """
    Go from flat embedding to initial voxel grid without too many params.
    
    pool -> dense -> reshape -> split -> reshapes -> add
    """
    embedding = tf.keras.layers.AveragePooling2D(image_features[0])
    x = tf.keras.layers.Dense(
        filters*sum(resolution),
        kernel_regularizer=kernel_regularizer)(embedding)
    x, y, z = tf.keras.layers.Lambda(
        tf.split,
        arguments=dict(
            num_or_size_splits=[filters*r for r in resolution],
            axis=-1))(tf.split)(x)
    x = tf.keras.layers.Reshape((-1, resolution[0], 1, 1, filters))(x)
    y = tf.keras.layers.Reshape((-1, 1, resolution[1], 1, filters))(y)
    z = tf.keras.layers.Reshape((-1, 1, 1, resolution[2], filters))(z)
    return tf.keras.layers.Lambda(_add)([x, y, z])


@gin.configurable(
    blacklist=['image_features', 'training', 'kernel_regularizer',
               'resolution'])
def reshape_initial_block(
        image_features, training=None, kernel_regularizer=None,
        resolution=(4, 4, 4), filters=128):
    """
    Convert rx * ry image features to rx * ry * rz voxel features.
    
    dense -> reshape
    """
    x = image_features[0]
    assert(x.shape[1:3].as_list() == list(resolution[:2]))
    filters = filters * resolution[2]
    x = tf.keras.layers.Dense(
        filters, kernel_regularizer=kernel_regularizer)(x)
    x = tf.keras.layers.Reshape(resolution + (filters,))
    return x


@gin.configurable(blacklist=['image_features', 'output_spec', 'training'])
def decode_conv(
        features, output_spec, training=None,
        initial_block=broadcast_initial_block,
        kernel_regularizer=None):
    x = initial_block(
        features, training=training, kernel_regularizer=kernel_regularizer)

    shape = x.shape[1:-1].as_list()
    out_shape = output_spec.shape[:-1].as_list()
    assert(all(o % s == 0 for o, s in zip(out_shape, shape)))
    num_convs = [o // s for o, s in zip(out_shape, shape)]
    if not all(n == num_convs[0] for n in num_convs[1:]):
        raise RuntimeError(
            'Inconsistent shape. Out shape is %s, but initial block shape is '
            '%s' % (out_shape, shape))
    num_convs = num_convs[0]
    filters = x.shape[-1]

    def activate(x):
        x = tf.keras.layers.BatchNormalization(scale=False)(
            x, training=training)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    for _ in num_convs:
        filters = filters // 2
        x = activate(x)
        x = tf.keras.layers.Conv3DTranspose(
            filters, 4, 2, kernel_regularizer=kernel_regularizer,
            use_bias=False)(x)
    
    x = activate(x)
    x = tf.keras.layers.Dense(1, kernel_reguarlizer=kernel_regularizer)(x)
    return x