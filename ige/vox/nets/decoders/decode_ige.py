from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging
import numpy as np
import tensorflow as tf
import gin
from ige.tf_compat import dim_value
from ige.ops import softabs
from ige.ops import softplus
from ige.unopt import UnrolledSGD


@gin.configurable(blacklist=['features', 'output_spec', 'training'])
def unet_adapter(
        image_features, output_spec, training=None, kernel_regularizer=None,
        initial_filters=256):
    
    def activate(x):
        x = tf.keras.layers.BatchNormalization(scale=False)(
            x, training=training)
        return tf.keras.layers.Activation('relu')(x)
    
    def double_res(x):
        h = x.shape[1]
        w = x.shape[2]
        return tf.image.resize_images(x, (2*h, 2*w), align_corners=True)

    dense_kwargs = dict(kernel_regularizer=kernel_regularizer, use_bias=False)
    
    filters = initial_filters
    x = image_features[0]
    x = tf.keras.layers.Dense(filters, **dense_kwargs)(x)
    x = activate(x)

    out_ims = [x]
    for image_feature in image_features[1:]:
        filters = filters // 2
        image_features = tf.keras.layers.Dense(
            filters, **dense_kwargs)(image_feature)
        x = tf.keras.layers.Dense(filters, **dense_kwargs)(x)
        x = tf.keras.layers.Concatenate(axis=-1)([x, image_feature])
        # upsize
        x = tf.keras.layers.Lambda(double_res)(x)
        x = activate(x)
        out_ims.append(x)
    
    # bring to output resolution
    while x.shape[1] < output_spec.shape[0]:
        filters //= 2
        x = tf.keras.layers.Dense(filters, **dense_kwargs)(x)
        x = tf.keras.layers.Lambda(double_res)(x)
        x = activate(x)
        out_ims.append(x)
    return tuple(out_ims)


@gin.configurable(blacklist=['args', 'output_spec'])
def get_vox_energy(
        args, output_spec, hidden_filters0=256, loss_filters0=128,
        kernel_regularizer=None, hidden_activation=softplus,
        loss_activation=softabs, hidden_kernel_size=3, loss_kernel_size=3):
    logits = args[0]
    image_features = args[1:]
    out_dim = output_spec.shape[0]
    assert(all(o == out_dim for o in output_spec.shape[1:3]))
    i0 = image_features[0]
    d0 = dim_value(i0.shape[1])
    depth_pooling = 1
    spatial_pooling = d0 // i0
    num_pools = int(np.log2(spatial_pooling))
    assert(d0 % i0 == 0)
    assert(2**num_pools == spatial_pooling)
    assert(len(image_features) >= num_pools)
    hidden_filters = hidden_filters0
    loss_filters = loss_filters0
    losses = []
    for image_feature in image_features[:num_pools]:
        pooled = tf.keras.layers.AveragePooling3D(
            (spatial_pooling, spatial_pooling, depth_pooling))(logits)
        pooled = tf.keras.layers.Lambda(tf.squeeze, arguments=dict(axis=-1))(
            pooled)
        x = tf.keras.layers.Concatenate(axis=-1)([image_feature, pooled])
        x = tf.keras.layers.Conv2D(
            hidden_filters, hidden_kernel_size,
            activation=hidden_activation)(x)
        x = tf.keras.layers.Conv2D(
            loss_filters, loss_kernel_size, activation=loss_activation)(x)
        loss = tf.keras.layers.Lambda(tf.reduce_mean)(loss)
        losses.append(loss)
    
        hidden_filters //= 2
        loss_filters //= 2
        spatial_pooling //= 2
        depth_pooling *= 2
    assert(depth_pooling == 2**num_pools)
    assert(spatial_pooling == 1)
    return tf.keras.layers.Add()(loss)


@gin.configurable(blacklist=['features', 'output_spec', 'training'])
def decode_ige(
        features, output_spec, training=None,
        y0_inference_fn=None, y0_model_weights=None,
        feature_adapter=unet_adapter,
        energy_fn=get_vox_energy, inner_opt_fn=UnrolledSGD):
    if y0_inference_fn is None:
        from ige.vox.nets.decoders import decode_conv
        y0_inference_fn = decode_conv
    y0 = y0_inference_fn(features)
    if y0_model_weights is not None:
        # load weights
        y0_model = tf.keras.models.Model(inputs=features, outputs=y0)
        y0_model_weights_path = os.path.expanduser(y0_model_weights_path)
        if os.path.isdir(y0_model_weights_path):
            from ige.callbacks import restore_model
            restore_model(y0_model, y0_model_weights_path)
        else:
            y0_model.load_weights(y0_model_weights_path)
            logging.info('Restored weights from %s' % y0_model_weights_path)
    
    if feature_adapter is not None:
        features = feature_adapter(features, output_spec)
    inner_opt = inner_opt_fn(energy_fn, num_optimized=1)
    final_pred, predictions = inner_opt([y0] + features)
    del final_pred
    predictions = tf.keras.layers.Concatenate(axis=0)([
        tf.keras.layers.Lambda(tf.expand_dims, arguments=dict(axis=0))(y0),
        predictions])
    return predictions
