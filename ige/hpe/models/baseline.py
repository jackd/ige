from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin
from ige.tf_compat import dim_value
from ige.layers import InputSink


@gin.configurable(blacklist=['inputs', 'output_spec'], module='hpe')
def get_baseline_inference(
        inputs, output_spec, training=None,
        filters=1024,
        num_blocks=2,
        dropout_rate=0.5,
        kernel_regularizer=None,
        residual=True,
        kernel_initializer='he_normal',
        bias_initializer='zeros',
        max_norm_first=False,
        max_norm_mid=False,
        max_norm_last=False,
        use_batch_norm=True,
        use_biases=None,
        scale_batch_norm=True,
        freeze_batch_norm=False):
    """
    Get network output based on https://arxiv.org/pdf/1705.03098.pdf .

    Note: Kaiming initialization is the same as He initialization
    (lead author's name is Kaiming He - https://arxiv.org/pdf/1502.01852.pdf).

    Args:
        inputs: 2d pose, or dict with 'pose_2d' as an entry.
        output_spec: `tf.keras.layers.InputSpec` of the desired output.
        training: flag indicating training for batch norm/dropout. If None,
            keras default is used.
        filters: number of filters in hidden layers.
        num_blocks: number of residual blocks.
        dropout_rate: rate used in `tf.keras.layers.Dropout`.
        kernel_regularizer: regularizer applied to each dense kernel.
        residual: flag indicating skip connections should be used.
        kernel_initializer: passed to each `tf.keras.layers.Dense` constructor.
        bias_initializer: passed to each `tf.keras.layers.Dense` constructor.
        max_norm_first: if True, first kernel is max-normed.
        max_norm_mid: if True, hidden layer kernels are max-normed.
        max_norm_last: if true, final layer kernel is max-normed.
        use_batch_norm: whether or not to use batch normalization.
        use_biases: whether or not to use biases. If None, we use them only
            when batch noralization is off.
        scale_batch_norm: indicates scaling should be used in batch norm.
        freeze_batch_norm: indicates with batch statistics should be used even
            during training.
    
    Returns:
        Tensor with spec given by `output_spec` with an additional leading
        batch dimension.    
    """
    if isinstance(inputs, dict):
        # keras requires all inputs to be used, so we `InputSink` the rest.
        inputs = inputs.copy()
        p2 = inputs.pop('pose_2d')
        if inputs:
            args = [p2] + tf.nest.flatten(inputs)
            p2, = InputSink(1)(args)
    else:
        p2 = inputs
    assert(p2.shape[1] == 16)
    dense_kwargs = dict(
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer)

    bn_training = False if freeze_batch_norm else training
    if use_biases is None:
        use_biases = not use_batch_norm

    max_norm_constraint = tf.keras.constraints.MaxNorm(1.0, axis=None)

    def activate(x):
        if use_batch_norm:
            layer = tf.keras.layers.BatchNormalization(
                scale=scale_batch_norm)
            x = layer(x, training=bn_training)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x, training=training)
        return x

    x = tf.keras.layers.Flatten()(p2)
    # max_norm might have an affect here?
    x = tf.keras.layers.Dense(
        filters, use_bias=use_biases, name='initial_dense',
        kernel_constraint=max_norm_constraint if max_norm_first else None,
        **dense_kwargs)(x)
    x = activate(x)
    mid_kernel_constraint = max_norm_constraint if max_norm_mid else None
    for i in range(num_blocks):
        x0 = x
        with tf.name_scope('block%d' % i):
            for j in range(2):
                # max_norming these cancels with batch norm scaling
                # from original? Maybe I'm missing something?
                # Did the authors know?
                x = tf.keras.layers.Dense(
                    filters, use_bias=use_biases, name='dense%d-%d' % (i, j),
                    kernel_constraint=mid_kernel_constraint,
                    **dense_kwargs)(x)
                x = activate(x)
        if residual:
            x = tf.keras.layers.Add()([x, x0])
    
    # No batch norm after this to negate max_norm affect
    n, d = (dim_value(s) for s in output_spec.shape[-2:])
    x = tf.keras.layers.Dense(
        n*d, use_bias=True, name='output_dense',
        kernel_constraint=max_norm_constraint if max_norm_last else None,
        **dense_kwargs)(x)
    p3c = tf.keras.layers.Reshape((n, d))(x)

    return p3c