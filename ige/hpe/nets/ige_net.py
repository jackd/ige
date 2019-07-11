from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import os
import tensorflow as tf
import gin
from ige.ops import softabs
from ige.ops import softplus
from ige.hpe import ops
from ige.hpe.ops import project_points_radial_normalized
from ige.hpe.data.skeleton import l_hip
from ige.hpe.data.skeleton import r_hip
from ige.hpe.data import h3m
from ige.tf_compat import dim_value
from ige.unopt import UnrolledSGD


def _get_feas_input(y):
    # top level fn so can be used in lambda layer and serialized
    scale2 = ops.get_square_skeleton_scale_factor(
        y, h3m.s16.index(l_hip), h3m.s16.index(r_hip))
    return ops.get_pairwise_dist2(y) / scale2


def _project(args):
    # top level fn so can be used in lambda layer and serialized
    pred, rad, tan = args
    return project_points_radial_normalized(
        pred, radial_dist_coeff=rad, tangential_dist_coeff=tan)[0]


@gin.configurable(blacklist=['x'], module='hpe')
def mlp_energy(
        x, hidden_units=(256,), final_units=8,
        hidden_activation=softplus, final_activation=softabs,
        kernel_regularizer=None, initializer_scale=1e-3, name=None):
    prefix = '' if name is None else ('%s-' % name )
    for i, h in enumerate(hidden_units):
        x = tf.keras.layers.Dense(
            h, activation=hidden_activation,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=initializer_scale, mode="fan_avg",
                distribution="truncated_normal"),
            name='%shidden%d' % (prefix, i))(x)
    x = tf.keras.layers.Dense(
        final_units, activation=final_activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=initializer_scale, mode="fan_avg",
            distribution="truncated_normal"),
        name='%sfinal' % prefix)(x)
    return tf.keras.layers.Lambda(
        tf.reduce_sum, arguments=dict(axis=-1), name='%sreduction' % prefix)(x)


@gin.configurable(module='hpe')
def get_combined_energy(
        args, proj_energy_fn=mlp_energy, feas_energy_fn=mlp_energy):
    pred, rad, tan, observation = args
    feas_energy = feas_energy_fn(
        tf.keras.layers.Lambda(_get_feas_input)(pred),
        name='feasibility')
    x_pred = tf.keras.layers.Lambda(_project)([pred, rad, tan])
    x = tf.keras.layers.Concatenate(axis=-2)([x_pred, observation])
    x = tf.keras.layers.Lambda(ops.get_pairwise_dist2)(x)
    proj_energy = proj_energy_fn(x, name='consistency')
    energy = tf.keras.layers.Add()([feas_energy, proj_energy])
    return energy


@gin.configurable(blacklist=['inputs', 'output_spec'], module='hpe')
def get_ige_inference(
        inputs, output_spec, y0_inference_fn=None,
        energy_fn=get_combined_energy, inner_opt_fn=UnrolledSGD,
        y0_model_weights_path=None):
    """
    Get inference via inverse graphics energy minimization.
    
    Args:
        inputs: dict with structure
          {
            'pose_2d': .,
            'intrinsics': {'radial_dist_coeff': ., 'tangential_dist_coeff': .}
          }
        output_spec: tf.keras.layers.LambdaSpec of the target output for a
          single example/step
        y0_inference_fn:  function generating the initial inference based on
            `inputs['p2']` and `output_spec`
        energy_fn: energy function to use. Should map a list of inputs
            ([prediction, radial_dist_coeff, tangential_dist_coeff, pose_2d])
            to a scalar energy value.
        inner_opt_fn: function mapping (energy, num_optimized) to a
            `tf.unopt.UnrolledOptimizer` layer instance.
        y0_model_weights_path: path to weights in `y0_inference_fn`, or
            directory containing multiple weights files loadable via
            `ige.callbacks.restore_model`.
    
    Returns:
        Sequence of pose estimates with shape 
        `[num_steps, batch_size] + output_spec.shape.as_list()`
    """
    p2 = inputs['pose_2d']
    if y0_inference_fn is None:
        from ige.hpe.nets import get_baseline_inference
        y0_inference_fn = get_baseline_inference
    y0 = y0_inference_fn(p2, output_spec)

    if y0_model_weights_path is not None:
        # load weights
        y0_model = tf.keras.models.Model(inputs=p2, outputs=y0)
        y0_model_weights_path = os.path.expanduser(y0_model_weights_path)
        if os.path.isdir(y0_model_weights_path):
            from ige.callbacks import restore_model
            restore_model(y0_model, y0_model_weights_path)
        else:
            y0_model.load_weights(y0_model_weights_path)
            logging.info('Restored weights from %s' % y0_model_weights_path)
    
    inner_opt = inner_opt_fn(energy_fn, num_optimized=1)
    
    intrinsics = inputs['intrinsics']
    final_pred, predictions = inner_opt([
        y0,
        intrinsics['radial_dist_coeff'],
        intrinsics['tangential_dist_coeff'],
        p2])
    del final_pred
    predictions = tf.keras.layers.Concatenate(axis=0)([
        tf.keras.layers.Lambda(tf.expand_dims, arguments=dict(axis=0))(y0),
        predictions])
    return predictions