from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

from ige.hpe.models.baseline import get_baseline_inference
from ige.hpe.problem import IgeProblem
from ige.hpe.losses import pose_loss, Alignment
# from sp2.data_source import PoseDataSource


ModeKeys = tf.estimator.ModeKeys


def model_fn(features, labels, mode):
    features = tf.nest.map_structure(
        lambda x: tf.keras.layers.Input(tensor=x), features)
    # features = features['p2']
    # labels = labels['p3c']
    inference = get_baseline_inference(
        features, problem.output_spec(), training=mode == ModeKeys.TRAIN,
        max_norm_first=False,
        max_norm_mid=False,
        max_norm_last=False,)
    model = tf.keras.models.Model(
        inputs=tf.nest.flatten(features), outputs=inference)
    loss = problem.loss(labels, inference)
    if model.losses:
        loss = tf.add_n(model.losses + [loss])
        
    metric_loss = tf.reduce_mean(pose_loss(
        labels, inference, order=1, alignment=Alignment.OPT_SCALE))
    tf.summary.scalar('l1_denormalized', metric_loss)
    eval_metric_ops = {
        'l1_loss': tf.metrics.mean(metric_loss)
    }
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
            float(initial_learning_rate), global_step, decay_steps, decay_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    updates = model.updates
    with tf.control_dependencies(updates):
        train_op = optimizer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(
        mode, inference, loss, train_op, eval_metric_ops)


problem = IgeProblem()

initial_learning_rate = 1e-3
decay_steps=100000
decay_rate=0.96
batch_size = 64
max_steps = 200*problem.examples_per_epoch('train') // batch_size
model_dir = os.path.expanduser('~/ige_models/estimator_test')

run_config = tf.estimator.RunConfig(
        keep_checkpoint_max=6,
        save_checkpoints_steps=20000,
        save_summary_steps=2000,
        log_step_count_steps=2000)

tf.logging.set_verbosity(tf.logging.INFO)
estimator = tf.estimator.Estimator(
    model_fn, model_dir, config=run_config)
# data_source = PoseDataSource(origin_change='scale')
train_spec = tf.estimator.TrainSpec(
    # lambda: data_source.get_inputs(ModeKeys.TRAIN, batch_size=batch_size),
    lambda: problem.get_dataset(
        split='train', batch_size=batch_size).repeat(),
    max_steps=max_steps)
eval_spec = tf.estimator.EvalSpec(
    # lambda: data_source.get_inputs(ModeKeys.EVAL, batch_size=batch_size))
    lambda: problem.get_dataset(
        split='validation', batch_size=batch_size).repeat())
tf.estimator.train_and_evaluate(
    estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)

