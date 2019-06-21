"""Train/validation/predict loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from ige import callbacks as cb
import gin
from absl import logging
from ige.tf_compat import is_v1_13


def _model(inputs, outputs):
    inputs = tf.nest.flatten(inputs)
    if len(inputs) == 1:
        inputs, = inputs
    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def batch_steps(num_examples, batch_size):
    """Get the number of batches, including possible fractional last."""
    steps = num_examples // batch_size
    if num_examples % batch_size > 0:
        steps += 1
    return steps


@gin.configurable
def default_model_dir(base_dir='~/ige_models', subdir='hpe', model_id=None):
    """
    Get a new directory at `base_dir/model_id`.

    If model_id is None, we use 'model%03d', counting up from 0 until we find
    a space, i.e. model000, model001, model002 ...
    """
    base_dir = os.path.join(base_dir, subdir)
    if model_id is None:
        i = 0
        model_dir = os.path.join(base_dir, 'model%03d' % i)
        while os.path.isdir(model_dir):
            i += 1
            model_dir = os.path.join(base_dir, 'model%03d' % i)
    else:
        model_dir = os.path.join(base_dir, model_id)
    return model_dir


@gin.configurable
def get_optimizer(factory=gin.REQUIRED, learning_rate=1e-3):
    return factory(lr=learning_rate)


@gin.configurable(module='main')
def train(
        problem, batch_size, epochs, inference_fn, optimizer, model_dir=None,
        callbacks=None, verbose=True, checkpoint_freq=1,
        validation_freq=1, summary_freq=10, lr_schedule=None,
        tensorboard_log_dir=None, save_operative_config=True):
    """
    Train a model on the given problem

    Args:
        problem: `ml_glaucoma.problems.Problem` instance
        batch_size: int, size of each batch for training/evaluation
        inference_fn: function mapping
            (inputs, output_spec, training=None) -> inference
        optimizer: `tf.keras.optimizers.Optimizer` instance.
        model_dir: directory in which to save models. If not provided,
            `ml_glaucoma.runners.default_model_dir()` is used.
        callbacks: list of callbacks in addition to those created below
        verbose: passed to `tf.keras.models.Model.fit`.
        checkpoint_freq: frequency in epochs at which to save weights.
        summary_freq: frequency in batches at which to save tensorboard
            summaries.
        lr_schedule: function mapping `epoch -> learning_rate`.
        tensorboard_log_dir: directory to log tensorboard summaries. If not
            provided, `model_dir` is used.

    Returns:
        `tf.keras` `History` object as returned by `model.fit`
    """
    if model_dir is None:
        model_dir = default_model_dir()
    if model_dir is not None:
        model_dir = os.path.expandvars(os.path.expanduser(model_dir))
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    operative_config = gin.operative_config_str()
    if save_operative_config:
        with open(os.path.join(model_dir, 'operative_config.gin'), 'w') as fp:
            fp.write(operative_config)

    train_ds, val_ds = tf.nest.map_structure(
        lambda split: problem.get_dataset(split, batch_size).repeat(),
        ('train', 'validation'))
    inputs = tf.nest.map_structure(
        lambda shape, dtype: tf.keras.layers.Input(
            shape=shape[1:], dtype=dtype),
        train_ds.output_shapes[0], train_ds.output_types[0])
    inference = inference_fn(inputs, problem.output_spec())
    model = _model(inputs, inference)

    model.compile(
        optimizer=optimizer,
        loss=problem.loss,
        metrics=problem.metrics)
    train_steps = batch_steps(
        problem.examples_per_epoch('train'), batch_size)
    validation_steps = batch_steps(
        problem.examples_per_epoch('validation'), batch_size)

    common_callbacks, initial_epoch = cb.get_callbacks(
        model,
        batch_size=batch_size,
        checkpoint_freq=checkpoint_freq,
        summary_freq=summary_freq,
        model_dir=model_dir,
        train_steps_per_epoch=train_steps,
        val_steps_per_epoch=validation_steps,
        lr_schedule=lr_schedule,
        tensorboard_log_dir=tensorboard_log_dir,
    )
    logging.info(
        'Beginning training.\nInitial epoch: %d\nOperative config:\n%s'
        % (initial_epoch, operative_config))
    if callbacks is None:
        callbacks = common_callbacks
    else:
        callbacks.extend(common_callbacks)

    def map_fn(inputs, labels):
        inputs = tuple(tf.nest.flatten(inputs))
        if len(inputs) == 1:
            inputs = inputs[0]
        return inputs, labels

    train_ds, val_ds = (ds.map(map_fn) for ds in (train_ds, val_ds))

    if is_v1_13:
        kwargs = {}
    else:
        kwargs = dict(validation_freq=validation_freq)  # not supported in 1.13

    return model.fit(
        train_ds,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=val_ds,
        steps_per_epoch=train_steps,
        validation_steps=validation_steps,
        initial_epoch=initial_epoch,
        **kwargs)


@gin.configurable(module='main')
def evaluate(problem, batch_size, inference_fn, optimizer, model_dir=None):
    """
    Evaluate the given model with weights saved as `model_dir`.

    Args:
        problems: `ml_glaucoma.problems.Problem` instance
        batch_size: size of each batch
        inference_fn: inference_fn: function mapping
            (inputs, output_spec, training) -> outputs
        optimizer: `tf.keras.optimizers.Optimizer` instance
        model_dir: string path to directory containing weight files.

    Returns:
        scalar or list of scalars - loss/metrics values
        (output of `tf.keras.models.Model.evaluate`)
    """
    logging.info(
        'Evaluating.\nOperative config:\n%s' % gin.operative_config_str())
    if model_dir is None:
        model_dir = default_model_dir()
    if model_dir is not None:
        model_dir = os.path.expanduser(model_dir)
    if not os.path.isdir(model_dir):
        raise RuntimeError('model_dir does not exist: %s' % model_dir)

    val_ds = problem.get_dataset('validation', batch_size)
    inputs = tf.nest.map_structure(
        lambda shape, dtype: tf.keras.layers.Input(
            shape=shape[1:], dtype=dtype),
        val_ds.output_shapes[0], val_ds.output_types[0])
    outputs = inference_fn(inputs, problem.output_spec())
    model = _model(inputs, outputs)
    model.compile(
        optimizer=optimizer,
        loss=problem.loss,
        metrics=problem.metrics)

    cb.restore_model(model, model_dir)
    validation_steps = batch_steps(
        problem.examples_per_epoch('validation'), batch_size)
    
    val_ds = val_ds.map(lambda x, y: (tuple(tf.nest.flatten(x)), y))

    logging.info('Running evaluation for %d steps' % validation_steps)
    return model.evaluate(val_ds, steps=validation_steps)


@gin.configurable(module='main')
def vis(problem, inference_fn=None, model_dir=None, split='train', batch_size=1):
    """Simple visualization of the given `Problem`."""
    if model_dir is None:
        model_dir = default_model_dir()
    if model_dir is not None:
        model_dir = os.path.expandvars(os.path.expanduser(model_dir))
    with tf.Graph().as_default():  # pylint:disable=not-context-manager
        dataset = problem.get_dataset(split=split, batch_size=batch_size)
        inputs, labels = tf.compat.v1.data.make_one_shot_iterator(
            dataset).get_next()
        if inference_fn is None:
            predictions = None
        else:
            outputs = inference_fn(
                tf.nest.map_structure(
                    lambda x: tf.keras.layers.Input(tensor=x), inputs),
                problem.output_spec())
            model = tf.keras.models.Model(tf.nest.flatten(inputs), outputs)
            predictions = model.outputs

        vis_kwargs = problem.previs(inputs, labels, predictions)
        with tf.compat.v1.Session() as sess:
            if inference_fn is not None:
                cb.restore_model(model, model_dir)
            try:
                while True:
                    batched_kwargs = sess.run(vis_kwargs)
                    for i in range(batch_size):
                        kwargs = tf.nest.map_structure(
                            lambda x: x[i], batched_kwargs)
                        problem.vis(**kwargs)
            except tf.errors.OutOfRangeError:
                pass


@gin.configurable(module='main')
def print_config(problem, batch_size, inference_fn, optimizer, model_dir=None):
    print(gin.operative_config_str())
