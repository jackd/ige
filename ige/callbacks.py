from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
import os
from ige.tf_compat import is_v1


class LoadingModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """
    ModelCheckpoint modified to automatically restore model.

    Weight restoration can be done manually using `self.restore`.

    Restoration only happens once by default, but you can force subsequent
    restorations using `self.restore(force_restore=True)`.
    """
    def __init__(self, model_dir, period=1, **kwargs):
        """
        Args:
            model_dir: directory to save weights. Files will have format
                '{model_dir}/{epoch:04d}.h5'.
            **kwargs: passed to `ModelCheckpoint.__init__`.
                All keys valid except `filepath`.
        """
        self._model_dir = model_dir
        self._filename = 'model-{epoch:04d}.h5'
        super(LoadingModelCheckpoint, self).__init__(
            filepath=os.path.join(self._model_dir, self._filename),
            period=period, **kwargs)
        self._restored = False

    def restore(self, save_path=None, force_restore=False):
        """
        Restore weights at path, or latest in model directory.

        Does nothing if the model has already been restored by this loader
        unless `force_restore` is True.
        """
        if not self._restored or force_restore:
            if save_path is None:
                save_path = self.latest_checkpoint
            if save_path is not None:
                self.model.load_weights(save_path)
            self._restored = True

    @property
    def latest_checkpoint(self):
        """Get the full path to the latest weights file."""
        filenames = tuple(fn for fn in os.listdir(self._model_dir)
                          if fn.startswith('model'))
        if len(filenames) == 0:
            return None
        latest = max(filenames, key=self.filename_epoch)
        return os.path.join(self._model_dir, latest)

    def filename_epoch(self, filename):
        """Get the epoch of the given file/path."""
        assert(filename.endswith('.h5'))
        return int(filename[-7:-3])

    def on_train_begin(self, logs=None):
        self._on_begin()

    def on_test_begin(self, logs=None):
        self._on_begin()

    def on_predict_begin(self, logs=None):
        self._on_begin()

    def _on_begin(self):
        self.restore()


class CustomTensorBoardV1(tf.keras.callbacks.TensorBoard):
    def __init__(self, custom_summary, *args, **kwargs):
        self.__custom_summary = custom_summary
        self.__last_write = 0
        super(CustomTensorBoardV1, self).__init__(*args, **kwargs)

    def __write_custom_summary(self, summary_val):
        if self._total_batches_seen - self.__last_write >= self.update_freq:
            self.writer.add_summary(summary_val, self._total_batches_seen)
            self.__last_write = self._total_batches_seen

    def on_train_begin(self, logs=None):
        self.model._fit_function.fetches.append(self.__custom_summary)
        self.model._fit_function.fetch_callbacks[
            self.__custom_summary] = self.__write_custom_summary
        super(CustomTensorBoardV1, self).on_train_begin(logs)

    def on_train_end(self, logs=None):
        self.model._fit_function.fetches.remove(self.__custom_summary)
        self.model._fit_function.fetch_callbacks.pop(self.__custom_summary)
        super(CustomTensorBoardV1, self).on_train_end(logs)


CustomTensorBoardV2 = tf.keras.callbacks.TensorBoard


@gin.configurable
def exponential_decay_lr_schedule(lr0, factor):
    """
    lambda epoch: lr0 * (factor ** epoch)

    The returned callback can be used in
    `tf.keras.callbacks.LearningRateScheduler`.
    """
    return lambda epoch: lr0 * (factor ** epoch)


def get_callbacks(
        model,
        batch_size,
        checkpoint_freq=5,
        summary_freq=10,
        model_dir=None,
        train_steps_per_epoch=None,
        val_steps_per_epoch=None,
        lr_schedule=None,
        tensorboard_log_dir=None,
        write_images=False,
        ):
    """
    Get common callbacks used in training.

    Args:
        model: `tf.keras.models.Model` to be used.
        batch_size: size of each batch - used to correct tensorboard initial
            step.
        checkpoint_freq: if not None, adds a `LoadingModelCheckpoint` which
            extends `ModelCheckpoint` to restore weights on fit/evaluate start
            and saves at this epoch frequency.
        summary_freq: if given, adds a `TensorBoard` callback that logs at this
            batch frequency.
        model_dir: directory in which to save weights
        train_steps_per_epoch: number of training steps per epoch. Necessary
            for initializing `TensorBoard` correctly when resuming training.
        val_steps_per_epoch: number of validation steps per epoch.
        lr_schedule: if provided, adds a `LearningRateScheduler` with this
            schedule.
        tensorboard_log_dir: if given, logs are written here. It not,
            `model_dir` is used
        write_images: passed to `TensorBoard`

    Returns:
        (callbacks, initial_epoch), where callbacks is a list of
        `tf.keras.callbacks.Callback` and `initial_epoch` corresponds to the
        epoch count of the weights loaded.
    """
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]

    initial_epoch = 0
    if checkpoint_freq is not None:
        saver_callback = LoadingModelCheckpoint(
            model_dir, period=checkpoint_freq)
        latest_checkpoint = saver_callback.latest_checkpoint
        if latest_checkpoint is not None:
            initial_epoch = saver_callback.filename_epoch(latest_checkpoint)
        callbacks.append(saver_callback)

    if summary_freq:
        tb_callback = None
        kwargs = dict(
            write_graph=False,
            log_dir=tensorboard_log_dir or model_dir,
            update_freq=summary_freq,
            write_images=write_images)
        if is_v1:
            custom_summary = tf.summary.merge_all()
            if custom_summary is None:
                tb_callback = tf.keras.callbacks.TensorBoard(**kwargs)
            else:
                tb_callback = CustomTensorBoardV1(
                    custom_summary,
                    **kwargs)
        else:
            # raise NotImplementedError('TODO')
            tb_callback = CustomTensorBoardV2(**kwargs)

        # These hacks involve private members - will probably break
        if train_steps_per_epoch is not None and initial_epoch > 0:
            initial_train_steps = \
                initial_epoch*train_steps_per_epoch
            tb_callback._total_batches_seen = initial_train_steps
            # v1 a sample is a batch, where as in v2 a sample is an element
            if is_v1:
                tb_callback._samples_seen = initial_train_steps
            else:
                tb_callback._samples_seen = initial_train_steps*batch_size
        if val_steps_per_epoch is not None and initial_epoch > 0:
            initial_val_steps = initial_epoch*val_steps_per_epoch
            tb_callback._total_val_batches_seen = initial_val_steps

        callbacks.append(tb_callback)

    if lr_schedule is not None:
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule))

    return callbacks, initial_epoch


def restore_model(model, model_dir):
    from absl import logging
    manager_cb = LoadingModelCheckpoint(model_dir=model_dir)
    manager_cb.set_model(model)
    manager_cb.restore()
    logging.info('Restored checkpoint from %s' % manager_cb.latest_checkpoint)
