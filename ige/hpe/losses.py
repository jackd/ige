from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ige.hpe import ops
from ige.tf_compat import dim_value
from ige.hpe.data import h3m
import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper # pylint: disable=no-name-in-module
import gin


class Alignment(object):
    PROCRUSTES = 'procrustes'
    OPT_SCALE = 'opt_scale'
    ORIGIN = 'origin'
    NONE = None

    @classmethod
    def as_tuple(cls):
        return (
            Alignment.PROCRUSTES,
            Alignment.OPT_SCALE,
            Alignment.ORIGIN,
            Alignment.NONE,
        )


def align(y_true, y_pred, alignment=Alignment.NONE, origin_index=None):
    return {
        Alignment.NONE: lambda y_true, y_pred: y_pred,
        Alignment.OPT_SCALE: ops.opt_scale_aligned,
        Alignment.ORIGIN: lambda y_true, y_pred: ops.origin_aligned(
            y_true, y_pred, origin_index),
        Alignment.PROCRUSTES: ops.procrustes_aligned,
    }[alignment](y_true, y_pred)


def pose_loss(
        y_true, y_pred, sample_weight=None, alignment=Alignment.NONE, order=2,
        eps=None, origin_index=None, add_origin=False, denormalize_stats=None,
        joint_reduction='mean', rescale_labels=False):
    if denormalize_stats is not None:
        y_true = ops.denormalized(y_true, **denormalize_stats)
        y_pred = ops.denormalized(y_pred, **denormalize_stats)

    if add_origin:
        assert(origin_index is not None)
        y_true = ops.append_midpoint(y_true, origin_index)
        y_pred = ops.append_midpoint(y_pred, origin_index)
    if rescale_labels:
        midpoint = ops.get_midpoint(y_true, origin_index, keepdims=True)
        y_true = y_true / tf.linalg.norm(midpoint, axis=-1, keepdims=True)
    if alignment is not None:
        y_pred = align(
            y_true, y_pred, alignment=alignment, origin_index=origin_index)
    err = ops.get_error(
        y_true, y_pred, order=order, eps=eps, joint_reduction=joint_reduction)
    if sample_weight is not None:
        err = err * sample_weight
    return err


@gin.configurable
class PoseLoss(LossFunctionWrapper):
    def __init__(
            self, reduction='sum_over_batch_size',
            alignment=Alignment.NONE, order=2, eps=None, add_origin=False,
            origin_index=None, denormalize_stats=None, name=None,
            joint_reduction='mean', rescale_labels=False):
        super(PoseLoss, self).__init__(
            pose_loss, reduction=reduction, name=name,
            alignment=alignment, order=order, eps=eps, add_origin=add_origin,
            origin_index=origin_index, joint_reduction=joint_reduction,
            rescale_labels=rescale_labels)
