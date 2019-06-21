from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.metrics import MeanMetricWrapper  # pylint: disable=no-name-in-module
from ige.hpe.losses import pose_loss
from ige.hpe.losses import Alignment


class PoseLoss(MeanMetricWrapper):
    def __init__(
            self, alignment=Alignment.NONE,
            order=2, eps=None, add_origin=False, origin_index=None,
            denormalize_stats=None, name=None):
        super(PoseLoss, self).__init__(
            pose_loss, name=name,
            alignment=alignment, order=order, eps=eps, add_origin=add_origin,
            origin_index=origin_index, denormalize_stats=denormalize_stats)
