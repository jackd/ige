from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper # pylint: disable=no-name-in-module
import gin


def continuous_iou(y_true, y_pred, weights=None):
    with tf.name_scope('continuous_iou'):
        intersection = tf.where(
            y_true, y_pred, tf.zeros_like(y_pred))
        union = tf.where(
            y_true, tf.ones_like(y_pred), y_pred)
        if weights is not None:
            intersection = intersection * weights
            union = union * weights
        intersection = tf.reduce_sum(intersection, axis=(1, 2, 3))
        union = tf.reduce_sum(union, axis=(1, 2, 3))
        iou = intersection / union
    return iou


def weighted_binary_crossentropy(
        y_true, y_pred, weights=None, alpha=None, gamma=None,
        from_logits=True):
    """
    Get possibly weighted cross-entropy loss.

    Based on https://arxiv.org/pdf/1708.02002.pdf

    Args:
        y_true: boolean dataset
        y_pred: output of model, probs or logits
        weights: optional float32 additional weighting factor. Shape must be
            broadcastable to `y_pred.shape`.
        alpha: if not None, values are weighted by (1+alpha) and (1-alpha).
            Note this is different to the paper, but means
            alpha = 0 corresponds to standard cross-entropy
            alpha > 0 emphasizes positive examples
            alpha < 0 emphasizes negative examples
            The form used in the paper can be calculated using
                loss = 0.5 * get_cross_entropy_loss(
                    y_pred, y_true, alpha=2*paper_alpha - 1)
        gamma: if not None, values are weighted by (1 - p_t)**gamma,
            where p_t is `probs` or `1 - probs` depending on y_true.
            ``. This implements focal loss from the paper.
        from_logits: if True, y_pred represents logits, otherwise it is treated
            as probabilities (sigmoid(logits)) and assumed to be in [0, 1].

    Returns: [batch_size] weighted average loss over each example.
    """
    with tf.name_scope('cross_entropy_weights'):
        if from_logits:
            logits = y_pred
        else:
            raise NotImplementedError()
        weights = [] if weights is None else [weights]

        if gamma is not None and gamma != 0:
            # focal loss
            if from_logits:
                focal_weights = tf.sigmoid(tf.where(y_true, -logits, logits))
            else:
                focal_weights = tf.where(y_true, 1 - y_pred, y_pred)

            if gamma == 2:
                focal_weights = tf.square(focal_weights)
            elif gamma != 1:
                focal_weights = focal_weights ** gamma
            weights.append(focal_weights)

        if alpha is not None and alpha != 0:
            # doubled alpha-balanced loss
            # so alpha=0 corresponds to standard cross-entropy
            # alpha > 0 penalizes false negatives
            # alpha < 0 penalizes false positives
            # also used in https://arxiv.org/pdf/1802.00411.pdf
            ones = tf.ones_like(logits)
            alpha_weights = tf.where(
                y_true, (1 + alpha) * ones, (1 - alpha) * ones)
            weights.append(alpha_weights)

        y_true = tf.to_float(y_true)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=logits)

        if len(weights) > 0:
            weights_total = weights[0]
            for w in weights[1:]:
                weights_total = weights_total * w
            loss = loss*weights_total

        axis = list(range(1, loss.shape.ndims))
        return tf.reduce_sum(loss * weights_total, axis=axis) / tf.reduce_sum(
            weights_total, axis=axis)


@gin.configurable
class ContinuousIou(LossFunctionWrapper):
    def __init__(self, reduction='sum_over_batch_size', **kwargs):
        super(ContinuousIou, self).__init__(
            continuous_iou, reduction=reduction, **kwargs)


@gin.configurable
class WeightedBinaryCrossentropy(LossFunctionWrapper):
    def __init__(
            self, reduction='sum_over_batch_size', alpha=None, gamma=None,
            from_logits=True):
        super(WeightedBinaryCrossentropy, self).__init__(
            weighted_binary_crossentropy,
            reduction=reduction,
            alpha=alpha,
            gamma=gamma,
            from_logits=from_logits,
        )