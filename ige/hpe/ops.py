from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from ige.tf_compat import dim_value

from ige.hpe.procrustes import procrustes_aligned  # noqa


def get_pairwise_difference(x):
    """
    Get the pairwise difference between penultimate axis values.

    Args:
        x: [..., n, d]: d-dimensional coordinates

    Returns:
        diff: [..., n * (n - 1) // 2, d] unique pairwise differences
            x_i - x_j for all i in [0, n), j in [i+1, n).
    """
    with tf.name_scope('pairwise_difference'):
        shape = x.shape
        n = dim_value(shape[-2])
        x0 = tf.expand_dims(x, axis=-3)
        x1 = tf.expand_dims(x, axis=-2)
        diff = x0 - x1
        mask = tf.linalg.band_part(tf.ones((n, n), dtype=np.bool), 0, -1)
        mask = tf.logical_and(mask, tf.logical_not(tf.eye(n, dtype=tf.bool)))
        masked = tf.boolean_mask(diff, mask, axis=shape.ndims - 2)
        masked = tf.reshape(masked, (-1, n * (n - 1) // 2, shape[-1]))
    return masked


def get_pairwise_dist2(x):
    """
    Get the squared pairwise distance between points.

    Args:
        x: [..., n, d] d-dimensional coordinates.

    Returns:
        dist2: [..., n * (n - 1) // 2] unique squared euclidean distances
            between points, |x_i - x_j|**2 for all i in [0, n), j in [i+1, n).
    """
    with tf.name_scope('pairwise_dist2'):
        diff = get_pairwise_difference(x)
        pairwise_dist2 = tf.reduce_sum(tf.square(diff), axis=-1)
    return pairwise_dist2


def get_square_skeleton_scale_factor(points, i0, i1, keepdims=True):
    """
    Get the squared scale factor associated with the given inferred skeleton.

    This is defined as the distance between points i0 and i1.

    Args:
        points: 3D pose, [..., num_joints, 3]
        i0: scalar int of first joint
        i1: scalar int of second joint
        keepdims: whether or not to keep axes

    Returns:
        Squared distance between left and right hips, [..., 1, 1] if keepdims
            else [...]
    """
    if i0 == i1:
        raise ValueError('indices cannot be the same')
    with tf.name_scope('square_skeleton_scale_factor'):
        points = tf.gather(points, [i0, i1], axis=-2)
        l, r = tf.split(points, [1, 1], axis=-2)
        dist2 = tf.reduce_sum(
            tf.math.squared_difference(l, r), keepdims=keepdims, axis=-1)
        if not keepdims:
            dist2 = tf.squeeze(dist2, axis=-1)
        return dist2


def get_opt_scale_factor(p3c_gt, p3c_inf, axis=(-2, -1), keepdims=False):
    """
    Get the optimal scaling factor to scale inferred pose to match labels.

    optimality is defined by least squared distance across specified axis/axes.

    Args:
        p3c_gt: 3d pose in camera coordinates, ground truth
        p3c_inf: 3d pose in camera coordinates, model output
        axis: int, or tuple of ints denoting optimization axis/axes
        keepdims: if True, reduced dimensions are kept with size 1.

    Returns:
        scaling factor, same size as broadcasted p3c_gt * p3c_inf without
            the specified `axis` (or replaced by 1 if `keepdims`).
    """
    with tf.name_scope('opt_scale_factor'):
        numer = tf.reduce_sum(p3c_gt*p3c_inf, axis=axis, keep_dims=keepdims)
        denom = tf.reduce_sum(
            tf.square(p3c_inf), axis=axis, keep_dims=keepdims)
        factor = numer / denom
    return factor


def opt_scale_aligned(p3c_gt, p3c_inf):
    """
    Inferred solution optimally scaled according to least squares.

    Args:
        p3c_gt: 3d pose in camera coordinates, ground truth
        p3c_inf: 3d pose in camera coordinates, model output

    Returns:
        An optimally scaled version of `p3c_inf`.
    """
    with tf.name_scope('opt_scale_aligned'):
        factor = get_opt_scale_factor(p3c_gt, p3c_inf, keepdims=True)
        return p3c_inf * factor


def origin_aligned(p3_gt, p3_inf, origin_index):
    """
    Inferred solution with origin shifted to ground truth origin.

    Args:
        p3_gt: (..., num_joints, d) pose, ground truth
        p3_inf: (..., num_joints, d) pose, model output
        origin_index: scalar int or tuple of ints defining the origin. If a
            tuple, the midpoint of the specified indices is used.

    Returns:
        `p3_inf` shifted such that the origin is atop `p3_inf`'s origin.
    """
    with tf.name_scope('origin_aligned'):
        origin_gt = get_midpoint(p3_gt, origin_index)
        origin_inf = get_midpoint(p3_inf, origin_index)
        return p3_inf - origin_inf + origin_gt


def denormalized(x, mean=None, std=None):
    """Invert normalization in-place."""
    if std is not None:
        x = x * std
    if mean is not None:
        x = x + mean
    return x


def normalized(x, mean=None, std=None):
    """`(x - mean) / std`, with identity ops for `None` mean/std."""
    if mean is not None:
        x = x - mean
    if std is not None:
        x = x / std
    return x



_reducers = {
    'mean': tf.reduce_mean,
    'sum': tf.reduce_sum
}


def get_error(
        y_true, y_pred, order=2, axis=-1, eps=None, joint_reduction='mean'):
    """
    Parameterized error function that potentially avoids sqrts near zero.

    Roughly equivalent to
    `err = reduce(norm(euclidean_distance(y_true, y_pred, axis=axis)**order))`

    Args:
        y_true: (B, num_joints, d) float label values.
        y_pred: (B, num_joints, d) float model output.
        order: order of error, e.g. 2 for squared difference.
        eps: optional small shift to avoid gradient issues for fractional
            powers near zero.

    Returns:
        error, shape (B,), mean per-joint error.
    """
    with tf.name_scope('error'):
        err = tf.reduce_sum(
            tf.math.squared_difference(y_true, y_pred), axis=-1)
        # err is squared euclidean distance, so order must be halved.
        if order != 2:
            if order % 2 == 0:
                err = tf.pow(err, order // 2)
            else:
                # small shift to avoid bad gradients near zero.
                if eps is not None:
                    if eps < 0:
                        raise ValueError(
                            'eps must be non-negative, got %f' % eps)
                    err += eps
                if order == 1:
                    err = tf.sqrt(err)
                else:
                    err = tf.pow(err, order / 2)
        # return tf.reduce_mean(err, axis=-1)
        return _reducers[joint_reduction](err, axis=-1)


def project_points_radial_normalized(
        points, radial_dist_coeff, tangential_dist_coeff):
    """
    Courtesy of Martineze et al.

    https://github.com/una-dinosauria/3d-pose-baseline
    """
    assert(len(points.shape) >= 2)
    assert(points.shape[-1] == 3)

    # pylint: disable=E0632
    XX, D = tf.split(points, (2, 1), axis=-1)
    XX = XX / D
    D = tf.squeeze(D, axis=-1)
    r2 = tf.reduce_sum(tf.square(XX), axis=-1)
    tangential_dist_coeff = tf.reverse(
        tangential_dist_coeff,
        [False] * (len(tangential_dist_coeff.shape)-1) + [True])

    r22 = r2*r2
    r23 = r22*r2
    r_pows = tf.stack((r2, r22, r23), axis=-1)
    radial = 1 + tf.reduce_sum(tf.expand_dims(
        radial_dist_coeff, axis=-2) * r_pows, axis=-1)
    tan = tf.reduce_sum(tf.expand_dims(
        tangential_dist_coeff, axis=-2)*XX, axis=-1)
    
    normalized_proj = XX * (
        tf.expand_dims(radial + tan, axis=-1)
        + tf.expand_dims(r2, axis=-1) * tf.expand_dims(
            tangential_dist_coeff, axis=-2))

    return normalized_proj, D, radial, tan, r2


def project_points_radial(P, f, c, k, p):
    """
    Project points from 3d to 2d using camera parameters
    including radial and tangential distortion

    Args:
        P: (B, N, 3) points in world coordinates
        f: (B, 2) Camera focal length
        c: (B, 2) Camera center
        k: (B, 3) Camera radial distortion coefficients
        p: (B, 2) Camera tangential distortion coefficients

    Returns:
        Proj: (B, N, 2) points in pixel space
        D: (B, N) depth of each point in camera space
        radial: (B, N) radial distortion per point
        tan: (B, N) tangential distortion per point
        r2: (B, N) squared radius of the projected points before distortion

    Courtesy of Martineze et al.

    https://github.com/una-dinosauria/3d-pose-baseline
    """
    nproj, D, radial, tan, r2 = project_points_radial_normalized(P, k, p)

    Proj = (tf.expand_dims(f, axis=-2) * nproj) + tf.expand_dims(c, axis=-2)

    return Proj, D, radial, tan, r2


def apply_rotation_matrix(points, R, inverse=False):
    """
    Apply rotation matrix R to points.

    Both `points` and `R` can have any number of leading dimensions, but they
    must be the same, or `R` can have 1 fewer. If 1 fewer, `R` is tiled along
    the third last dimension. After this optional tiling, all leading
    dimensions must match.

    Args:
        points: (..., num_joints, 3) float array of optionally batched joint
            coordinates.
        R: (..., 3, 3) float array of optionally batched rotation matrices.
            No check is made to ensure these are valid rotation matrices.
        inverse: if True, the inverse rotation is applied. Note the inverse is
            assumed to be the transpose, so if `R` is not a valid rotation
            matrix results with `inverse=True` will be rubbish.

    Returns:
        rotated points, same size as `points`.
    """
    points = tf.convert_to_tensor(points)
    R = tf.convert_to_tensor(R)
    assert(R.shape[-2:] == (3, 3))
    assert(points.shape[-1] == 3)
    if R.shape.ndims == points.shape.ndims - 1:
        repeats = [1]*points.shape.ndims
        repeats[-3] = tf.shape(points)[-3]
        R = tf.tile(tf.expand_dims(R, axis=-3), points)
    assert(R.shape.ndims == points.shape.ndims)
    # return batch_matmul(points, R, transpose_b=not inverse)
    return tf.matmul(points, R, transpose_b=not inverse)


def p3c_to_p3w(p3c, R, t):
    """Convert camera coordinates to world coordinates."""
    t = tf.expand_dims(t, axis=-2)
    p3w = apply_rotation_matrix(p3c, R, inverse=True) + t
    return p3w


def p3w_to_p3c(p3w, R, t):
    """Convert world coordinates to camera coordinates."""
    t = tf.expand_dims(t, axis=-2)
    p3c = apply_rotation_matrix(p3w - t, R, inverse=True)
    return p3c


def get_midpoint(points, indices, axis=-2, keepdims=False):
    """
    Get the midpoint of `points` at `indices` on the specified axis.

    Args:
        points: (..., num_joints, d) d-dimensional points
        indices: int or tuple or ints giving indices of `points` at `axis` to
            compute midpoint of
        keepdims: see `Returns` section below.

    Returns:
        (..., d) float midpoint of specified indices, or shaped
        ..., 1, d) if `keepdims`.
    """
    with tf.name_scope('midpoint'):
        indices = tf.convert_to_tensor(indices, dtype=tf.int64)
        if indices.shape.ndims == 0:
            mid = tf.gather(points, indices, axis=axis)
            if keepdims:
                mid = tf.expand_dims(mid, axis=-2)
            return mid
        else:
            mid = tf.gather(points, indices, axis=axis)
            return tf.reduce_mean(mid, axis=axis, keepdims=keepdims)


def append_midpoint(points, indices):
    """Get a pose with appended origin joint between the hip joints."""
    with tf.name_scope('add_origin'):
        mid = get_midpoint(points, indices, keepdims=True)
        return tf.concat([points, mid], axis=-2)
