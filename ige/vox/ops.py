from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_grid_points(n, include_ends, dtype=tf.float32):
    """
    Get linearly spaced grid points between 0 and 1 with or without ends.

    e.g.
    get_grid_points(6, include_ends=True) == [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    get_grid_points(5, include_ends=False) == [0.1, 0.3, 0.5, 0.7, 0.9]

    Args:
        n: number of grid points
        include_ends: if True, both 0 and 1 are included in the returned
            tensor, i.e. the values correspond to boundaries of evenly spaced
            intervals. If False, the values correspond to center points of
            evenly spaced intervals.
        dtype: of returned tensor
    
    Returns:
        tensor shape `(n,)` dtype `dtype`.
    """
    with tf.name_scope('grid_points'):
        if include_ends:
            return tf.linspace(
                tf.zeros((), dtype=dtype), tf.ones((), dtype=dtype), n)
        else:
            x = tf.range(n, dtype=dtype)
            x += 0.5
            x = x / n
            return x


def get_z_points(z_near, z_far, n, include_ends, dtype=tf.float32):
    assert(z_near.shape.as_list() == z_far.shape.as_list())
    nd = z_near.shape.ndims
    z0 = get_grid_points(n, include_ends=include_ends, dtype=dtype)
    z_near, z_far = (tf.expand_dims(z, axis=-1) for z in (z_near, z_far))
    return z_near + tf.reshape(z0, (1,)*nd + (n,))*(z_far - z_near)


def get_frustum_weights(z_near, z_far, n, dtype=tf.float32):
    # # Approx frustum weights, w = z^2
    # z = get_z_points(z_near, z_far, n, include_ends=False, dtype=dtype)
    # w = tf.square(z)

    # # Exact frustum weights, w = (z*(z + dz) + dz^2)*dz
    # from difference of pyramid volumes
    z = get_z_points(z_near, z_far, n+1, include_ends=True, dtype=dtype)
    zn = z[..., :-1]
    zf = z[..., 1:]
    dz = zf - zn
    w = (zn * zf + tf.square(dz)/3)*dz
    return tf.expand_dims(tf.expand_dims(w, axis=-2), axis=-2)
