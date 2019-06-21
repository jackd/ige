"""Tensorflow utilities to deal with the cameras of `human3.6m`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _validate_transform_shapes(points, rotation, translation):
    shape = points.shape
    if shape[-1] != 3:
        raise ValueError(
            "points must have shape[-1] == 3, got %s" % str(shape))
    shape = rotation.shape
    if shape != (3, 3):
        raise ValueError(
            "rotation must have shape (3, 3), got %s" % str(shape))
    shape = translation.shape
    if shape != (3,):
        raise ValueError(
            "translation must have shape (3,), got %s" % str(shape))


def world_to_camera_frame(points_world, rotation, translation):
    """
    Convert points from world to camera coordinates

    Args
        points_world: (N, 3) 3d points in world coordinates
        rotation: (3, 3) Camera rotation matrix
        translation: (3,) Camera translation parameters

    Returns
        points_cam: (N, 3) 3d points in camera coordinates
    """
    _validate_transform_shapes(points_world, rotation, translation)
    return tf.matmul(points_world - translation, rotation, transpose_b=True)


def camera_to_world_frame(points_camera, rotation, translation):
    """Inverse of world_to_camera_frame

    Args
        points_camera: (N, 3) points in camera coordinates
        rotation: (3, 3) Camera rotation matrix
        translation: (3,) Camera translation parameters
    Returns
        points_world: (N, 3) points in world coordinates
    """
    _validate_transform_shapes(points_camera, rotation, translation)
    return tf.matmul(points_camera, rotation) + translation
