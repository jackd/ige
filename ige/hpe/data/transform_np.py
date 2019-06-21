"""Numpy version of transforms to deal with the cameras of `human3.6m`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def project_points_radial(
        points, focal_length, center, radial_dist_coeff,
        tangential_dist_coeff):
    """
    Project points from 3d to 2d using camera parameters with distortion.

    Args
        points: (N, 3) points in world coordinates
        focal_length: (2,) Camera focal length
        center: (2,) Camera center
        radial_dist_coeff: (3,) Camera radial distortion coefficients
        tangential_dist_coeff: (2,) Camera tangential distortion coefficients

    Returns
        proj: (N, 2) points in pixel space
        depth: (N,) depth of each point in camera space
        radial_dist: (N,) radial distortion per point
        tangential_dist: (N,) tangential distortion per point
        r2: (N,) squared radius of the projected points before distortion
    """
    assert(len(points.shape) == 2)
    assert(points.shape[-1] == 3)

    # pylint: disable=unbalanced-tuple-unpacking
    xy = points[..., :2]
    depth = points[..., 2:]
    xy = xy / depth
    depth = np.squeeze(depth, axis=-1)
    r2 = np.sum(np.square(xy), axis=-1)

    r22 = r2*r2
    r23 = r22*r2
    r_pows = np.stack((r23, r22, r2), axis=-1)
    radial_dist = 1 + np.sum(radial_dist_coeff * r_pows, axis=-1)
    tangential_dist = np.sum(tangential_dist_coeff*xy, axis=1)

    distorted_xy = xy * (
        np.expand_dims(radial_dist + tangential_dist, axis=-1) +
        np.expand_dims(r2, axis=-1) * np.expand_dims(
            tangential_dist_coeff, axis=-2))

    proj = (focal_length * distorted_xy) + center

    return proj, depth, radial_dist, tangential_dist, r2


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
    return np.matmul(points_world - translation, rotation.T)


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
    return np.matmul(points_camera, rotation) + translation
