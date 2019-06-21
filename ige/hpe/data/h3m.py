from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ige.hpe.data import skeleton as s


def _s32_links():
    """Original 32 joint skeleton links.

    This contains a lot of redundant information. We don't bother with parent
    information here as this is purely used to conversion
    """
    joints = ['joint%d' % i for i in range(32)]
    for (i, v) in (
            (0, s.pelvis),
            (1, s.r_hip),
            (2, s.r_knee),
            (3, s.r_ankle),
            (6, s.l_hip),
            (7, s.l_knee),
            (8, s.l_ankle),
            (12, s.thorax),
            (13, s.neck),
            (14, s.head_center),
            (15, s.head_back),
            (17, s.l_shoulder),
            (18, s.l_elbow),
            (19, s.l_wrist),
            (25, s.r_shoulder),
            (26, s.r_elbow),
            (27, s.r_wrist),
            ):
        joints[i] = v
    return tuple((j, None) for j in joints)


# original 32 joint skeleton, containing repeated/calculated joints
s32 = s.Skeleton(_s32_links(), name="h3m_s32")

# reduced skeleton, containin 16 independent joints and pelvis
# pelvis is calculated as center of hips
s17 = s.Skeleton((
        (s.pelvis, None),
        (s.r_hip, s.pelvis),
        (s.r_knee, s.r_hip),
        (s.r_ankle, s.r_knee),
        (s.l_hip, s.pelvis),
        (s.l_knee, s.l_hip),
        (s.l_ankle, s.l_knee),
        (s.thorax, s.pelvis),
        (s.neck, s.thorax),
        (s.head_center, s.neck),
        (s.head_back, s.head_center),
        (s.l_shoulder, s.neck),
        (s.l_elbow, s.l_shoulder),
        (s.l_wrist, s.l_elbow),
        (s.r_shoulder, s.neck),
        (s.r_elbow, s.r_shoulder),
        (s.r_wrist, s.r_elbow),
), name="h3m_s17")

# s17 without pelvis
s16 = s.Skeleton((
        (s.r_hip, s.thorax),
        (s.r_knee, s.r_hip),
        (s.r_ankle, s.r_knee),
        (s.l_hip, s.thorax),
        (s.l_knee, s.l_hip),
        (s.l_ankle, s.l_knee),
        (s.thorax, None),
        (s.neck, s.thorax),
        (s.head_center, s.neck),
        (s.head_back, s.head_center),
        (s.l_shoulder, s.neck),
        (s.l_elbow, s.l_shoulder),
        (s.l_wrist, s.l_elbow),
        (s.r_shoulder, s.neck),
        (s.r_elbow, s.r_shoulder),
        (s.r_wrist, s.r_elbow),
), name="h3m_s16")

# s16 without thorax or head_center
s14 = s.Skeleton((
        (s.head_back, None),
        (s.neck, s.head_back),
        (s.r_shoulder, s.neck),
        (s.r_elbow, s.r_shoulder),
        (s.r_wrist, s.r_elbow),
        (s.l_shoulder, s.neck),
        (s.l_elbow, s.l_shoulder),
        (s.l_wrist, s.l_elbow),
        (s.r_hip, s.neck),
        (s.r_knee, s.r_hip),
        (s.r_ankle, s.r_knee),
        (s.l_hip, s.neck),
        (s.l_knee, s.l_hip),
        (s.l_ankle, s.l_knee),
), name="h3m_s14")


_origin_indices = {
    14: (s14.index(s.l_hip), s14.index(s.r_hip)),
    16: (s16.index(s.l_hip), s16.index(s.r_hip)),
    17: s17.index(s.pelvis),
}


def get_origin_index(num_joints):
    return _origin_indices[num_joints]
