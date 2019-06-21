"""2D-3D human pose dimensionality lifting on Human 3.6 Million."""
# TODO(jackd):
# tests

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import random
import itertools

from absl import logging

import numpy as np
import tensorflow as tf

import tensorflow_datasets.public_api as tfds

from ige.hpe.data import skeleton as skel
from ige.hpe.data import mpii
from ige.hpe.data import h3m
from ige.hpe.data import transform_np as transform
from ige.hpe import ops
from tensorflow_datasets.core.download import resource as resource_lib

import h5py


H3M_CITATIONS = ("""\
@article{h36m_pami,
    author = {Ionescu, Catalin and Papava, Dragos and Olaru,
              Vlad and Sminchisescu, Cristian},
    title = {Human3.6M: Large Scale Datasets and Predictive Methods for
             3D Human Sensing in Natural Environments},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
    publisher = {IEEE Computer Society},
    year = {2014}
}""", """\
@inproceedings{IonescuSminchisescu11,
    author = {Catalin Ionescu, Fuxin Li, Cristian Sminchisescu},
    title = {Latent Structured Models for Human Pose Estimation},
    booktitle = {International Conference on Computer Vision},
    year = {2011}
}""")

BASELINE_CITATION = """\
@inproceedings{martinez_2017_3dbaseline,
    title={A simple yet effective baseline for 3d human pose estimation},
    author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and
            Little, James J.},
    booktitle={ICCV},
    year={2017}
}"""


CAMERA_IDS = (
    "54138969",
    "55011271",
    "58860488",
    "60457274",
)


_camera_index = {k: i for i, k in enumerate(CAMERA_IDS)}


TRAIN_SUBJECT_IDS = ('S1', 'S5', 'S6', 'S7', 'S8')
VALIDATION_SUBJECT_IDS = ('S9', 'S11')
SUBJECT_IDS = TRAIN_SUBJECT_IDS + VALIDATION_SUBJECT_IDS
_subject_index = {k: i for i, k in enumerate(SUBJECT_IDS)}

_SUBJECT_ID_FEATURE = tfds.features.ClassLabel(names=SUBJECT_IDS)
_SEQUENCE_ID_FEATURE = tfds.features.Text()
_CAMERA_ID_FEATURE = tfds.features.ClassLabel(names=CAMERA_IDS)

ExtrinsicCameraParams = collections.namedtuple(
    'ExtrinsicCameraParams', ['rotation', 'translation'])
IntrinsicCameraParams = collections.namedtuple(
    'IntrinsicCameraParams',
    ['focal_length', 'center', 'radial_dist_coeff', 'tangential_dist_coeff'])


def load_camera_params(path, subject_ids=SUBJECT_IDS, camera_ids=CAMERA_IDS):
    """Loads the cameras parameters of h36m

    Args:
        path: path to hdf5 file with h3m camera data
        subject_ids: list/tuple of subject ids
        camera_ids: list/tuple of camera_ids

    Returns:
        extrinsics: ExtrinsicCameraParams, named tuple with:
                rotation: Camera rotation matrix
                    shape (num_subject, num_cameras, 3, 3)

                translation: Camera translation parameters
                    shape (num_subject, num_cameras, 3)

        intrinsics: list of lists of IntrinsicCameraParams, named tuple with:
                focal_length: Camera focal length
                    shape (num_subject, num_cameras)
                center: Camera center
                    shape (num_subject, num_cameras, 2)
                radial_dist: Camera radial distortion coefficients
                    shape (num_subject, num_cameras, 3)
                tangential_dist: Camera tangential distortion coefficients
                    shape (num_subject, num_cameras, 2)

        The first two axis correspond to the order of `subject_ids` and
        `camera_ids` inputs.
    """

    num_subjects = len(subject_ids)
    num_cameras = len(camera_ids)

    def _init(*trailing_dims):
        return np.zeros(
            (num_subjects, num_cameras) + trailing_dims, dtype=np.float32)

    rotation = _init(3, 3)
    translation = _init(3)
    focal_length = _init(2)
    center = _init(2)
    radial_dist_coeff = _init(3)
    tangential_dist_coeff = _init(2)

    rest = (
        translation, focal_length, center, radial_dist_coeff,
        tangential_dist_coeff)
    rest_keys = ('T', 'f', 'c', 'k', 'p')

    with tf.io.gfile.GFile(path, "rb") as fp:
        hf = h5py.File(fp, "r")    # pylint: disable=no-member
        for i, subject_id in enumerate(subject_ids):
            for j, camera_id in enumerate(camera_ids):
                group = hf["subject%d/camera%d" % (
                    int(subject_id[1:]), _camera_index[camera_id]+1)]
                rotation[i, j] = np.array(group['R']).T
                for param, key in zip(rest, rest_keys):
                    param[i, j] = group[key][:, 0]

    extr = ExtrinsicCameraParams(rotation, rest[0])
    intr = IntrinsicCameraParams(*rest[1:])
    return extr, intr


def _filename_3d(sequence_id):
    return "%s.h5" % sequence_id.replace('_', ' ')


def _filename_2d(sequence_id, camera_id):
    return "%s.%s.h5" % (sequence_id.replace(' ', '_'), camera_id)


def _get_base_resource(manual_dir):
    # dl_manager doesn't like dropbox apparently...
    path = os.path.join(manual_dir, "h36m.zip")
    if not tf.io.gfile.exists(path):
        if not tf.io.gfile.exists(manual_dir):
            tf.io.gfile.makedirs(manual_dir)
        url = "https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip"
        ex = "wget -O %s %s" % (path, url)
        msg = ("Please manually download files from %s and place it at %s"
               "e.g.\n%s" % (url, path, ex))
        raise AssertionError(msg)
    return resource_lib.Resource(
        path=path,
        extract_method=resource_lib.ExtractMethod.ZIP)


def _pose_feature(skeleton, num_dims, dtype=tf.float32):
    return tfds.features.Tensor(
        shape=(skeleton.num_joints, num_dims), dtype=dtype)


class PoseLoader(collections.Mapping):
    @property
    def name(self):
        raise NotImplementedError

    @property
    def skeleton(self):
        raise NotImplementedError

    @property
    def value_feature(self):
        raise NotImplementedError

    @property
    def key_features(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def __len__(self):
        return sum(1 for _ in self)

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError

    def map(self, map_fn, name, value_feature=None):
        if value_feature is None:
            value_feature = self.value_feature
        return MappedLoader(
            self, map_fn, name=name, value_feature=value_feature)

    def download_and_prepare(self, dl_manager):
        pass


class MappedLoader(PoseLoader):
    def __init__(self, base, map_fn, name, value_feature):
        self._base = base
        self._map_fn = map_fn
        self._value_feature = value_feature

    @property
    def skeleton(self):
        return self._base.skeleton

    @property
    def value_feature(self):
        return self._value_feature

    def __getitem__(self, key):
        return self._map_fn(key, self._base[key])

    def __len__(self):
        return len(self._base)

    def __contains__(self, key):
        return key in self._base

    def keys(self):
        return self._base.keys()

    def download_and_prepare(self, dl_manager):
        self._base.download_and_prepare(dl_manager)


def _download_and_prepare_base(dl_manager):
    base_dir = dl_manager.extract(
            _get_base_resource(dl_manager.manual_dir))
    return os.path.join(base_dir, 'h36m')


class H5DirectoryLoader(PoseLoader):
    def __init__(self, group_key, name, skeleton):
        self._group_key = group_key
        self._name = name
        self._skeleton = skeleton

    @property
    def skeleton(self):
        return self._skeleton

    @property
    def name(self):
        return self._name

    def _path(self, key):
        raise NotImplementedError

    def __contains__(self, key):
        return tf.io.gfile.exists(self._path(key))

    def __getitem__(self, key):
        path = self._path(key)
        if not tf.io.gfile.exists(path):
            raise KeyError("Invalid key %s. No file at %s" % (str(key), path))
        with tf.io.gfile.GFile(path, "rb") as fobj:
            data = h5py.File(fobj, "r")[self._group_key][:]
        return data


class P3WorldLoader(H5DirectoryLoader):
    def __init__(self, skeleton):
        self._camera_params = None
        self._h36m_dir = None
        self._indices = skel.conversion_indices(h3m.s32, skeleton)
        super(P3WorldLoader, self).__init__(
            group_key="3D_positions", name="p3w", skeleton=skeleton)

    @property
    def key_features(self):
        out = collections.OrderedDict()
        out['subject_id'] = _SUBJECT_ID_FEATURE
        out['sequence_id'] = _SEQUENCE_ID_FEATURE
        return out

    @property
    def value_feature(self):
        return _pose_feature(self._skeleton, num_dims=3)

    def _subject_dir(self, subject_id):
        if self._h36m_dir is None:
            raise RuntimeError(
                'Cannot get subject dir before download_and_prepare called.')
        return os.path.join(
            self._h36m_dir, subject_id, "MyPoses", "3D_positions")

    def _path(self, key):
        subject_id, sequence_id = key
        return os.path.join(
            self._subject_dir(subject_id),
            _filename_3d(sequence_id))

    def __getitem__(self, key):
        data = super(P3WorldLoader, self).__getitem__(key)
        data = np.reshape(data, (32, 3, -1))
        data = np.transpose(data, (2, 0, 1))
        return data.astype(np.float32)[:, self._indices]

    def keys(self):
        for subject_id in SUBJECT_IDS:
            for fn in tf.io.gfile.listdir(self._subject_dir(subject_id)):
                sequence_id, ext = fn.split(".")
                assert(ext == "h5")
                yield (subject_id, sequence_id)

    @property
    def camera_params(self):
        if self._camera_params is None:
            self._camera_params = load_camera_params(self.camera_path)
        return self._camera_params
    
    @property
    def camera_path(self):
        return os.path.join(self._h36m_dir, 'cameras.h5')

    def download_and_prepare(self, dl_manager):
        self._h36m_dir = _download_and_prepare_base(dl_manager)


def _per_camera_key_features():
    out = collections.OrderedDict()
    out['subect_id'] = _SUBJECT_ID_FEATURE
    out['sequence_id'] = _SEQUENCE_ID_FEATURE
    out['camera_id'] = _CAMERA_ID_FEATURE
    return out


class P3CameraLoader(PoseLoader):
    def __init__(self, world_loader):
        self._p3w = world_loader

    @property
    def name(self):
        return 'p3c'

    @property
    def skeleton(self):
        return self._p3w.skeleton

    @property
    def key_features(self):
        return _per_camera_key_features()

    @property
    def value_feature(self):
        return self._p3w.value_feature

    def __getitem__(self, key):
        subject_id, sequence_id, camera_id = key
        extr = [e[_subject_index[subject_id], _camera_index[camera_id]]
                for e in self.camera_params[0]]
        p3w = self._p3w[subject_id, sequence_id]

        p3c = transform.world_to_camera_frame(p3w.reshape((-1, 3)), *extr)
        return p3c.reshape(p3w.shape)

    def __len__(self):
        return len(self._p3w) * len(CAMERA_IDS)

    def keys(self):
        return (k + (c,) for k, c in itertools.product(self._p3w, CAMERA_IDS))

    def __contains__(self, key):
        return key[:-1] in self._p3w and key[-1] in CAMERA_IDS

    @property
    def camera_params(self):
        return self._p3w.camera_params
    
    @property
    def camera_path(self):
        return self._p3w.camera_path

    def download_and_prepare(self, dl_manager):
        self._p3w.download_and_prepare(dl_manager)


def ProjectionsLoader(camera_loader):

    def map_fn(key, p3c):
        intrinsics = camera_loader.camera_params[1]
        subject_id, sequence_id, camera_id = key
        del sequence_id
        intr = [i[_subject_index[subject_id], _camera_index[camera_id]]
                for i in intrinsics]
        base = camera_loader[key]
        num_joints = base.shape[-2]
        return transform.project_points_radial(
            base.reshape(-1, 3), *intr)[0].reshape((-1, num_joints, 2))

    feature = camera_loader.value_feature
    shape = list(feature.shape)
    shape[-1] = 2
    feature = tfds.features.Tensor(shape=tuple(shape), dtype=feature.dtype)
    return camera_loader.map(map_fn, name='p2', value_feature=feature)


class HourglassLoader(H5DirectoryLoader):
    def __init__(self, subject_ids=SUBJECT_IDS):
        self._subject_ids = subject_ids
        super(HourglassLoader, self).__init__(
            group_key="poses", skeleton=mpii.s16, name='hourglass')

    @property
    def key_features(self):
        return _per_camera_key_features()

    @property
    def value_feature(self):
        return _pose_feature(mpii.s16.num_joints, num_dims=2)

    def download_and_prepare(self, dl_manager):
        self._h36m_dir = _download_and_prepare_base(dl_manager)

    def _subject_dir(self, subject_id):
        return os.path.join(
            self._h36m_dir, subject_id,
            "MyPoses", "3D_positions", "StackedHourglass")

    def _path(self, key):
        subject_id, sequence_id, camera_id = key
        return os.path.join(
            self._subject_dir(subject_id),
            _filename_2d(sequence_id, camera_id))

    def keys(self):
        for subject_id in self._subject_ids:
            for fn in tf.io.gfile.listdir(self._subject_dir(subject_id)):
                sequence_id, camera_id, ext = fn.split(".")
                assert(ext == "h5")
                yield (subject_id, sequence_id, camera_id)


class FinetunedHourglassLoader(H5DirectoryLoader):
    def __init__(self, subject_ids=SUBJECT_IDS):
        self._subject_ids = subject_ids
        super(FinetunedHourglassLoader, self).__init__(
            group_key="poses", name='finetuned_hourglass', skeleton=mpii.s16)

    @property
    def key_features(self):
        return _per_camera_key_features()

    @property
    def value_feature(self):
        return _pose_feature(mpii.s16.num_joints, num_dims=2)

    def download_and_prepare(self, dl_manager):
        manual_dir = dl_manager.manual_dir
        url = "https://drive.google.com/open?id=0BxWzojlLp259S2FuUXJ6aUNxZkE"
        # DL manager fails to get drive data, so using manual fallback
        path = os.path.join(
            manual_dir, "stacked_hourglass_fined_tuned_240.tar.gz")
        if not tf.io.gfile.exists(path):
            raise AssertionError(
                "You must download finetuned dataset files manually from %s "
                "and place them in `%s`. Some users experience issues using "
                "chrome which are resolved by using firefox." % (url, path))
        resource = resource_lib.Resource(
            path=path,
            extract_method=resource_lib.ExtractMethod.TAR_GZ,
        )
        self._finetuned_dir = dl_manager.extract(resource)

    def _subject_dir(self, subject_id):
        return os.path.join(
            self._finetuned_dir, subject_id, "StackedHourglassFineTuned240")

    def _path(self, key):
        subject_id, sequence_id, camera_id = key
        return os.path.join(
            self._subject_dir(subject_id),
            _filename_2d(sequence_id, camera_id))

    def keys(self):
        for subject_id in self._subject_ids:
            for fn in tf.io.gfile.listdir(self._subject_dir(subject_id)):
                sequence_id, camera_id, ext = fn.split(".")
                assert(ext == "h5")
                yield (subject_id, sequence_id, camera_id)


def get_origin_indices(skeleton):
    if skeleton == mpii.s16:
        indices = tuple(
            mpii.s16.index(j) for j in (skel.l_hip, skel.r_hip))
    else:
        indices = h3m.get_origin_index(skeleton.num_joints)
    return np.asanyarray(indices)


def get_skeleton_origin(skeleton, points, keepdims=False):
    indices = get_origin_indices(skeleton)
    return ops.get_midpoint(points, indices)


def shift_origin(skeleton, points):
    return points - get_skeleton_origin(skeleton, points, keepdims=True)


class H3mLiftConfig(tfds.core.BuilderConfig):
    def __init__(
            self, p2_loader, p3_loader, **kwargs):
        self._p2_loader = p2_loader
        self._p3_loader = p3_loader
        super(H3mLiftConfig, self).__init__(**kwargs)

    def load_camera_params(self):
        return self._p3_loader.load_camera_params()

    @property
    def skeleton_2d(self):
        return self._p2_loader.skeleton
    
    @property
    def skeleton_3d(self):
        return self._p3_loader.skeleton

    @property
    def features(self):
        return tfds.features.FeaturesDict(dict(
            subject_id=_SUBJECT_ID_FEATURE,
            camera_id=_CAMERA_ID_FEATURE,
            sequence_id=_SEQUENCE_ID_FEATURE,
            pose_2d=self._p2_loader.value_feature,
            pose_3d=self._p3_loader.value_feature,
        ))

    @property
    def supervised_keys(self):
        return ('pose_2d', 'pose_3d')
    
    @property
    def camera_path(self):
        return self._p3_loader.camera_path

    def download_and_prepare(self, dl_manager):
        for loader in (self._p2_loader, self._p3_loader):
            loader.download_and_prepare(dl_manager)
    
    def load_pose_data(self, subject_id, sequence_id, camera_id, take_every=None):
        p2 = self._p2_loader[subject_id, sequence_id, camera_id]
        try:
            p3 = self._p3_loader[subject_id, sequence_id, camera_id]
        except KeyError:
            p3 = self._p3_loader[subject_id, sequence_id]
        if take_every is not None:
            p2 = p2[::take_every]
            p3 = p3[::take_every]
        return p2, p3

    def load_sequence_data(
            self, subject_id, sequence_id, camera_id, take_every=None):
        p2, p3 = self.load_pose_data(
            subject_id, sequence_id, camera_id, take_every)
        return dict(
            pose_2d=p2,
            pose_3d=p3,
            subject_id=subject_id,
            sequence_id=sequence_id,
            camera_id=camera_id
        )
    
    def keys(self):
        return self._p2_loader.keys()

    def load_all_data(
            self, key_filter=lambda subject_id, sequece_id, camera_id: True,
            take_every=None, repeat_ids=True, shuffle_files=False,
            shuffle_all=False):
        import random
        import tqdm
        data = []
        keys = [key for key in self._p2_loader if key_filter(*key)]
        if shuffle_files:
            random.shuffle(keys)
        logging.info('Gathering all data for H3mLift/%s' % self.name)
        data = [
            self.load_sequence_data(*key, take_every=take_every)
            for key in tqdm.tqdm(keys)]
        lengths = [d['pose_2d'].shape[0] for d in data]
        out = {
            k: np.concatenate([d[k] for d in data])
            for k in ('pose_2d', 'pose_3d')}
        ids = ('subject_id', 'sequence_id', 'camera_id')
        for i in ids:
            out[i] = [d[i] for d in data]
        if repeat_ids:
            for i in ids:
                out[i] = np.repeat(out[i], lengths)
            if shuffle_all:
                order = list(range(len(out['pose_2d'])))
                random.shuffle(order)
                for k in out:
                    out[k] = out[k][order]

        return out


_p3c = P3CameraLoader(P3WorldLoader(h3m.s16))

PROJECTION = H3mLiftConfig(
    ProjectionsLoader(P3CameraLoader(P3WorldLoader(mpii.s16))), _p3c,
    name='projection', version="0.0.1",
    description="ground truth projections with camera frame 3d pose")

HOURGLASS = H3mLiftConfig(
    HourglassLoader(), _p3c,
    name="hourglass", version="0.0.1",
    description="hourglass detections with camera frame 3d pose")

FINETUNED = H3mLiftConfig(
    FinetunedHourglassLoader(), _p3c,
    name="finetuned", version="0.0.1",
    description="finetued hourglass detections with camera fram 3d pose")


class H3mLift(tfds.core.GeneratorBasedBuilder):
    """2D - 3D pose lifting task for human pose estimation on human 3.6m.

    `H3mLift.load_camera_params` provides camera parameters. Basic
    transformations and projection operations are available in `transform`.

    For information about the joints used, see `skeleton`

    3D data is provided in world coordinates by default on a 16 joint h3m.

    2D data is available from 3 configs:
        * `ground_truth`: the 3D poses projected to 2D using camera parameters,
            using the same skeleton as 3D poses (h3m.s16)
        * `hourglass`: a stacked hourglass network trained on MPII images.
            Note the skeleton here is `h3m.mpii_s16`, which has the same
            number of joints, but different joints (only 1 joint in the head
            and an independent pelvis).
        * `hourglass_finetuned`: similar to `hourglass` except the network is
            finetuned on h3m dataset (same source as the 3D poses).
    """

    BUILDER_CONFIGS = [PROJECTION, HOURGLASS, FINETUNED]

    def _copy_camera_data(self, src):
        cameras_path = self._camera_path
        if not tf.io.gfile.exists(cameras_path):
            folder = os.path.dirname(cameras_path)
            if not tf.io.gfile.exists(folder):
                tf.io.gfile.makedirs(folder)
            tf.io.gfile.copy(src, cameras_path)

    @property
    def _camera_path(self):
        return os.path.join(self._data_dir, "h36m", "cameras.h5")

    def load_camera_params(
            self, subject_ids=SUBJECT_IDS, camera_ids=CAMERA_IDS):
        return load_camera_params(
            self._camera_path, subject_ids=subject_ids, camera_ids=camera_ids)

    def _info(self):
        config = self.builder_config
        h3m_url = "http://vision.imar.ro/human3.6m/description.php"
        baseline_url = "https://github.com/una-dinosauria/3d-pose-baseline"
        hourglass_url = "https://github.com/princeton-vl/pose-hg-demo"
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                    "2D / 3D human poses for varying number of joints and "
                    "2D sources"),
            features=config.features,
            urls=[h3m_url, baseline_url, hourglass_url],
            supervised_keys=config.supervised_keys,
            citation="\n".join(H3M_CITATIONS + (BASELINE_CITATION,)),
        )

    def _split_generators(self, dl_manager):
        self.builder_config.download_and_prepare(dl_manager)
        self._copy_camera_data(self.builder_config.camera_path)
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=10,
                gen_kwargs=dict(subject_ids=TRAIN_SUBJECT_IDS)
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                num_shards=4,
                gen_kwargs=dict(subject_ids=VALIDATION_SUBJECT_IDS)
            )
        ]
    
    def _generate_examples(self, subject_ids):
        subject_ids = set(subject_ids)
        def key_filter(subject_id, sequence_id, camera_id):
            return subject_id in subject_ids

        data = self.builder_config.load_all_data(
            key_filter=key_filter, repeat_ids=True)
        n = data['pose_2d'].shape[0]
        for i in range(n):
            yield {k: v[i] for k, v in data.items()}
    
    def num_examples(self, split):
        return int(self.info.splits[split].num_examples)
