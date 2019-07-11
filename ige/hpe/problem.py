from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gin
from absl import logging
from ige.metrics import FinalStepMetric
from ige.losses import MultiStepLoss
from ige.problem import Problem
from ige.problem import TfdsProblem
from ige.hpe import ops
from ige.hpe import metrics as m
from ige.hpe.losses import Alignment
from ige.hpe.losses import PoseLoss
from ige.hpe.losses import pose_loss
from ige.hpe.data import skeleton as s
from ige.hpe.data import h3m
from ige.hpe.data import mpii
from ige.hpe.data import builder as b

SPACE_SCALE = 1000.0  # mm -> m


@gin.configurable(module='hpe')
class HpeProblem(TfdsProblem):
    def __init__(
            self,
            config=b.PROJECTION,
            include_procrustes=False,
            include_intrinsics=False,
            prediction_is_sequence=False,
            loss_decay=0.9,
            download_and_prepare=True,
            shuffle_buffer=100000):
        """"
        Problem specificaiton for human pose estimation.

        Args:
            config: one of `H3mLift.BUILDER_CONFIG` or a name
                ('projection', 'hourglass', 'finetuned')
            include_procrustes: if True, includes metric calculated with
              procrustes alignment. Takes longer - not advised for training.
            include_intrinsics: whether or not to include camera intrinsics
              in the dataset inputs.
            prediction_is_sequence: If True, metrics and losses will expect
              a sequence of predictions on the first dimension. 
              `self.output_spec` will remain unchanged.
            loss_decay: if `prediction_is_sequence`, this value weights the
              steps exponentially, ignored otherwise.
            download_and_prepare: if True, checks to see if files exist and
              generates them if not. If False, will raise an error if the files
              are not present.
            shuffle_buffer: number of examples used in shuffle buffer. Note
              data stored on disk is shuffled during initial processing, so
              there are no sequences of consecutive even without the per-epoch
              shuffling.
        """
        builder = b.H3mLift(config=config)

        p3f = builder.info.features['pose_3d']
        output_spec = tf.keras.layers.InputSpec(
            shape=p3f.shape, dtype=p3f.dtype)
        
        alignments = (
        (Alignment.OPT_SCALE, Alignment.PROCRUSTES) if include_procrustes
        else (Alignment.OPT_SCALE,))

        origin_index = b.get_origin_indices(builder.builder_config.skeleton_3d)
        # NB: changing reductions to 'mean' below and changing learning rate
        # seems like a good idea, though results are worse. Something to do
        # with Adam?
        loss = PoseLoss(alignment=Alignment.OPT_SCALE,
                    order=1, rescale_labels=True,
                    origin_index=origin_index,
                    reduction='sum', joint_reduction='sum')
        metrics = [
            m.PoseLoss(
                alignment=alignment,
                add_origin=True,
                origin_index=origin_index,
                name='pose_loss_%s' % alignment,
                order=1)
            for alignment in alignments]

        if download_and_prepare:
            builder.download_and_prepare()
            download_and_prepare = False
        intrinsics = builder.load_camera_params()[1]
        
        def map_fn(inputs):
            subject_id, camera_id = (
                inputs[k] for k in ('subject_id', 'camera_id'))
            intr = b.IntrinsicCameraParams(*(
                tf.constant(i)[subject_id, camera_id] for i in intrinsics))
            pose_2d = (inputs['pose_2d'] - intr.center) / intr.focal_length
            pose_3d = inputs['pose_3d'] / SPACE_SCALE
            inputs = dict(pose_2d=pose_2d)
            if include_intrinsics:
                inputs['intrinsics'] = dict(
                    radial_dist_coeff=intr.radial_dist_coeff,
                    tangential_dist_coeff=intr.tangential_dist_coeff)
            return inputs, pose_3d

        if prediction_is_sequence:
            loss = MultiStepLoss(loss, base_ndims=3, loss_decay=loss_decay)
            metrics = [FinalStepMetric(met, base_ndims=3) for met in metrics]
        
        self._prediction_is_sequence = prediction_is_sequence
        super(HpeProblem, self).__init__(
            builder, loss, metrics, output_spec=output_spec,
            map_fn=map_fn,
            as_supervised=False, shuffle_buffer=shuffle_buffer,
            download_and_prepare=download_and_prepare)

    
    def previs(self, inputs, labels, predictions=None):
        """Called before vis."""
        # can be merged into vis in 2.0/when eager mode is norm.
        from ige.hpe.ops import project_points_radial_normalized
        pose_2d = inputs['pose_2d'] if isinstance(inputs, dict) else inputs
        out = dict(pose_2d=pose_2d, labels=labels)
        if predictions is not None:
            predictions, = predictions
            if self._prediction_is_sequence:
                predictions = predictions[-1]  # only plot last
            out['predictions'] = predictions
            if len(predictions.shape) == len(labels.shape) + 1:
                predictions = predictions[-1]
            out['scale_aligned'] = scale_aligned = ops.opt_scale_aligned(
                labels, predictions)
            out['proc_aligned'] = proc_aligned = ops.procrustes_aligned(
                labels, predictions)
            out['scale_err'] = pose_loss(labels, scale_aligned, order=1)
            out['proc_err'] = pose_loss(labels, proc_aligned, order=1)

        return out
    
    def vis(self, pose_2d, labels, predictions=None,
            scale_aligned=None, proc_aligned=None, scale_err=None,
            proc_err=None):
        """Visualize numpy data output of previs for a single example."""
        import matplotlib.pyplot as plt
        from ige.hpe.vis import vis2d
        from ige.hpe.vis import vis3d
        from ige.hpe.vis import show
        config = self._builder.builder_config
        skeleton_3d = config.skeleton_3d
        skeleton_2d = config.skeleton_2d
        vis2d(skeleton_2d, pose_2d, linewidth=1)
        plt.title('Model input')
        if predictions is None:
            vis3d(skeleton_3d, labels, linewidth=1)
            plt.title('Label')
        else:
            for pred, name in (
                    (scale_aligned, 'scale'),
                    (proc_aligned, 'procrustes')):
                ax = vis3d(skeleton_3d, labels, linewidth=1)
                vis3d(
                    skeleton_3d, pred, linestyle='dashed', linewidth=2,
                    ax=ax)
                plt.title(
                    '%s aligned prediction (dashed) vs ground truth' % name)
            print('opt_scale err: %f' % scale_err)
            print('proc err: %f' % proc_err)
        show()