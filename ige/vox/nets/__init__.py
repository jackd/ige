from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gin
from ige.vox.nets import image_nets
from ige.vox.nets import decoders


@gin.configurable
def get_voxel_inference(
        inputs, output_spec, training=None,
        feature_extractor=image_nets.mobilenet_features,
        voxel_decoder=decoders.decode_conv):
    features = feature_extractor(inputs, output_spec, training)
    voxels = voxel_decoder(features, output_spec, training)
    return voxels