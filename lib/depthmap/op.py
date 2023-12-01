from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

from tensorflow.python.framework import ops

ops_lib = load_library.load_op_library(
    resource_loader.get_path_to_datafile('libtf_depth_map.so'))
op_depth_map = ops_lib.depth_map
op_ray_detect = ops_lib.ray_detect

@ops.RegisterGradient("DepthMap")
def _DepthMapGrad(op_depth_map, grad):
    del op_depth_map, grad
    return [None]

@ops.RegisterGradient("RayDetect")
def _RayDetectGrad(op_ray_detect, grad):
    del op_ray_detect, grad
    return [None]