from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ["DEPTH_MAP_CONFIG"] = "./config/depth_map_config_train.yaml"
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops

ops_lib = load_library.load_op_library(
    resource_loader.get_path_to_datafile('lib/depthmap/libtf_depth_map.so'))
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

import numpy as np
from tensorflow.python.framework import ops
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

if __name__ == '__main__':
  param = [[93.42151, 231.2633, 9804.1748046875, 0.0, 0, 1] for _ in range(50 * 12800)]

  out = op_depth_map(param)
  out = out[0]
  print(out)

  out = op_ray_detect(param)

  print(out)
