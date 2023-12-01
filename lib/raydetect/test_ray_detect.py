from ctypes import *

import os
os.environ["DEPTH_MAP_CONFIG"] = "config/depth_map_config_train.yaml"

ray_detect_module = CDLL("./lib/raydetect/libutils_dynamic.so", RTLD_GLOBAL)
ray_detect = ray_detect_module.ray_detect_mesh
ray_detect.argtypes = [c_float, c_float, c_float, c_float, c_float, c_float, c_float]
ray_detect.restype = c_int
# x1 y1 z1 x2 y2 z2 map_id
# r = 0 means visible, r = 1 means invisible
r = ray_detect(93.42151, 231.2633, 9804.1748046875, 93.42151, -231.2633, 9804.1748046875, 1);
#r = ray_detect(3000, 3000 ,3000, -3000, -3000, -3000, 1);

print(r);
