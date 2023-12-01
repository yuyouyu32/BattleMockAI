import numpy as np
import os
import pandas as pd
import cv2
import time
import pickle
import sbe
import sys
from matplotlib import image
from multiprocessing import Process
from math import pi,cos,sin
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Lambda

from rl.env.agent.config import *

MAX_DIST = 6200
def multi_process(input_file, input_map, input_map_crop, output_file, depthmap_type, process_num = 1):
    print(input_file)
    replay = pickle.load(open(input_file, 'rb'))

    # turn replay to frames
    frames = replay["obs_list"][:-1]   # there is no action at the last frame
    rewards = replay["reward_list"]
    v = 0.0
    for i in range(len(frames)):
        #print(rewards[i]["damage_rew"])
        v = max(v, rewards[i]["damage_rew"])
    print(v)
    print(v/len(frames))

if __name__ == "__main__":
    input_file = sys.argv[1]
    input_map = sys.argv[2]
    output_file = sys.argv[3]
    depthmap_type = sys.argv[4]

    input_map_crop = "/".join(output_file.split("/")[:-2]) + "/tmp/map.png"

    multi_process(input_file, input_map, input_map_crop, output_file, depthmap_type, 1)
