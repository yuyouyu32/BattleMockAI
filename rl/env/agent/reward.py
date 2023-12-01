import sys
import math
from rl.common.common import *

def kill_rew(frame_info, prev_frame_info, rew_info, stats_info):
    rew = 0.0
    if not prev_frame_info:
        return rew

    kill_num = 0
    for enemy1, enemy0 in zip(frame_info["enemy"], prev_frame_info["enemy"]):
        if enemy1["hp"] == 0 and enemy0["hp"] != 0:
            kill_num += 1
    stats_info["kill"] = rew 
    rew_info[sys._getframe().f_code.co_name] = kill_num
    return rew

def win_rew(frame_info, prev_frame_info, rew_info, stats_info):
    win = 1
    for enemy in frame_info["enemy"]:
        if enemy["hp"] != 1:
            win = 0
    rew = 8 * win 
    stats_info["win"] = rew  
    rew_info[sys._getframe().f_code.co_name] = rew
    return rew

def reward_process(game_num, frame_info, hist, stats_info):
    rew_sum = 0.0
    rew_info = {}
    prev_frame_info = None
    if len(hist) > 0:
        prev_frame_info = hist[-1]

    rew_sum += kill_rew(frame_info, prev_frame_info, rew_info, stats_info)
    rew_sum += win_rew(frame_info, prev_frame_info, rew_info, stats_info)

    stats_info["rew_sum"] = rew_sum

    return rew_sum, rew_info
