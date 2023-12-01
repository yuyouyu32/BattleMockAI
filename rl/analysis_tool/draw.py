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
matplotlib.use('Agg')
os.environ["DEPTH_MAP_CONFIG"] = gDepthmapConfigPathDebug 
import rl.depthmap.op as op

def proc(v):
    if type(v) == np.float64:
        v = np.around(v, 3)
    if type(v) == float:
        v = round(v, 2)
    if type(v) == list:
        for idx in range(len(v)): 
            v[idx] = round(v[idx], 2)
    return v

MAX_FRAME = 10000
PNG_SIZE = [1000, 1000] 

def map_crop(input_map, map_crop_path, frame):
    center_x = frame["main_player"]["pos_x"]
    center_y = frame["main_player"]["pos_y"]
    # get BOX
    length = 10000 
    min_x = center_x - length/2
    max_x = center_x + length/2
    min_y = center_y - length/2
    max_y = center_y + length/2
    # process out of range
    min_x = max(min_x, gMapRangeX[0])
    min_y = max(min_y, gMapRangeY[0])
    max_x = min(max_x, gMapRangeX[1])
    max_y = min(max_y, gMapRangeY[1])
    # get pic range
    ratio_x = PNG_SIZE[0]/(gMapRangeX[1] - gMapRangeX[0]) 
    ratio_y = PNG_SIZE[1]/(gMapRangeY[1] - gMapRangeY[0]) 
    pic_min_x = int((min_x - gMapRangeX[0]) * ratio_x) 
    pic_min_y = int((min_y - gMapRangeY[0]) * ratio_y)
    pic_max_x = int((max_x - gMapRangeX[0]) * ratio_x) 
    pic_max_y = int((max_y - gMapRangeY[0]) * ratio_y)
    # to garantee high-precision, derivate map_range from pic_range reversely
    min_x = pic_min_x / ratio_x + gMapRangeX[0]
    min_y = pic_min_y / ratio_y + gMapRangeY[0]
    max_x = pic_max_x / ratio_x + gMapRangeX[0]
    max_y = pic_max_y / ratio_y + gMapRangeY[0]
    map_range = [[min_x, max_x], [min_y, max_y]]
    # crop map
    img = plt.imread(input_map)
    cropped_img = img[pic_min_y:pic_max_y, pic_min_x:pic_max_x]    # (left ,upper, right, lower)
    image.imsave(map_crop_path, cropped_img)
    return map_range

def process_frames(input_file, input_map, output_file, frames, rewards, frame_ids):
    # global define
    r = 20
    
    def draw_player_arrow(mp):
        # player arrow
        x, y, z = mp["pos_x"], mp["pos_y"], mp["pos_z"]
        dx, dy = r*cos(mp["yaw"]*pi/180), r*sin(mp["yaw"]*pi/180)
        ec = '#00CD00'    # green
        fc = "#FF0000"
        axs[1].arrow(x, y, dx, dy, head_width = 60, head_length = 120, fc = fc, ec = ec, fill=False)
    
    def draw_enemy_arrow(enemy):
        x, y, z = enemy["pos_x"], enemy["pos_y"], enemy["pos_z"] 
        dx, dy = r*cos(enemy["yaw"]*pi/180), r*sin(enemy["yaw"]*pi/180)
        visible = enemy["is_visible"]
        ec = '#1E90FF'    # blue 
        fill = False 
        if enemy["is_fire"]:
            fill = True 
        if enemy["is_jump"]:
            ec = '#87CEFA'  # light blue 
        if enemy["is_jet_forward"] or enemy["is_jet_up"]:
            ec = '#FFF68F'   # yellow
        fc = "#FF0000"
        if not visible:
            ec = "#BEBEBE"
            fc = ec
        axs[1].arrow(x, y, dx, dy, head_width = 60, head_length = 120, fc = fc, ec = ec, fill=fill)
        axs[1].text(x, y, str(enemy["character_id"]), color="#FF4040")
    
    def draw_dest_arrow(dest):
        x, y = dest[0], dest[1] 
        ec = '#FF0000'    # red 
        #axs[1].arrow(x, y, dx, dy, head_width = 60, head_length = 120, fc = fc, ec = ec, fill=fill)
        axs[1].scatter(x, y, c='r', marker='o', s = 60)
    
    def draw_map(i, map_range):
        axs[1].cla()

        agent = frames[i]
        mp = agent["main_player"]
        enemies = agent["enemy"]
        action = agent["action"]

        # player arrow
        draw_player_arrow(mp)

        # enemy arrow
        for enemy in enemies:
            draw_enemy_arrow(enemy)

        # destination arrow
        #draw_dest_arrow(mp["dest_pos"])
    
        # ax
        axs[1].set(xlim=(map_range[0][0], map_range[0][1]), ylim=(map_range[1][1], map_range[1][0]))
        axs[1].imshow(img, extent=[map_range[0][0], map_range[0][1], map_range[1][0], map_range[1][1]], origin='lower')
        axs[1].text(map_range[0][0], map_range[1][1], "fid: "+str(fid), color="#F8F8FF")
    
    def draw_table(i):
        max_len = 0
        agent = frames[i]
        reward_info = rewards[i]
        for d in [agent["main_player"], agent["enemy"][0], agent["action"]]:
            max_len = max(max_len, len(d))
        player_col = ["" for i in range(max_len)]
        enemy0_col = ["" for i in range(max_len)]
        act_rew_col = ["" for i in range(max_len)]
        for idx, (k,v) in enumerate(agent["main_player"].items()):
            player_col[idx] = k+": "+str(proc(v))
        for idx, (k,v) in enumerate(agent["enemy"][0].items()):
            enemy0_col[idx] = k+": "+str(proc(v))
        for idx, (k,v) in enumerate(agent["action"].items()):
            if k == "MoveDir":
                v =  MoveDirMeta.reverse(v)
            elif k == "TurnLR":
                v =  TurnLRMeta.reverse(v)
            elif k == "TurnUD":
                v =  TurnUDMeta.reverse(v)
            act_rew_col[idx] = k+": "+str(v)
        act_rew_offset = len(agent["action"])
        act_rew_col[act_rew_offset] = ""
        act_rew_offset += 1
        act_rew_col[act_rew_offset] = ""
        act_rew_offset += 1
        act_rew_col[act_rew_offset] = "==Rew=="
        act_rew_offset += 1
        for idx, (k,v) in enumerate(reward_info.items()):
            act_rew_col[idx+act_rew_offset] = k[:7] + ":" + str(proc(v))

        data = {
                '==player==': player_col,
                '==enemy0==': enemy0_col,
                '==action==': act_rew_col,
                }
        df = pd.DataFrame(data)
        tb = axs[2].table(cellText=df.values,
                          colLabels=df.columns,
                          bbox=[0, 0, 1, 1],)
        tb.auto_set_font_size(True)
        tb.auto_set_column_width(col=list(range(len(df.columns))))
        tb.auto_set_font_size(False)
        tb.set_fontsize(6)
        tb.scale(2, 2)
    
    def draw_depthmap(i):
        pic_path = "/".join(output_file.split("/")[:-2]) + "/tmp/" + str(i) + ".png"
        depthmaps = frames[i]["depthmap"]["depthmap_array"] 
        cv2.imwrite(pic_path, depthmaps)
        depth_img = plt.imread(pic_path)
        axs[0].imshow(depth_img, cmap=plt.cm.gray)
    
    # layout1
    fig, axs = plt.subplots(1, 3, figsize=(16, 12), gridspec_kw={'width_ratios': [1, 2, 1]})
    for i in frame_ids:
        print(i)
        # crop image
        map_crop_path = "/".join(output_file.split("/")[:-2]) + "/tmp/" + str(i) + ".png" 
        map_range = map_crop(input_map, map_crop_path, frames[i])
        img = plt.imread(map_crop_path)

        # draw left sub map
        fid = frames[i]["frame_id"]

        draw_map(i, map_range)
        draw_depthmap(i)

        axs[0].axis('off')
        axs[2].axis('off')
    
        # draw right sub map
        draw_table(i)
    
        plt.subplots_adjust(wspace =0, hspace =0)
        plt.axis('off')
        pic_path = output_file + "_" + str(i) + ".png"
        #fig.tight_layout()
        plt.savefig(pic_path)
    cv2.destroyAllWindows()

MAX_DIST = 6200
def multi_process(input_file, input_map, input_map_crop, output_file, depthmap_type, process_num = 1):
    print(input_file)
    replay = pickle.load(open(input_file, 'rb'))
    print(replay)

    # turn replay to frames
    frames = replay["obs_list"][:-1]   # there is no action at the last frame
    rewards = replay["reward_list"]
    print(len(replay["frame_id_list"]))
    print(len(replay["action_list"]))
    for i in range(len(frames)):
        frames[i]["frame_id"] = replay["frame_id_list"][i] 
        frames[i]["action"] = replay["action_list"][i] 

    if depthmap_type == "layer":
        # process layer depthmap
        def single_depthmap_process(params):
            depthmap_feature = np.array([params])
            depthmaps_flat = Lambda(op.op_depth_map, name="depth_map")(depthmap_feature)    # (bs*depthmap_num) * fnum -> (bs*depthmap_num) * 50
            depthmaps = tf.squeeze(depthmaps_flat).numpy()    # (bs*depthmap_num) * 14400 -> bs * depthmap_num * 90 * 160
            depthmaps = tf.squeeze(depthmaps).numpy()    # 50 
            return depthmaps 
        for i in range(len(frames)):
            depthmap_layers = []
            for st in range(int(len(frames[i]["depthmap"]["depthmap"])/6)):
                params = frames[i]["depthmap"]["depthmap"][st*6:st*6+6]
                depthmap_layer = single_depthmap_process(params)
                depthmap_layers.append(depthmap_layer)
            depthmap_layers = np.array(depthmap_layers)
            depthmap_layers = 1.0*depthmap_layers/MAX_DIST * 255
            depthmaps = np.uint8(depthmap_layers)    # 50 
    elif depthmap_type == "norm":
        # process org depthmap
        for i in range(len(frames)):
            (x,y,z,yaw,pitch,map_id) = frames[i]["depthmap"]["depthmap"][:6]
            z = z - 200 + 180
            depthmap_feature = np.array([[x,y,z,yaw,pitch,map_id]])
            #print(depthmap_feature)
            depthmaps_flat = Lambda(op.op_depth_map, name="depth_map")(depthmap_feature.astype(np.int32))    # (bs*depthmap_num) * fnum -> (bs*depthmap_num) * 14400
            depthmaps = tf.reshape(depthmaps_flat, [-1, 1, 90, 160])    # (bs*depthmap_num) * 14400 -> bs * depthmap_num * 90 * 160
            depthmaps = tf.squeeze(depthmaps).numpy()    # (bs*depthmap_num) * 14400 -> bs * depthmap_num * 90 * 160
            depthmaps = 1.0*depthmaps/MAX_DIST * 255
            depthmaps = np.uint8(tf.squeeze(depthmaps).numpy())    # (bs*depthmap_num) * 14400 -> bs * depthmap_num * 90 * 160
            frames[i]["depthmap"]["depthmap_array"] = depthmaps 

    process_list = []
    frames = frames[:MAX_FRAME]
    for pid in range(process_num):
        frame_ids = [i for i in range(len(frames)) if i%process_num==pid]
        print("start process {}".format(frame_ids))
        process_list.append(Process(target=process_frames, args=(input_file, input_map, output_file, frames, rewards, frame_ids)))
        process_list[-1].start()
    for idx in range(len(process_list)):
        process_list[idx].join()

if __name__ == "__main__":
    input_file = sys.argv[1]
    input_map = sys.argv[2]
    output_file = sys.argv[3]
    depthmap_type = sys.argv[4]

    input_map_crop = "/".join(output_file.split("/")[:-2]) + "/tmp/map.png"

    multi_process(input_file, input_map, input_map_crop, output_file, depthmap_type, 50)
