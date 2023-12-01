import math
import os
import shutil

def check_and_clean(path, restore=False):
    if not restore or not os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

def calc_ver_angle(x1,y1,z1,x2,y2,z2):
    a = math.sqrt((x2-x1)**2 +(y2-y1)**2)
    angle = math.atan2(z2-z1, a)
    return angle

def calc_hor_angle(x1,y1,x2,y2): 
    angle=0
    d_y= y2-y1
    d_x= x2-x1
    radian = math.atan2(d_y, d_x)
    angle = radian * 180 / math.pi
    if angle < 0:
        angle += 360
    return angle

def cal_rel_hor_angle(mp_x, mp_y, mp_z, mp_yaw, x, y, z):
    abs_hor_angle = calc_hor_angle(mp_x,mp_y,x,y)
    rel_hor_angle = abs_hor_angle - mp_yaw
    if rel_hor_angle > 180:
        rel_hor_angle -= 360
    if rel_hor_angle < -180:
        rel_hor_angle += 360
    return rel_hor_angle

def calc_dist(agent_dict_1, agent_dict_2):
    sx,sy,sz = (agent_dict_1["pos_x"], agent_dict_1["pos_y"], agent_dict_1["pos_z"])
    tx,ty,tz = (agent_dict_2["pos_x"], agent_dict_2["pos_y"], agent_dict_2["pos_z"])
    distance = math.sqrt((sx-tx)**2+(sy-ty)**2+(sz-tz)**2)
    return distance

def calc_dist_tuple(pos1, pos2):
    sx,sy,sz = (pos1[0], pos1[1], pos1[2])
    tx,ty,tz = (pos2[0], pos2[1], pos2[2])
    distance = math.sqrt((sx-tx)**2+(sy-ty)**2+(sz-tz)**2)
    return distance

