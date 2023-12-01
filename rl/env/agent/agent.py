import numpy as np
import logging
import os
import time
import random
import multiprocessing
from collections import deque, OrderedDict
from .feature_process import FeatureProcess
from .feature_def import SingleEnemyFeatDim, SingleAllyFeatDim, SinglePlayerFeatDim, SingleBuffDim
from .config import *
from .reward import reward_process 
from .record import ReplayRecord
from rl.env.act_obs_space import *
from rl.common.logger import Logger
from uuid import uuid4

class Agent():
    def __init__(self, env_id, config, fname = None):
        self.uid = str(uuid4())
        self.logger = Logger(log_level = logging.ERROR)
        self.config = config
        self.fp = FeatureProcess(self.logger)
        self.fname = fname
        self.env_id = env_id
        self.replay_record = ReplayRecord(gRecordPath, self.uid)    # full battle info record, for debug 
        self.hist = deque(maxlen=5)

        self.game_num = 0
        self.reset()

    def reset(self):
        self.uid = str(uuid4())
        self.frame_id = 0
        if self.fname: # for sl analysis
            self.replay_record.reset(self.fname, random.random() < 1)
        else:
            self.replay_record.reset(self.uid, random.random() < 1)

    def process_obs(self, frame_info, use_skill_label = False, for_inference = False):
        self.frame_id += 1
        mp = frame_info["MainCharState"][0]
        allies = frame_info["TeamsCharState"]
        enemies = frame_info["EnemiesCharState"]
        skills = frame_info["MainCharState"][0]["skillData"]
        # proprecess feature 
        mp_feature, mp_buff, skill_mask, self.skill_t_ally_masks, self.skill_t_enemy_masks = self.fp.player_process(mp, frame_info, self.replay_record.get_mp_obs_dict())   # fnum
        mp_feat_num = mp_feature.shape
        ally_feature = np.array([])   # [ally_num * fnum, ]
        ally_buff = np.array([])   # [ally_num * buff_num * fnum, ]
        enemy_feature = np.array([])   # [enemy_num * fnum, ]
        enemy_buff = np.array([])   # [enemy_num * buff_num * fnum, ]
        skill_feature = np.array([])   # [action_num * fnum, ]
        ally_mask = [1]   # [ally_num=4 + mp_num=1]
        enemy_mask = []   # [enemy_num=5]
        for idx in range(ALLY_NUM):
            if idx < len(allies) and allies[idx]["oHeloID"] != -1:
                ally = allies[idx]
                single_ally_feature, single_ally_buff, single_ally_mask = self.fp.ally_process(ally, frame_info, self.replay_record.get_ally_obs_dict(idx))
                #print("alive {}".format(single_ally_feature[:HeroNum]))
            else:
                single_ally_feature, single_ally_buff, single_ally_mask  = np.zeros(SingleAllyFeatDim), np.zeros(SingleBuffDim*BUFF_NUM), 0
            ally_feature = np.concatenate((ally_feature, single_ally_feature), axis=0)
            ally_buff = np.concatenate((ally_buff, single_ally_buff), axis=0)
            ally_mask.append(single_ally_mask)
        for idx in range(ENEMY_NUM):
            if idx < len(enemies) and enemies[idx]["oHeloID"] != -1:
                enemy = enemies[idx]
                single_enemy_feature, single_enemy_buff, single_enemy_mask = self.fp.enemy_process(enemy, frame_info, self.replay_record.get_enemy_obs_dict(idx))
            else:
                single_enemy_feature, single_enemy_buff, single_enemy_mask = np.zeros(SingleEnemyFeatDim), np.zeros(SingleBuffDim*BUFF_NUM), 0
            enemy_feature = np.concatenate((enemy_feature, single_enemy_feature), axis=0)
            enemy_buff = np.concatenate((enemy_buff, single_enemy_buff), axis=0)
            enemy_mask.append(single_enemy_mask)
        for idx in range(ACTION_NUM):
            skill = skills[idx]
            single_skill_feature = self.fp.skill_process(skill, frame_info, self.replay_record.get_skill_obs_dict(idx))
            skill_feature = np.concatenate((skill_feature, single_skill_feature), axis=0)

        if for_inference:
            obs_flat = np.concatenate([np.array([mp["oHeloID"]], dtype=np.float32), 
                                       np.array(mp_feature, dtype=np.float32),
                                       np.array(mp_buff, dtype=np.float32),
                                       np.array(ally_feature, dtype=np.float32),
                                       np.array(ally_buff, dtype=np.float32),
                                       np.array(enemy_feature, dtype=np.float32),
                                       np.array(enemy_buff, dtype=np.float32),
                                       np.array(skill_feature, dtype=np.float32),
                                       np.array(ally_mask, dtype=np.float32),
                                       np.array(enemy_mask, dtype=np.float32),
                                       np.array(skill_mask, dtype=np.float32),
                                       np.array(self.skill_t_ally_masks, dtype=np.float32),
                                       np.array(self.skill_t_enemy_masks, dtype=np.float32),
                                       np.array([0], dtype=np.float32)])
            return obs_flat, self.skill_t_ally_masks, self.skill_t_enemy_masks
        else:
            obs = OrderedDict({"hero_id": np.array([mp["oHeloID"]], dtype=np.float32), 
                   "mp_feature": np.array(mp_feature, dtype=np.float32),
                   "mp_buff": np.array(mp_buff, dtype=np.float32),
                   "ally_feature": np.array(ally_feature, dtype=np.float32),
                   "ally_buff": np.array(ally_buff, dtype=np.float32),
                   "enemy_feature": np.array(enemy_feature, dtype=np.float32),
                   "enemy_buff": np.array(enemy_buff, dtype=np.float32),
                   "skill_feature": np.array(skill_feature, dtype=np.float32),
                   "ally_mask": np.array(ally_mask, dtype=np.float32),
                   "enemy_mask": np.array(enemy_mask, dtype=np.float32),
                   "skill_mask": np.array(skill_mask, dtype=np.float32),
                   "skill_t_ally_masks": np.array(self.skill_t_ally_masks, dtype=np.float32),
                   "skill_t_enemy_masks": np.array(self.skill_t_enemy_masks, dtype=np.float32),
                   "skill_slot_label": np.array([0], dtype=np.float32),
                   })
            if use_skill_label:
                obs["skill_slot_label"] = np.array([frame_info["PlayerAction"]["aSkillSlot"]%10-1], dtype=np.float32)
            return obs

    def process_action(self, frame):
        """
        frame -> action_list
        """
        target = frame["PlayerAction"]["aTarget"] - 1
        if target < 5:
            skill_target_ally = target
            skill_target_enemy = 0   # should be masked
        else:
            skill_target_ally = 0   # should be masked
            skill_target_enemy = target-5
        actions = [frame["PlayerAction"]["aSkillSlot"]%10-1, skill_target_ally, skill_target_enemy]
        self.replay_record.add_actions(actions)
        return actions

    def reverse_action(self, actions, skill_t_ally_masks=None, skill_t_enemy_masks=None):
        """
        return action_list
        """
        if not skill_t_ally_masks or not skill_t_enemy_masks:
            skill_t_ally_masks = self.skill_t_ally_masks
            skill_t_enemy_masks = self.skill_t_enemy_masks
        skill_pred, ally_selected_pred, enemy_selected_pred = actions
        skill_select = skill_pred+1
        need_select_ally = skill_t_ally_masks[skill_pred] != 0
        need_select_enemy = skill_t_enemy_masks[skill_pred] != 0
        if need_select_ally: 
            ally_select = ally_selected_pred + 1
        else:
            ally_select = -1
        if need_select_enemy: 
            enemy_select = enemy_selected_pred + 1 + 5
        else:
            enemy_select = -1
        return skill_select, ally_select, enemy_select 

    def reverse_action2(self, frame, actions, skill_t_ally_masks=None, skill_t_enemy_masks=None):
        """
        add action_info to frame
        """
        if not skill_t_ally_masks or not skill_t_enemy_masks:
            skill_t_ally_masks = self.skill_t_ally_masks
            skill_t_enemy_masks = self.skill_t_enemy_masks
        skill_pred, ally_selected_pred, enemy_selected_pred = actions
        frame["PlayerAction"]["aSkillSlot"] = skill_pred+1
        need_select_ally = skill_t_ally_masks[skill_pred] != 0
        need_select_enemy = skill_t_enemy_masks[skill_pred] != 0
        if need_select_ally: 
            frame["PlayerAction"]["aTarget"] = ally_selected_pred + 1
        if need_select_enemy: 
            frame["PlayerAction"]["aTarget"] = enemy_selected_pred + 1 + 5
        info = "ally_mask: {}  enemy_mask: {}\n skill_pred: {} mask {}_{}\n target: {}".format(skill_t_ally_masks, skill_t_enemy_masks, skill_pred, skill_t_ally_masks[skill_pred], skill_t_enemy_masks[skill_pred], frame["PlayerAction"]["aTarget"])
        return need_select_ally, need_select_enemy, info
