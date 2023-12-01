import numpy as np
import os
from ctypes import *
from .config import *
from .feature_def import * 
from rl.common.logger import Logger
from rl.common.common import * 

class FeatureProcess():
    def __init__(self, logger = None):
        self.logger = logger

    def process_features(self, name, FeatureMeta, raw_feat_dict, frame_info=None, fdict_for_analysis=None):
        features = []
        for key, meta in FeatureMeta.items():
            # norm_val could be []. means meta process only product analysis dict, not train features
            if meta.type == "norm":
                norm_val = meta.process(key, raw_feat_dict, fdict_for_analysis=fdict_for_analysis)
            elif meta.type == "cross":
                norm_val = meta.process(key, raw_feat_dict, frame_info, fdict_for_analysis=fdict_for_analysis)
            # add to list 
            if type(norm_val) == list:
                feature_len = len(norm_val)
                features.extend(norm_val)
            else:
                feature_len = 1
                features.append(norm_val)
            # print feature position
            if feature_len != 0:
                self.logger.info("{} key {} val {}, index:{}, len:{}".format(name, key, norm_val, len(features), feature_len))
        return features

    def player_process(self, mp, frame_info, fdict_for_analysis = None):
        features = self.process_features("Player", PlayerFeatureDict, mp, frame_info,
                                         fdict_for_analysis=fdict_for_analysis)
        buffs = []
        for idx, buff in enumerate(mp["buffData"]):
            if idx < mp["bBuffNum"]:
                one_buff = self.process_features("Buff", BuffFeatureDict, buff, frame_info,
                                         fdict_for_analysis=fdict_for_analysis)
            else:
                one_buff = [0 for _ in range(SingleBuffDim)]
            buffs.extend(one_buff)
        skill_mask, skill_t_ally_mask, skill_t_enemy_mask = [], [], []
        assert len(mp["skillData"]) == ACTION_NUM
        for skill in mp["skillData"]:
            skill_mask.append(int(skill["sCD"]==0))
            skill_t_ally_mask.append(int(skill["sTargetCamp"]==2 and skill["sOperationType"] in [1, 4]))
            skill_t_enemy_mask.append(int(skill["sTargetCamp"]==1 and skill["sOperationType"] in [1, 4]))
        fdict_for_analysis["oHeloID"] = mp["oHeloID"]
        fdict_for_analysis["skill_mask"] = skill_mask
        fdict_for_analysis["skill_t_ally_mask"] = skill_t_ally_mask
        fdict_for_analysis["skill_t_enemy_mask"] = skill_t_enemy_mask
        fdict_for_analysis["sSkillID"] = [t["sSkillID"] for t in mp["skillData"]]
        return np.array(features), np.array(buffs), skill_mask, skill_t_ally_mask, skill_t_enemy_mask

    def ally_process(self, ally, frame_info, fdict_for_analysis = None):
        # main process
        features = self.process_features("Ally", AllyFeatureDict, ally, frame_info, 
                                         fdict_for_analysis=fdict_for_analysis)
        buffs = []
        for idx, buff in enumerate(ally["buffData"]):
            if idx < ally["bBuffNum"]:
                one_buff = self.process_features("Buff", BuffFeatureDict, buff, frame_info,
                                             fdict_for_analysis=fdict_for_analysis)
            else:
                one_buff = [0 for _ in range(SingleBuffDim)]
            buffs.extend(one_buff)
        ally_mask = int(ally["pHP"] >= 0)
        return np.array(features), np.array(buffs), ally_mask

    def enemy_process(self, enemy, frame_info, fdict_for_analysis = None):
        # main process
        features = self.process_features("Enemy", EnemyFeatureDict, enemy, frame_info, 
                                         fdict_for_analysis=fdict_for_analysis)
        buffs = []
        for idx, buff in enumerate(enemy["buffData"]):
            if idx < enemy["bBuffNum"]:
                one_buff = self.process_features("Buff", BuffFeatureDict, buff, frame_info,
                                         fdict_for_analysis=fdict_for_analysis)
            else:
                one_buff = [0 for _ in range(SingleBuffDim)]
            buffs.extend(one_buff)
        enemy_mask = int(enemy["pHP"] > 0)
        return np.array(features), np.array(buffs), enemy_mask

    def skill_process(self, skill, frame_info, fdict_for_analysis = None):
        # main process
        features = self.process_features("Skill", SkillFeatureDict, skill, frame_info, 
                                         fdict_for_analysis=fdict_for_analysis)
        return np.array(features)
