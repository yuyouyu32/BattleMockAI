from collections import OrderedDict
from math import cos, sin, pi
import random 
from .feature_def import *
from .buff_map import buff_map
from .config import *
from rl.common.common import *
from rl.env.act_obs_space import *

"""
basic meta define
"""
class ContinueFeatureMeta(): 
    def __init__(self, minv, maxv, norm_minv=0.0, norm_maxv=1.0, only_for_analysis = False):
        self.type = "norm"
        self.minv = minv
        self.maxv = maxv
        self.norm_minv = norm_minv
        self.norm_maxv = norm_maxv
        self.norm_range = norm_maxv-norm_minv 
        self.only_for_analysis = only_for_analysis
        self.len = 1

    def norm(self, val):
        if val > self.maxv:
            return self.norm_maxv
        if val < self.minv:
            return self.norm_minv
        return ((val- self.minv) / (self.maxv - self.minv)) * self.norm_range + self.norm_minv

    def process(self, key, feat_dict, fdict_for_analysis = None):
        if fdict_for_analysis != None:
            fdict_for_analysis[key] = feat_dict[key]
            fdict_for_analysis[key+"_norm"] = self.norm(feat_dict[key])
        if self.only_for_analysis:
            return []
        return self.norm(feat_dict[key])
        
class BoolFeatureMeta(): 
    def __init__(self, only_for_analysis = False):
        self.type = "norm"
        self.only_for_analysis = only_for_analysis
        self.len = 1

    def norm(self, val):
        """currently use 0/1 instead of T/F"""
        assert val in (0, 1)
        return val
    
    def process(self, key, feat_dict, fdict_for_analysis = None):
        if fdict_for_analysis != None:
            fdict_for_analysis[key] = feat_dict[key]
            fdict_for_analysis[key+"_norm"] = self.norm(feat_dict[key])
        if self.only_for_analysis:
            return []
        return self.norm(feat_dict[key])
        
class CategoryFeatureMeta(): 
    def __init__(self, category_num, only_for_analysis = False):
        self.type = "norm"
        self.category_num = category_num
        self.only_for_analysis = only_for_analysis
        self.warning = None
        self.len = category_num

    def norm(self, key, val):
        vector = [0 for _ in range(self.category_num)]
        if type(val) != list:    # onehot-category
            val = [val]
        for idx in val: 
            if idx < 0:   # in dislyte data, usually -1 repersent None and not 0 present
                # usually not error
                continue
            #elif idx >= self.category_num:
            #    print("warning! {} out of range {}".format(key, val))
            #    continue
            vector[idx] = 1
        return vector

    def process(self, key, feat_dict, fdict_for_analysis = None):
        try:
            if fdict_for_analysis != None:
                fdict_for_analysis[key] = feat_dict[key]
            if self.only_for_analysis:
                return []
            norm_val =self.norm(key, feat_dict[key])
            #if self.warning:
            #    print(self.warning.format(key))
            return norm_val
        except:
            print('Error', key, feat_dict[key])
        


class SkillCDFeatureMeta(ContinueFeatureMeta): 
    def __init__(self, minv, maxv, only_for_analysis = False):
        super(SkillCDFeatureMeta, self).__init__(minv, maxv, only_for_analysis)
        self.len = 3 

    def process(self, key, feat_dict, fdict_for_analysis = None):
        skill_cds = []
        for skill in feat_dict["skillData"]:
            skill_cds.append(self.norm(skill["sCD"]))
        if fdict_for_analysis != None:
            fdict_for_analysis[key] = skill_cds
        if self.only_for_analysis:
            return []
        return skill_cds

class BidCategoryFeatureMeta(CategoryFeatureMeta): 
    def __init__(self, category_num, only_for_analysis = False):
        super(BidCategoryFeatureMeta, self).__init__(category_num, only_for_analysis)
        self.len = category_num

    def process(self, key, feat_dict, fdict_for_analysis = None):
        if fdict_for_analysis != None:
            fdict_for_analysis[key] = feat_dict[key]
        if self.only_for_analysis:
            return []
        if feat_dict[key] not in buff_map:
            norm_val =self.norm(key, [])
            print("warning! buff id {} not in buff_map".format(feat_dict[key]))
            raise RuntimeError("warning! buff id {} not in buff_map".format(feat_dict[key]))
        else:
            norm_val =self.norm(key, buff_map[feat_dict[key]])
        return norm_val

"""
feature define
type 1: norm original feature, meta func process(key, dict), process dict[key], meta key must equal to raw sbe key
type 2: add customized feature, meta func process(Nil, dict), process dict 
"""
# ============= player features ============
PlayerFeatureDict = OrderedDict()
PlayerFeatureDict["oPosIdx"] = CategoryFeatureMeta(PosNum)
PlayerFeatureDict["pHP"] = ContinueFeatureMeta(-1, HpMax)
PlayerFeatureDict["pMAXHP"] = ContinueFeatureMeta(-1, HpMax)
PlayerFeatureDict["oAP"] = ContinueFeatureMeta(0, ApMAX)
PlayerFeatureDict["pATK"] = ContinueFeatureMeta(0, AtkMAX)
PlayerFeatureDict["pARM"] = ContinueFeatureMeta(0, ArmMax)
PlayerFeatureDict["pSPEED"] = ContinueFeatureMeta(0, SpeedMax)
PlayerFeatureDict["pHIT"] = ContinueFeatureMeta(HitMin, 0)
PlayerFeatureDict["pCRP"] = ContinueFeatureMeta(0, CrpMax)
PlayerFeatureDict["pUCP"] = ContinueFeatureMeta(0, UcpMax)
PlayerFeatureDict["pCRD"] = ContinueFeatureMeta(0, CrdMax)
PlayerFeatureDict["pACC"] = ContinueFeatureMeta(0, AccMax)
PlayerFeatureDict["pRES"] = ContinueFeatureMeta(0, ResMax)
PlayerFeatureDict["sSkillCD"] = SkillCDFeatureMeta(0, MaxCD)
PlayerFeatureDict["oCharTypeID"] = CategoryFeatureMeta(HeroTypeNum)

PlayerFeatureDict["oTotalShield"] = ContinueFeatureMeta(0, ShieldMax)
PlayerFeatureDict["bConfrontation"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bImmunity"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bImprison"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bInvincible"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bProvoke"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bSilent"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bSleep"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bStealth"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bStone"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bStun"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bUnrecoverable"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bCharge"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bChooseskill"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bDokkaebiBuff"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bFafnirimmunity"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bFafnirimprison"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bFafnirstun"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bForget"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bNightmare"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bPunish"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bRecovermaxhp"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bReverse"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["bWeak"] = ContinueFeatureMeta(0, MaxBuffTurns)
PlayerFeatureDict["equipSuitData"] = CategoryFeatureMeta(equipSuitNum)
SinglePlayerFeatDim = sum([v.len for k,v in PlayerFeatureDict.items()])

# ============= ally features ============
AllyFeatureDict = OrderedDict()
AllyFeatureDict["oHeloID"] = CategoryFeatureMeta(HeroNum)
AllyFeatureDict["oPosIdx"] = CategoryFeatureMeta(PosNum)
AllyFeatureDict["pHP"] = ContinueFeatureMeta(-1, HpMax)
AllyFeatureDict["pMAXHP"] = ContinueFeatureMeta(-1, HpMax)
AllyFeatureDict["oAP"] = ContinueFeatureMeta(0, ApMAX)
AllyFeatureDict["pATK"] = ContinueFeatureMeta(0, AtkMAX)
AllyFeatureDict["pARM"] = ContinueFeatureMeta(0, ArmMax)
AllyFeatureDict["pSPEED"] = ContinueFeatureMeta(0, SpeedMax)
AllyFeatureDict["pHIT"] = ContinueFeatureMeta(HitMin, 0)
AllyFeatureDict["pCRP"] = ContinueFeatureMeta(0, CrpMax)
AllyFeatureDict["pUCP"] = ContinueFeatureMeta(0, UcpMax)
AllyFeatureDict["pCRD"] = ContinueFeatureMeta(0, CrdMax)
AllyFeatureDict["pACC"] = ContinueFeatureMeta(0, AccMax)
AllyFeatureDict["pRES"] = ContinueFeatureMeta(0, ResMax)
AllyFeatureDict["sSkillCD"] = SkillCDFeatureMeta(0, MaxCD)
AllyFeatureDict["oCharTypeID"] = CategoryFeatureMeta(HeroTypeNum)
AllyFeatureDict["oTotalShield"] = ContinueFeatureMeta(0, ShieldMax)

AllyFeatureDict["bConfrontation"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bImmunity"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bImprison"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bInvincible"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bProvoke"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bSilent"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bSleep"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bStealth"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bStone"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bStun"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bUnrecoverable"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bCharge"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bChooseskill"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bDokkaebiBuff"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bFafnirimmunity"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bFafnirimprison"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bFafnirstun"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bForget"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bNightmare"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bPunish"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bRecovermaxhp"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bReverse"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["bWeak"] = ContinueFeatureMeta(0, MaxBuffTurns)
AllyFeatureDict["equipSuitData"] = CategoryFeatureMeta(equipSuitNum)
SingleAllyFeatDim = sum([v.len for k,v in AllyFeatureDict.items()])

# ============= enemy features ============
EnemyFeatureDict = OrderedDict()
EnemyFeatureDict["oHeloID"] = CategoryFeatureMeta(HeroNum)
EnemyFeatureDict["oPosIdx"] = CategoryFeatureMeta(PosNum)
EnemyFeatureDict["pHP"] = ContinueFeatureMeta(-1, HpMax)
EnemyFeatureDict["pMAXHP"] = ContinueFeatureMeta(-1, HpMax)
EnemyFeatureDict["oAP"] = ContinueFeatureMeta(0, ApMAX)
EnemyFeatureDict["pATK"] = ContinueFeatureMeta(0, AtkMAX)
EnemyFeatureDict["pARM"] = ContinueFeatureMeta(0, ArmMax)
EnemyFeatureDict["pSPEED"] = ContinueFeatureMeta(0, SpeedMax)
EnemyFeatureDict["pHIT"] = ContinueFeatureMeta(HitMin, 0)
EnemyFeatureDict["pCRP"] = ContinueFeatureMeta(0, CrpMax)
EnemyFeatureDict["pUCP"] = ContinueFeatureMeta(0, UcpMax)
EnemyFeatureDict["pCRD"] = ContinueFeatureMeta(0, CrdMax)
EnemyFeatureDict["pACC"] = ContinueFeatureMeta(0, AccMax)
EnemyFeatureDict["pRES"] = ContinueFeatureMeta(0, ResMax)
EnemyFeatureDict["sSkillCD"] = SkillCDFeatureMeta(0, MaxCD)
EnemyFeatureDict["oCharTypeID"] = CategoryFeatureMeta(HeroTypeNum)
EnemyFeatureDict["oTotalShield"] = ContinueFeatureMeta(0, ShieldMax)

EnemyFeatureDict["bConfrontation"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bImmunity"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bImprison"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bInvincible"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bProvoke"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bSilent"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bSleep"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bStealth"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bStone"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bStun"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bUnrecoverable"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bCharge"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bChooseskill"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bDokkaebiBuff"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bFafnirimmunity"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bFafnirimprison"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bFafnirstun"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bForget"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bNightmare"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bPunish"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bRecovermaxhp"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bReverse"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["bWeak"] = ContinueFeatureMeta(0, MaxBuffTurns)
EnemyFeatureDict["equipSuitData"] = CategoryFeatureMeta(equipSuitNum)
SingleEnemyFeatDim = sum([v.len for k,v in EnemyFeatureDict.items()])
# --- customized feature ----

# ============= skill features ============
SkillFeatureDict = OrderedDict()
SkillFeatureDict["sSkillType"] = CategoryFeatureMeta(SkillTypeNum)
SkillFeatureDict["sResLevel"] = ContinueFeatureMeta(0, ResLevelMax)
SkillFeatureDict["sAwakenType"] = CategoryFeatureMeta(AwakenTypeNum)
SkillFeatureDict["sSkillLevel"] = ContinueFeatureMeta(0, SkillLevelMax)
SkillFeatureDict["sOperationType"] = CategoryFeatureMeta(OperationTypeNum)
SkillFeatureDict["sTargetCamp"] = CategoryFeatureMeta(TargetCampNum)
SkillFeatureDict["sSkillAIType"] = CategoryFeatureMeta(SkillAITypeNum)
SingleSkillFeatDim = sum([v.len for k,v in SkillFeatureDict.items()])

# ============= buff features ============
BuffFeatureDict = OrderedDict()
BuffFeatureDict["bCfgID"] = BidCategoryFeatureMeta(TotalBuffNum)
BuffFeatureDict["bTime"] = ContinueFeatureMeta(0, MaxBuffTurns)
BuffFeatureDict["bCfgIsClear"] = BoolFeatureMeta()
BuffFeatureDict["bCfgIsTrans"] = BoolFeatureMeta()
BuffFeatureDict["bCfgNature"] = CategoryFeatureMeta(BuffTypeNum)
SingleBuffDim = sum([v.len for k,v in BuffFeatureDict.items()])
import json
# Store variables in a dictionary
data = {
    "MP_FEAT_NUM": SinglePlayerFeatDim,
    "ALLY_FEAT_NUM": SingleAllyFeatDim,
    "ENEMY_FEAT_NUM": SingleEnemyFeatDim,
    "SKILL_FEAT_NUM": SingleSkillFeatDim,
    "BUFF_FEAT_NUM": SingleBuffDim
}
print(data)
# Convert dictionary to JSON and save to a file
with open("rl/env/act_obs_space.json", "w") as json_file:
    json.dump(data, json_file, indent=4)