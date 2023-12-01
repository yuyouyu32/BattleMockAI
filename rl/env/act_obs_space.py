from gym.spaces import Discrete, Box, Dict, Dict
from collections import deque, OrderedDict
from rl.env.agent.buff_map import buff_map
import json


'''
======== 1. observation space ========
'''
HeroNum = 132
TotalSkillNum = HeroNum * 3
TotalBuffNum = len(buff_map) + 1
ACTION_NUM = 3
ALLY_NUM = 4
ENEMY_NUM = 5
BUFF_NUM = 12


import json

# Read data from JSON file
with open("rl/env/act_obs_space.json", "r") as json_file:
    data = json.load(json_file)

# Assign values to variables
MP_FEAT_NUM = data["MP_FEAT_NUM"]
ALLY_FEAT_NUM = data["ALLY_FEAT_NUM"]
ENEMY_FEAT_NUM = data["ENEMY_FEAT_NUM"]
SKILL_FEAT_NUM = data["SKILL_FEAT_NUM"]
BUFF_FEAT_NUM = data["BUFF_FEAT_NUM"]

# keep both ally_mask and skill_t_ally_masks, for furture optimization 
OBS_SPACE = Dict(OrderedDict({'hero_id':Box(low=0, high=HeroNum, shape=(1,)),
                  'mp_feature':Box(low=0, high=1, shape=(MP_FEAT_NUM,)),
                  'mp_buff':Box(low=0, high=1, shape=(BUFF_NUM * BUFF_FEAT_NUM,)),
                  'ally_feature':Box(low=0, high=1, shape=(ALLY_NUM*ALLY_FEAT_NUM,)),
                  'ally_buff':Box(low=0, high=1, shape=(ALLY_NUM*BUFF_NUM*BUFF_FEAT_NUM,)),
                  'enemy_feature':Box(low=0, high=1, shape=(ENEMY_NUM*ENEMY_FEAT_NUM,)),
                  'enemy_buff':Box(low=0, high=1, shape=(ENEMY_NUM*BUFF_NUM*BUFF_FEAT_NUM,)),
                  'skill_feature':Box(low=0, high=1, shape=(ACTION_NUM*SKILL_FEAT_NUM,)),
                  'ally_mask':Box(low=0, high=1, shape=(ALLY_NUM+1,)),
                  'enemy_mask':Box(low=0, high=1, shape=(ENEMY_NUM,)),
                  'skill_mask':Box(low=0, high=1, shape=(ACTION_NUM,)),
                  'skill_t_ally_masks':Box(low=0, high=1, shape=(ACTION_NUM,)),
                  'skill_t_enemy_masks':Box(low=0, high=1, shape=(ACTION_NUM,)),
                  'skill_slot_label':Box(low=0, high=3, shape=(1,))}))

# for k,v in OBS_SPACE.items():
#     print("{} : {}".format(k, v.shape))

'''
======== 2. observation space ========
'''
ACTION_SPACE = Dict(OrderedDict({
    'Skill': Discrete(ACTION_NUM),
    'AllySelect': Discrete(ALLY_NUM+1),
    'EnemySelect': Discrete(ENEMY_NUM),
    }))

# for sl
ActionName = ['Skill', 'AllySelect', 'EnemySelect']
ActionDim = [ACTION_NUM, ALLY_NUM+1, ENEMY_NUM]

'''
========  3. config =========
'''
# tablet feature config
# (1) hero type
#HERO_TYPES = [1,2,3,4,5,1,2,3] * 10
#assert HeroNum == len(HERO_TYPES)

# (2) skill type
#TotalSkillNum = HeroNum * 3
#SKILL_TYPES = [2, 1, 2, 1, 1, 0, 1, 2] * 3 * 10
#assert TotalSkillNum == len(SKILL_TYPES)

# (3) skill target type
# id of SKILL_TARGET_TYPES == 3*hero_id + hero_skill_id
#SKILL_TARGET_TYPES = [0, 1, 2, 0, 1, 0, 1, 2] * 3 * 10   # 0 None, 1 choose ally, 2 choose enemy 
#assert HeroNum*3 == len(SKILL_TARGET_TYPES) == TotalSkillNum

# (4) buff type
#BuffTypeNum = 10
#BuffTypeDict = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#assert BuffTypeNum == len(BuffTypeDict)

# other feature config
PosNum = 11
HpMax = 100000
ShieldMax = 100000 / 2
ApMAX = 1000
AtkMAX = 10000
ArmMax = 10000
SpeedMax = 500
HitMin = -1   
#BlpMax = 1   # !
CrpMax = 2.5
UcpMax = 0.5
CrdMax = 5
AccMax = 1.4
ResMax = 2
MaxCD = 10
HeroTypeNum = 6

SkillTypeNum = 3
ResLevelMax = 7
AwakenTypeNum = 4
SkillLevelMax = 10
SkillAITypeNum = 23
OperationTypeNum = 5
TargetCampNum = 3

MaxBuffTurns = 15
BuffTypeNum = 5

equipSuitNum = 26