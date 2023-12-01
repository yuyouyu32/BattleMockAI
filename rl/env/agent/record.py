from uuid import uuid4
import pickle
from rl.env.act_obs_space import *

class ReplayRecord():
    # record replay for analysis and draw
    def __init__(self, path, uid):
        self.path = path
        self.reset(uid)

    def reset(self, uid, is_dump = False):
        self.outf = self.path + '/replay_' + str(uid) + '.pickle'
        self.frame_id_list = []
        self.obs_list = []
        self.action_list = []
        self.reward_list = []
        self.is_dump = is_dump

    def update_frame(self, frame_id):
        if not self.is_dump:
            return
        cur_obs_dict = {"main_player": {},
                        "ally":[{} for _ in range(ALLY_NUM)],
                        "enemy":[{} for _ in range(ENEMY_NUM)],
                        "skill":[{} for _ in range(ACTION_NUM)]}
        self.frame_id_list.append(frame_id)
        self.obs_list.append(cur_obs_dict)

    def get_mp_obs_dict(self):
        if not self.is_dump:
            return None
        return self.obs_list[-1]["main_player"]

    def get_ally_obs_dict(self, aid):
        if not self.is_dump:
            return None
        return self.obs_list[-1]["ally"][aid]

    def get_enemy_obs_dict(self, eid):
        if not self.is_dump:
            return None
        return self.obs_list[-1]["enemy"][eid]

    def get_skill_obs_dict(self, sid):
        if not self.is_dump:
            return None
        return self.obs_list[-1]["skill"][sid]

    def add_actions(self, actions):
        if not self.is_dump:
            return None
        return self.action_list.append(actions)

    def add_reward(self, reward_info):
        if not self.is_dump:
            return
        self.reward_list.append(reward_info)

    def dump(self, done):
        if not self.is_dump:
            return
        if done:
            replay = {}
            replay["frame_num"] = len(self.obs_list)
            replay["frame_id_list"] = self.frame_id_list 
            replay["obs_list"] = self.obs_list
            replay["action_list"] = self.action_list
            replay["reward_list"] = self.reward_list
            pickle.dump(replay, open(self.outf, "wb"))
            self.reset(0)

    def size(self):
        return len(self.frame_id_list)
