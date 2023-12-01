import gym
from gym.spaces import Discrete, Box, Dict, Tuple
import numpy as np
import os
import random
import time
from multiprocessing import  Process, Queue
from multiprocessing import shared_memory

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import override
from rl.env.config import gAgentNum
from rl.env.agent.agent import Agent  
from rl.common.logger import logger
from .act_obs_space import OBS_SPACE, ACTION_SPACE

import numpy as np
import time
import sys

class PackDislyteEnv(MultiAgentEnv):
    def __init__(self, config):
        self.worker_index = config.worker_index
        self.vector_index = config.vector_index
        self.env_id = self.worker_index * 10000 + self.vector_index

        self.action_space = ACTION_SPACE
        self.observation_space = OBS_SPACE

        self.agent_num = gAgentNum 
        self.agents = [Agent(self.env_id, config) for _ in range(self.agent_num)]
        self.cur_job = None
        self.pre_t = 0
        self.pre_frame_id = -1

    def reset(self):
        # reset all agents
        agents_frame_info = self.client({}, reset = True)
        obs_dict = {}
        for aid in range(self.agent_num):
            self.agents[aid].reset()
            obs, _, _, _ = self.agents[aid].process_obs_rew(agents_frame_info[aid], [])
            obs_dict[aid] = obs
        return obs_dict

    def step(self, action_dict):
        t1 = time.time()

        #print("======== get action ======")
        #print(action_dict)

        # 1. record actions
        for idx in range(self.agent_num):
            if idx not in action_dict: # sub-env is done
                continue
            self.agents[idx].process_action(action_dict[idx]) 
        t2 = time.time()

        # 2. interact with game
        agents_frame_info = self.client(action_dict)
        t3 = time.time()

        # 3. process obs & reward
        obs_dict, rew_dict, done_dict, info_dict = {}, {}, {}, {}
        done_dict["__all__"] = True 
        for idx in range(self.agent_num):
            if idx not in action_dict: # sub-env is done
                continue
            obs, rew, done, info = self.agents[idx].process_obs_rew(agents_frame_info[idx], action_dict[idx])   # process_obs_rew(st, at+1)
            obs_dict[idx] = obs
            rew_dict[idx] = rew
            done_dict[idx] = done
            info_dict[idx] = info
            done_dict["__all__"] = done_dict["__all__"] and done 
            #print("======= set obs ==========")
            #print(obs)

        # debug logging
        t4 = time.time()
        if random.random() < 0.01:
            logger.info("{} rllib step time: {}".format(self.vector_index, t1-self.pre_t))
            logger.info("{} action process time: {}".format(self.vector_index, t2-t1))
            logger.info("{} env interact time: {}".format(self.vector_index, t3-t2))
            logger.info("{} obs_rew process time: {}".format(self.vector_index, t4-t3))
        self.pre_t = time.time()
        return obs_dict, rew_dict, done_dict, info_dict

    def client(self, action_dict, reset = False):
        # set reset
        if reset:
            logger.debug("reset!")
            self.doReset()
        # set action
        elif len(action_dict) != 0:   # not first frame
            # action_dict[agent_idx] = action, action[action_name] = val
            self.setAction(self.cur_job, action_dict)

        # get state
        states = self.getState()

        cur_frame_id = states[0]["frame_counter"]
        if self.pre_frame_id != cur_frame_id - 1:
            logger.error("agent {} frame non-continous ! cur_frame_id {}, pre_frame_id {}".format(self.vector_index, cur_frame_id, self.pre_frame_id))
        self.pre_frame_id = cur_frame_id

        # fill in frame_info
        frame_info = []
        for i in range(gAgentNum):
            frame_info.append(self.state_process(states[i])) 
        return frame_info 

    def state_process(self, state):
        return state

    # -------------- game interface ------------------
    def setAction(self, job, actions):
        return
    
    def getState(self):
        states = []
        frame_counter = 1
        is_done = False
        for i in range(gAgentNum):
            mp = {"hero_id": 1, "hp":1, "type":1, "skill_cool_down":[0, 0, 0], "alive": 1, "buff_id":[2,3]}
            ally = {"hero_id": 2, "hp":50, "type":2, "skill_cool_down":[0, 0, 0], "alive": 1, "buff_id":[]}
            enemy = {"hero_id": 3, "hp":90, "type":3, "skill_cool_down":[0, 0, 0], "alive": 1, "buff_id":[2,3]}
            d = {"frame_counter": frame_counter,
                 "is_done": is_done, 
                 "mp": mp, 
                 "ally": [ally, ally, ally, ally],
                 "enemy": [enemy, enemy, enemy, enemy, enemy]}
            states.append(d)
        return states
    
    def doReset(self):
        return
