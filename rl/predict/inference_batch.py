import logging
import numpy as np
import time
import math
import torch
import torchvision
import multiprocessing
from rl.env.act_obs_space import *
from rl.env.agent.agent import Agent
from rl.train.config import *
from rl.train.model import SLModel 
from rl.common.logger import Logger

logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('torchvision').setLevel(logging.ERROR)

def handle_requests(input_queue, output_queue):
    fake_agent = Agent(0, {})
    fake_agent.replay_record.update_frame(0)  # fake
    while True:
        (idxs, frames) = input_queue.get()
        
        obs_flats = []  # bs * fnum
        obs_flat_chunk = []
        skill_t_ally_masks_chunk = []
        skill_t_enemy_masks_chunk = []
        for frame in frames: 
            obs_flat, skill_t_ally_masks, skill_t_enemy_masks = fake_agent.process_obs(frame, use_skill_label=False, for_inference=True)
            obs_flat_chunk.append(obs_flat)
            skill_t_ally_masks_chunk.append(skill_t_ally_masks)
            skill_t_enemy_masks_chunk.append(skill_t_enemy_masks)

        output_queue.put((idxs, np.array(obs_flat_chunk), skill_t_ally_masks_chunk, skill_t_enemy_masks_chunk))

class Predictor():
    def __init__(self, process_num=1):
        self.process_num = process_num
        self.fake_agent = Agent(0, {})
        self.fake_agent.replay_record.update_frame(0)  # fake
        logger = Logger(gLogPath, logging.INFO)
        self.model = SLModel(total_steps = 10, is_inference = True, logger = logger)
        ret_info = self.model.restore()
        print(ret_info)
        # multi-process
        self.requests_queue = multiprocessing.Queue(64)
        self.results_queue = multiprocessing.Queue(64)
        self.processes = []
        for _ in range(self.process_num):
            process = multiprocessing.Process(target=handle_requests, args=(self.requests_queue, self.results_queue))
            process.start()
            self.processes.append(process)

    def predict(self, frames):
        obs_flat_list = []  # bs * fnum
        t1 = time.time()
        # split data and put request
        chunk_size = math.ceil(len(frames)/self.process_num)
        for idx in range(self.process_num):
            self.requests_queue.put((list(range(idx,min(len(frames), idx+chunk_size))), frames[idx:idx+chunk_size]))
        # get result from multi-q
        idx_list = []
        obs_flat_list = []
        skill_t_ally_masks_list = []
        skill_t_enemy_masks_list = []
        for _ in range(self.process_num):
            (idxs_chunk, obs_flat_chunk, skill_t_ally_masks_chunk, skill_t_enemy_masks_chunk) = self.results_queue.get()
            obs_flat_list.append(obs_flat_chunk)
            idx_list.extend(idxs_chunk)
            skill_t_ally_masks_list.extend(skill_t_ally_masks_chunk)
            skill_t_enemy_masks_list.extend(skill_t_enemy_masks_chunk)

        # model inference
        obs_flat_np = np.concatenate(obs_flat_list, axis=0)
        t2 = time.time()
        obs_flat = torch.tensor(obs_flat_np).to(DEVICE)
        t3 = time.time()
        skill_pred, ally_selected_pred, enemy_selected_pred = self.model.predict_step(obs_flat)
        t4 = time.time()
        skill_pred, ally_selected_pred, enemy_selected_pred = np.array(skill_pred.cpu()).tolist(), np.array(ally_selected_pred.cpu()).tolist(), np.array(enemy_selected_pred.cpu()).tolist()

        # post process
        t5 = time.time()
        action_jsons = []
        #print("{}\n{}\n{}".format(skill_pred, ally_selected_pred, enemy_selected_pred))
        for idx in idx_list:
            action = {}
            action["aTarget"], action["aSkillSlot"], action["aSkillFullID"] = self.fake_agent.reverse_action(
                                                                                     [skill_pred[idx], ally_selected_pred[idx], enemy_selected_pred[idx]],
                                                                                     skill_t_ally_masks_list[idx], skill_t_enemy_masks_list[idx])
            action_jsons.append(action)
        #print(f"preprocess: {t2-t1} to_GPU: {t3-t2} inference: {t4-t3} to_CPU: {t5-t4} postprocess: {time.time()-t5}")
        return action_jsons


if __name__ == "__main__":
    import sys
    import json
    import os
    import glob
    predictor = Predictor()
    files = glob.glob("test_data/*.json")
    bs = 200
    frames = []
    for fpath in files:
        fid='_'.join(fpath.split("/")[-2:])
        with open(fpath, 'r') as f:
            tmp_frames = json.load(f)
            frames.extend(tmp_frames)
            if len(frames) > bs:
                print(f"process frame num {len(frames)}")
                act = predictor.predict(frames)
                #print(act)
                frames = []
