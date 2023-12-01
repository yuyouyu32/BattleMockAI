import logging
import numpy as np
from rl.common.logger import Logger
import torch
import torchvision
from rl.env.act_obs_space import *
from rl.env.agent.agent import Agent
from .config import *
from .model import SLModel 
from .dataloader import TorchDataLoader as Dataloader

logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('torchvision').setLevel(logging.ERROR)


skill_correct, ally_correct, enemy_correct = [], [], []

class Predictor():
    def __init__(self, is_online = False):
        self.fake_agent = Agent(0, {})
        self.fake_agent.replay_record.update_frame(0)  # fake
        logger = Logger(gLogPath, logging.INFO)
        self.is_online = is_online
        self.model = SLModel(total_steps = 10, is_inference = True, logger = logger)
        ret_info = self.model.restore()
        print(ret_info)

    def predict(self, frame):
        obs = self.fake_agent.process_obs(frame, use_skill_label=False)
        obs_flat = np.array([])
        for k, v in obs.items():
            obs_flat = np.concatenate([obs_flat, v])
        obs_flat = np.array([obs_flat])  # 1 * fnum
        obs_flat = torch.tensor(obs_flat, dtype=torch.float32).to(DEVICE)
        print(obs_flat)
        print(obs_flat.shape)
        skill_pred, ally_selected_pred, enemy_selected_pred = self.model.predict_step(obs_flat)
        skill_pred, ally_selected_pred, enemy_selected_pred = np.array(skill_pred[0].cpu()), np.array(ally_selected_pred[0].cpu()), np.array(enemy_selected_pred[0].cpu())
    
        if not self.is_online:
            labels = self.fake_agent.process_action(frame)
            #print("==="*6)
            need_select_ally, need_select_enemy, info = self.fake_agent.reverse_action(frame, [skill_pred, ally_selected_pred, enemy_selected_pred])
            #print(need_select_ally)
            #print(need_select_enemy)
            #print(info)
            #print("***"*5)

            pos = frame["MainCharState"][0]["oPosIdx"]
            heroid = frame["MainCharState"][0]["oHeloID"]
            dump_info = [labels[0]+1, labels[1]+1, labels[2]+1, skill_pred+1, ally_selected_pred+1, enemy_selected_pred+1, int(labels[0]==skill_pred),int(labels[1]==ally_selected_pred),int(labels[2]==enemy_selected_pred)]
            dump_info = [str(heroid), str(pos)] + [str(t) for t in dump_info]

            #print("skill predict {} label {}".format(skill_pred, labels[0]))
            skill_correct.append(skill_pred==labels[0])
            if not need_select_ally or skill_pred!=labels[0]: 
                #print("ally predict {} label {}".format(ally_selected_pred, labels[1]))
                dump_info[3] = ''
                dump_info[6] = ''
                dump_info[9] = ''
            else:
                ally_correct.append(ally_selected_pred==labels[1])
            if not need_select_enemy or skill_pred!=labels[0]:
                #print("enemy predict {} label {}".format(enemy_selected_pred, labels[2]))
                dump_info[4] = ''
                dump_info[7] = ''
                dump_info[10] = ''
            else:
                enemy_correct.append(enemy_selected_pred==labels[2])
            #print(dump_info)
            return ','.join(dump_info)
        else:
            _, _, info = self.fake_agent.reverse_action(frame, [skill_pred, ally_selected_pred, enemy_selected_pred])


if __name__ == "__main__":
    import sys
    import json
    import os
    import glob
    dump_result = True
    #predictor = Predictor(is_online=True)
    predictor = Predictor(is_online=False)
    fout = open('output/dump_replay/pred_red.csv', 'w')
    fout.write('fid,hero_id,pos,skill,ally,enemy,skill_p,allt_p,enemy_p,skill_correct,ally_correct,enemy_correct\n')

    info = []
    files = glob.glob("test_data/*.json")
    for fpath in files:
        print(fpath)
        fid='_'.join(fpath.split("/")[-2:])
        with open(fpath, 'r') as f:
            print(fpath)
            if dump_result:
                frames = json.load(f)
            for frame in frames:
                dump_info = predictor.predict(frame)
                #print(frame["PlayerAction"])
                if dump_result:
                    fout.write(fid+","+dump_info+'\n')
                info.append(fid+","+dump_info)
    print(sum(skill_correct)/(len(skill_correct)+0.0001))
    print(sum(ally_correct)/(len(ally_correct)+0.0001))
    print(sum(enemy_correct)/(len(enemy_correct)+0.0001))
    fout.close()

    # dump by hero
    info_map = {}
    acc_map = {}
    for line in info:
        terms = line.split(',')
        hero_id = terms[1]
        if hero_id not in info_map:
            info_map[hero_id] = [line]
        else:
            info_map[hero_id].append(line)

    for hero, lines in info_map.items():
        with open('output/dump_replay/'+hero+'.csv', 'w') as fout:
            fout.write('fid,hero_id,pos,skill,ally,enemy,skill_p,allt_p,enemy_p,skill_correct,ally_correct,enemy_correct\n')
            skill_c, ally_c, enemy_c = [], [], []
            for line in lines:
                terms = line.split(',')
                if terms[9] != '':
                    skill_c.append(int(terms[9]))
                if terms[10] != '':
                    ally_c.append(int(terms[10]))
                if terms[11] != '':
                    enemy_c.append(int(terms[11]))
                fout.write(line+'\n')
            acc_map[hero] = [sum(skill_c)/(len(skill_c)+0.0001), sum(ally_c)/(len(ally_c)+0.0001), sum(enemy_c)/(len(enemy_c)+0.0001), len(lines)]
            if len(skill_c) == 0:
                acc_map[hero][0] = ''
            if len(ally_c) == 0:
                acc_map[hero][1] = ''
            if len(enemy_c) == 0:
                acc_map[hero][2] = ''
            fout.write(','.join([str(t) for t in acc_map[hero]]) + '\n')

    with open('output/dump_replay/summary.csv', 'w') as fout:
        fout.write('hero_id,skill_acc,ally_acc,enemy_acc\n')
        for hero,info in acc_map.items():
            fout.write(','.join([hero]+[str(t) for t in info])+'\n')
        fout.write(','.join([str(t) for t in [sum(skill_correct)/(len(skill_correct)+0.0001), sum(ally_correct)/(len(ally_correct)+0.0001), sum(enemy_correct)/(len(enemy_correct)+0.0001)]])+'\n')
        

