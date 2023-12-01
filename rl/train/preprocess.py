import os
import torch
import json
import glob
import random
import shutil
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
from tqdm import tqdm

from rl.env.agent.agent import Agent
from .config import *

def move_file(source, destination):
    shutil.move(source, destination)

def split_validation_set(data_folder, valid_folder, ratio=120, shuffle=True):
    # Ensure the valid folder exists
    os.makedirs(valid_folder, exist_ok=True)

    # Get a list of all data files
    data_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f)) and f.endswith(".pt")]
    if shuffle:
        random.shuffle(data_files)

    # Calculate the number of files to move to the valid folder
    num_valid_files = math.ceil(len(data_files) / ratio)

    # Move files to the valid folder with ThreadPoolExecutor for concurrency
    with ThreadPoolExecutor() as executor:
        for i in range(num_valid_files):
            source = os.path.join(data_folder, data_files[i])
            destination = os.path.join(valid_folder, data_files[i])
            executor.submit(move_file, source, destination)


ally_sample_rate = 4
enemy_sample_rate = 1
MAX_FRAME_NUM_PER_FILE = 4000

def process_single_frame(fake_agent, frame):
    frame_data = {}
    obs = fake_agent.process_obs(frame, use_skill_label=True)
    skill_slot = int(obs["skill_slot_label"])
    frame_data["obs"] = np.array([])
    for k, v in obs.items():
        frame_data["obs"] = np.concatenate([frame_data["obs"], v])
    frame_data["action"] = np.array(fake_agent.process_action(frame), dtype=np.float32) 
    return frame_data, int(obs["skill_t_ally_masks"][skill_slot]), int(obs["skill_t_enemy_masks"][skill_slot])

def process_single_game(frames, fname):
    game_data = []
    # fake rl agent
    fake_agent = Agent(0, {}, fname = fname)

    #print("process {} frame...".format(len(frames)))
    for frame_id, frame in enumerate(frames):
        fake_agent.replay_record.update_frame(frame_id)
        frame_data, is_ally_selection, is_enemy_selection = process_single_frame(fake_agent, frame)
        
        # Convert NumPy arrays to PyTorch tensors
        frame_data["obs"] = torch.from_numpy(frame_data["obs"]).float()
        frame_data["action"] = torch.from_numpy(frame_data["action"]).float()

        # up sample
        if is_ally_selection:
            for _ in range(ally_sample_rate):
                game_data.append(frame_data)
        # one sample
        game_data.append(frame_data)
    if len(game_data) > 0:
        # fake_agent.replay_record.dump(True)
        fake_agent.replay_record.dump(False)
    return game_data


def single_preprocess(input_files, output_dir):
    total_frame_num = 0
    game_data = []
    for fpath in tqdm(input_files, desc="Processing files"):
        fname = (fpath.split('/')[-2]+"_"+fpath.split('/')[-1])[:-5]
        try:
            with open(fpath, 'r') as f:
                #print("process {} ... cur accumulate {}".format(fpath, len(game_data)))
                frames = json.load(f)
                if len(frames)<4 or len(frames)>50:
                    continue
                game_data.extend(process_single_game(frames, fname))
                total_frame_num += len(game_data)
        except RuntimeError as e:
            print(e)
            return
        except Exception as e:
            print(e)
        if len(game_data) >= MAX_FRAME_NUM_PER_FILE:
            torch.save(game_data[:MAX_FRAME_NUM_PER_FILE], output_dir + fname + ".pt")
            print('Length: ', len(game_data[:MAX_FRAME_NUM_PER_FILE]), 'Save in', output_dir + fname + ".pt")
            
            game_data = game_data[MAX_FRAME_NUM_PER_FILE:]
    # process tail 
    torch.save(game_data, output_dir + fname + ".pt")

def multi_preprocess(input_dir, output_dir, process_num = 1):
    files = glob.glob(input_dir + "/*/*.json")
    # files = [f.replace('\\', '/') for f in files]

    file_chunks = np.array_split(files, process_num)

    process_list = []

    for idx, block_files in enumerate(file_chunks):
        process_list.append(Process(target=single_preprocess, args=(block_files, output_dir)))
        process_list[-1].start()
        print("process {} file_num {} start, from {} to {} ".format(idx, len(block_files), block_files[0], block_files[-1]))

    for idx in range(len(process_list)):
        process_list[idx].join()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        process_num = int(sys.argv[1])
        version = sys.argv[2]
    else:
        raise Exception("Please input process_num and version!")
    gJsonInputPath = gJsonInputPath.format(version)
    print(gJsonInputPath)
    if not os.path.exists(gJsonInputPath):
        raise Exception("JSON input path not exist!")
    gTrainDataPath = f"{gTrainDataPath}train_{version}/"
    gValidDataPath = f"{gValidDataPath}valid_{version}/"
    if not os.path.exists(gTrainDataPath):
        os.makedirs(gTrainDataPath)
    if process_num == 1:
        files = glob.glob(gJsonInputPath + "/*/*.json")
        single_preprocess(files, gTrainDataPath)
    elif process_num > 1:
        multi_preprocess(gJsonInputPath, gTrainDataPath, process_num)
    else:
        raise Exception("process_num must be greater than 0!")
    if not os.path.exists(gValidDataPath):
        os.makedirs(gValidDataPath)
    split_validation_set(gTrainDataPath, gValidDataPath)