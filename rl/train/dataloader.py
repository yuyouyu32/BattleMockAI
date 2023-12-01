import random
import time
import multiprocessing as mp
import torch
import torch.multiprocessing as tmp
import glob
from typing import List

from .config import DEVICE

if tmp.get_start_method(allow_none=True) != 'spawn':
    tmp.set_start_method('spawn', force=True)



class MultiProcessDataset:
    def __init__(self, batch_size: int, pt_files: List[str], buffer_size: int = 80000, shuffle: bool = True, drop_last: bool =False) -> None:
        self.pt_files = pt_files
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        # print("Total number of pt files: {}".format(len(self.pt_files)))
        self.drop_last = drop_last
        self.buffer = []
        self.remain_files = True

    def load_data(self):
        while len(self.buffer) < self.buffer_size and self.pt_files:
            pt_file = self.pt_files.pop()
            data_batch = torch.load(pt_file)
            self.buffer.extend(data_batch)
            # print("Load data from {}".format(pt_file), "Buffer size: {}".format(len(self.buffer)))
        if self.shuffle:
            random.shuffle(self.buffer)
        self.remain_files = bool(self.pt_files)
        return self.remain_files
    
    def pop_data(self):
        batch_obs = []
        batch_action = []
        for _ in range(self.batch_size):
            if len(self.buffer) == 0:
                break
            data = self.buffer.pop()
            batch_obs.append(data['obs'])
            batch_action.append(data['action'])
        if self.drop_last:
            if len(batch_obs) < self.batch_size:
                return torch.tensor([]), torch.tensor([])
        if len(batch_obs) > 0:
            batch_obs = torch.stack(batch_obs)
            batch_action = torch.stack(batch_action)
        return batch_obs, batch_action


def chunks(lst, n):
    """Split 'lst' into 'n' equally sized chunks."""
    chunk_size = (len(lst) + n - 1) // n
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def load_data_to_queue(queue: mp.Queue, dataset: MultiProcessDataset):
    remain_files = dataset.load_data()
    while remain_files or dataset.buffer:
        obs, action = dataset.pop_data()
        if len(obs) == 0:
            remain_files = dataset.load_data()
            continue
        while queue.full():
            time.sleep(0.1)
        queue.put((obs.to(DEVICE), action.to(DEVICE)))
    while not queue.empty():
        time.sleep(0.5)

def fetch_data(data_dir: str, buff_size: int, batch_size: int, queue: mp.Queue, num_workers: int, shuffle: bool, drop_last: bool):
    pt_files = glob.glob(data_dir + "/*/*.pt")
    if shuffle:
        random.shuffle(pt_files)
     # Split pt_files into equal parts
    pt_files_chunks = chunks(pt_files, num_workers)   
    
    # Create dataset and loader_process pairs for each worker
    datasets = [MultiProcessDataset(batch_size=batch_size, pt_files=pt_files_chunks[i], buffer_size=buff_size, shuffle=shuffle, drop_last=drop_last) for i in range(num_workers)]
    loader_processes = [mp.Process(target=load_data_to_queue, args=(queue, datasets[i])) for i in range(num_workers)]

    # Start all loader_processes
    for process in loader_processes:
        process.start()
    
    while any(process.is_alive() for process in loader_processes) or not queue.empty():
        if not queue.empty():
            yield queue.get()
        else:
            # print("Queue is empty or processes are still running. Waiting...")
            time.sleep(0.1)

    # Wait for all loader_processes to finish
    for process in loader_processes:
        process.join()