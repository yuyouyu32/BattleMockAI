import os
import torch
import glob
from torch.utils.data import Dataset, DataLoader
import random
from .config import DEVICE

class BufferedDataset(Dataset):
    def __init__(self, pt_files, buffer_size):
        self.pt_files = pt_files
        self.buffer_size = buffer_size
        self.buffer = []

    def load_data(self):
        while len(self.buffer) < self.buffer_size and self.pt_files:
            pt_file = self.pt_files.pop()
            data_batch = torch.load(pt_file)
            self.buffer.extend(data_batch)
            # print("Load data from {}".format(pt_file), "Buffer size: {}".format(len(self.buffer)))
        remain_files = bool(self.pt_files)
        return remain_files

    def __len__(self):
        return len(self.buffer)
    
    def get_size(self):
        # each pt file contains 4000 frames
        return 4000 * len(self.pt_files)

    def __getitem__(self, idx):
        return self.buffer[idx]['obs'], self.buffer[idx]['action']

class TorchDataLoader(DataLoader):
    def __init__(self, batch_size, data_dir, buffer_size=80000, shuffle=True, drop_last=False):
        self.data_dir = data_dir
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pt_files = glob.glob(data_dir + "/*_v330/*.pt")
        print("Total number of pt files: {}".format(len(self.pt_files)))
        self.drop_last = drop_last

        if self.shuffle:
            random.shuffle(self.pt_files)
        self.dataset = BufferedDataset(self.pt_files, self.buffer_size)
        self.remain_files = self.dataset.load_data()

        super(TorchDataLoader, self).__init__(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=os.cpu_count())
    

    def __iter__(self):
        while self.dataset.buffer or self.pt_files:
            if len(self.dataset) < self.batch_size:
                if self.drop_last:
                    break
                else:
                    actual_batch_size = len(self.dataset)
            else:
                actual_batch_size = self.batch_size

            if self.shuffle:
                batch_indices = random.sample(range(len(self.dataset.buffer)), actual_batch_size)
            else:
                batch_indices = list(range(actual_batch_size))

            batch_obs = []
            batch_action = []

            for index in sorted(batch_indices, reverse=True):
                obs, action = self.dataset[index]
                self.dataset.buffer.pop(index)
                batch_obs.append(obs)
                batch_action.append(action)

            batch_obs = torch.stack(batch_obs).to(DEVICE)
            batch_action = torch.stack(batch_action).to(DEVICE)

            if self.remain_files and len(self.dataset) < self.buffer_size:
                self.remain_files = self.dataset.load_data()

            yield batch_obs, batch_action

class PTFilesDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.pt_files = glob.glob(data_dir + "/*_batch/*.pt")

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        file_path = self.pt_files[idx]
        data = torch.load(file_path)
        return torch.stack([item['obs'] for item in data]).squeeze(dim=0), torch.stack([item['action'] for item in data]).squeeze(dim=0)
    