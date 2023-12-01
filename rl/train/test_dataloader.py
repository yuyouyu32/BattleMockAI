
from .config import *
import time

# from .dataloader import TorchDataLoader as Dataloader
# print("Running DataLoader with multiple workers...")
# total_processed = 0
# start_time = time.perf_counter()
# data_loader_multiple_workers = Dataloader(gBatch_size, gTrainDataPath, buffer_size=80000)
# for i, (obs, action) in enumerate(data_loader_multiple_workers):
#     time.sleep(1)  # Simulate data processing time
#     # print("Batch {} loaded. Batch size: {}".format(i, len(obs)))
#     # print("Buffer size: {}".format(data_loader_multiple_workers.dataset.get_buffer_size()))
#     total_processed += len(obs)
#     print("Total processed: {}".format(total_processed))
#     if total_processed > 5e5:
#         break
# end_time = time.perf_counter()
# print("Multiple workers took {:.2f} seconds.".format(end_time - start_time))


# from .dataloader import PTFilesDataset
# from torch.utils.data import DataLoader
# import os
# print("Running DataLoader with multiple workers...")
# total_processed = 0
# start_time = time.perf_counter()
# pt_files_dataset = PTFilesDataset(gTrainDataPath)
# data_loader = DataLoader(pt_files_dataset, batch_size=1, shuffle=True, num_workers=0)
# for i, (obs, action) in enumerate(data_loader):
#     time.sleep(1)  # Simulate data processing time
#     # print("Batch {} loaded. Batch size: {}".format(i, len(obs)))
#     # print("Buffer size: {}".format(data_loader_multiple_workers.dataset.get_buffer_size()))
#     obs, action = obs.squeeze(dim=0), action.squeeze(dim=0)
#     total_processed += len(obs)
#     print("Total processed: {}".format(total_processed))
#     if total_processed > 5e5:
#         break
# end_time = time.perf_counter()
# print("Multiple workers took {:.2f} seconds.".format(end_time - start_time))

from .dataloader import fetch_data
import multiprocessing as mp
from .config import *


if __name__ == '__main__':
    buff_size = 40000
    batch_size = 512
    queue = mp.Queue(maxsize=50)
    num_workers = 2
    shuffle = True
    drop_last = False
    for i in range(2):
        dataset = fetch_data(data_dir=gTrainDataPath ,buff_size=buff_size, batch_size=batch_size, queue=queue, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last)
        print("Running DataLoader with multiple workers...")
        total_processed = 0
        start_time = time.perf_counter()
        for obs, action in dataset:
            total_processed += len(obs)
            time.sleep(1)
            print("Total processed: {}".format(total_processed))
        end_time = time.perf_counter()
        print("Multiple workers took {:.2f} seconds.".format(end_time - start_time))