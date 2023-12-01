import time
import logging
import numpy as np
from rl.common.logger import Logger
from rl.env.act_obs_space import *
from .config import *
from .model import SLModel 
from .dataloader import fetch_data
import glob
import multiprocessing as mp
from tensorboardX import SummaryWriter
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, current_loss, model: SLModel):
        if self.best_loss is None:
            self.best_loss = current_loss
            model.logger.info(model.manager.save_ckpt())
        elif self.best_loss - current_loss > self.min_delta:
            self.best_loss = current_loss
            model.logger.info(model.manager.save_ckpt())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Indicator():
    def __init__(self, mode):
        self.mode = mode
        self.reset()

    def reset(self):
        self.sample_num = 0
        self.loss_avg = 0.0
        self.loss_list = [[] for _ in range(len(ActionName))]
        self.acc_list = [[] for _ in range(len(ActionName))]
        if self.mode == "valid":
            self.label_dist = [[0.0 for _ in range(d)] for d in ActionDim]
            self.predict_dist = [[0.0 for _ in range(d)] for d in ActionDim]

    def update_acc_loss(self, loss, loss_list, acc_list):
        # calc avg
        for tidx in range(len(ActionName)):
            self.loss_avg += loss
            self.loss_list[tidx].append(loss_list[tidx])
            if 1>=acc_list[tidx]>=0:
                self.acc_list[tidx].append(acc_list[tidx])

    def update_label_dist(self, label_list, predict_label_list):
        """
        label_list: [bs, filter_bs, filter_bs]
        predict_label_list: [bs, filter_bs, filter_bs]
        """
        for idx in range(len(ActionName)):
            for bidx in range(len(label_list[0])):
                if bidx >= len(label_list[idx]):
                    break
                predict_index = predict_label_list[idx][bidx]
                label_index = label_list[idx][bidx]
                assert label_index%1 == 0.0
                label_index = int(label_index)
                self.label_dist[idx][label_index] += 1
                self.predict_dist[idx][predict_index] += 1

    def update(self, loss_list, acc_list, label_list = None, predict_label_list = None):
        """
        loss_list: [1, 1, 1, 1]      # already do target mask, but could be problem if not target in the whole batch 
        acc_list: [1, 1, 1]          # already do target mask, but could be problem if not target in the whole batch
        """
        loss = loss_list[0]  # total loss
        loss_list = loss_list[1:]   # 3 loss
        self.sample_num += 1 
        self.update_acc_loss(loss, loss_list, acc_list)
        if self.mode == "valid":
            self.update_label_dist(label_list, predict_label_list)

    def show(self, logger):
        logger.info("")
        logger.info("loss {}".format(self.loss_avg))
        logger.info("")
        for tidx in range(len(ActionName)):
            logger.info("action_{} sl_loss: {}, acc: {}".format(tidx,
                                                                 sum(self.loss_list[tidx])/len(self.loss_list[tidx]),
                                                                 sum(self.acc_list[tidx])/len(self.loss_list[tidx])))
        if self.mode == "valid":
            logger.info("")
            for tidx, label_name in enumerate(ActionName):
                label_dist_str = label_name + "_label"
                predict_dist_str = label_name + "_predict"
                sub_action_dim = ActionDim[tidx]
                for action_sub_idx in range(sub_action_dim):
                    label_dist_str += " {}: {:.3f}".format(action_sub_idx, self.label_dist[tidx][action_sub_idx]/(sum(self.label_dist[tidx])+0.00001))
                    predict_dist_str += " {}: {:.3f}".format(action_sub_idx, self.predict_dist[tidx][action_sub_idx]/(sum(self.predict_dist[tidx])+0.00001))
                logger.info(label_dist_str)
                logger.info(predict_dist_str) 
                logger.info("")
        self.reset()

def tf_writter(writer, loss_list, acc_list, step, prefix='train'):
    loss_names = ['total_loss', 'skill_loss', 'ally_selected_loss', 'enemy_selected_loss']
    acc_names = ['skill_acc', 'ally_selected_acc', 'enemy_selected_acc']
    for index, name in enumerate(loss_names):
        writer.add_scalar(f'{prefix}/Loss/{name}', loss_list[index], step)
    for index, name in enumerate(acc_names):
        writer.add_scalar(f'{prefix}/Acc/{name}', acc_list[index], step)

def train():
    train_indicator = Indicator("train")
    train_frame_num = 4000 * len(glob.glob(gTrainDataPath+"/*/*.pt"))

    valid_indicator = Indicator("valid")

    logger = Logger(gLogPath, logging.INFO)
    total_steps = gMax_epoch*train_frame_num/gBatch_size
    model = SLModel(total_steps=total_steps, logger = logger)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    logger.info("{} frame for {} steps".format(train_frame_num, total_steps))
    
    writer = SummaryWriter(gSummaryPath)
    if gUseRestore:
        ret_info = model.restore_from_pth(gRestorePath)
        logger.info(ret_info)
        logger.info("Continue training from checkpoint {}".format(gRestorePath))
    for epoch in range(gMax_epoch):
        train_data_loader = fetch_data(data_dir=gTrainDataPath ,buff_size=40000, batch_size=512, queue=mp.Queue(maxsize=50), num_workers=2, shuffle=True, drop_last=False)
        train_indicator.reset()
        logger.info("======== train ========")
        pre_step_stime = 0

        for step, batch in enumerate(train_data_loader):  
            step_stime = time.time()
            # train op
            loss_list, predict_label_list, label_list, acc_list = model.step(batch)
            # write to tensorboard
            tf_writter(writer, loss_list, acc_list, model.manager.train_step, prefix='train')
            model.manager.train_step += 1
            # update indicator
            train_indicator.update(loss_list, acc_list)
            # show train info
            if (step+1) % gPrintstep == 0:
                logger.info("*** epoch {} step {} cost time: {} {}***".format(epoch, step, time.time()-step_stime, time.time()-pre_step_stime))
                train_indicator.show(logger)
            pre_step_stime = step_stime

        valid_data_loader = fetch_data(data_dir=gValidDataPath ,buff_size=40000, batch_size=512, queue=mp.Queue(maxsize=50), num_workers=2, shuffle=True, drop_last=False)
        valid_indicator.reset()
        logger.info("======== valid ========")
        loss_record = np.array([0.0, 0.0, 0.0, 0.0])
        acc_record = np.array([0.0, 0.0, 0.0])
        for step, batch in enumerate(valid_data_loader):
            # train op
            loss_list, predict_label_list, label_list, acc_list = model.step(batch, is_train=False)
            loss_record += np.array(loss_list)
            acc_record += np.array(acc_list)
            valid_indicator.update(loss_list, acc_list, label_list, predict_label_list)
        loss_list = loss_record/(step+1)
        acc_list = acc_record/(step+1)
        tf_writter(writer, loss_list, acc_list, model.manager.valid_step, prefix='valid')
        model.manager.valid_step += 1
        early_stopping.step(loss_list[0], model)


        # show valid info
        logger.info("")
        logger.info("=== epoch {} valid step ==== ".format(epoch))
        valid_indicator.show(logger)
        logger.info("============================ ")
        logger.info("")
        
        if early_stopping.early_stop:
            logger.info("======== Early stopping triggered ========")
            break
                

if __name__ == "__main__":
    train()
