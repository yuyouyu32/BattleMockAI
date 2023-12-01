import os
import re
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.onnx

from .network import CustomModel
from rl.common.common import check_and_clean
from rl.common.logger import Logger
from .config import *

class MyLRSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, target_lr, warmup_steps, total_steps, hold=None):
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps + 0.01
        self.total_steps = total_steps
        self.hold = hold if hold is not None else warmup_steps
        super(MyLRSchedule, self).__init__(optimizer)

    def get_lr(self):
        # Step number from the built-in counter
        step = self._step_count

        # Cosine annealing
        learning_rate = 0.5 * self.target_lr * (1 + np.cos(np.pi * float(step - self.warmup_steps - self.hold) / float(self.total_steps - self.warmup_steps - self.hold)))

        # Warm-up learning rate
        warmup_lr = self.target_lr * (step / self.warmup_steps)

        # Choose between `warmup_lr`, `target_lr`, and `learning_rate` based on whether `step < warmup_steps` and we're still holding.
        # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
        if self.hold > 0:
            learning_rate = np.where(step > self.warmup_steps + self.hold,
                                     learning_rate, self.target_lr)
        
        learning_rate = np.where(step < self.warmup_steps, warmup_lr, learning_rate)
        
        # Apply the updated learning rate to all optimizer parameter groups
        return [learning_rate for _ in self.base_lrs]

class CheckpointManager:
    def __init__(self, model, optimizer, path, max_to_keep=10):
        self.model = model
        self.optimizer = optimizer
        self.path = path
        self.max_to_keep = max_to_keep
        self.train_step = 0
        self.valid_train_step = 0
        self.valid_step = 0

    @staticmethod
    def extract_number(filename):
        match = re.search(r'checkpoint_(\d+)\.pth', filename)
        return int(match.group(1)) if match else None
    
    def restore(self):
        checkpoints = sorted([os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith(".pth")], key=self.extract_number)
        if not checkpoints:
            raise RuntimeError(f"No existing model in {self.path}, initializing from scratch.")
        
        latest_checkpoint = checkpoints[-1]
        return self.restore_from_pth(latest_checkpoint)
    
    def restore_from_pth(self, pth_path):
        checkpoint = torch.load(pth_path)
        self.valid_step = checkpoint["step"]
        return self.restore_from_pth(checkpoint)
    
    def restore_from_pth(self, pth_path):
        checkpoint = torch.load(pth_path)
        self.valid_step = checkpoint["step"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return f"Restored from {pth_path}"

    def save_ckpt(self):
        checkpoint_path = os.path.join(self.path, f"checkpoint_{self.valid_step}.pth")     
          
        # delete old checkpoint
        checkpoints = sorted([os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith(".pth")], key=self.extract_number)
        num_checkpoints = len(checkpoints)
        if num_checkpoints > self.max_to_keep:
            for i in range(num_checkpoints - self.max_to_keep):
                os.remove(checkpoints[i])
        torch.save({"step": self.valid_step,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   checkpoint_path)
        
        return "Saved checkpoint {}".format(checkpoint_path)
    
    def convert_to_onnx(self, onnx_path):
        self.restore()
        self.model.eval()
        dummy_input = torch.ones([gBatch_size, FS]).to(DEVICE)
        traced_model = torch.jit.trace(self.model, dummy_input)
        output_names = ["output" + str(i) for i in range(4)]
        torch.onnx.export(model=traced_model, args=dummy_input, f=onnx_path, opset_version=15, input_names=["input"], output_names=output_names, verbose=True)

        return f"ONNX model saved at {onnx_path}"


class SLModel:
    def __init__(self, total_steps, is_inference=False, logger=None, is_debug=False):
        # train parameter
        self.batch_size = gBatch_size
        
        # create model
        obs_space = torch.zeros(sum(v.shape[0] for k, v in OBS_SPACE.items())).to(DEVICE)
        action_space = torch.zeros(len(ActionDim)).to(DEVICE)
        self.model = CustomModel(obs_space, action_space, sum(ActionDim)).to(DEVICE)

        # optimizer
        self.opt = optim.Adam(self.model.parameters(), lr=gLR)
        self.scheduler = MyLRSchedule(self.opt, target_lr=gLR, warmup_steps=int(total_steps * 0.05), total_steps=total_steps)
        
        # logger
        if logger is None:
            self.logger = Logger(gLogPath)
        else:
            self.logger = logger
        
        self.logger.info("===================Model runing on {}===================".format(DEVICE))
        
        # checkpoint
        if is_inference:
            self.manager = CheckpointManager(model=self.model, optimizer=None, path=gCkptPath, max_to_keep=10)
            self.logger.info(self.manager.restore())
        else:
            check_and_clean(gCkptPath, gUseRestore)
            check_and_clean(gSummaryPath, gUseRestore)
            # self.summary_writer = torch.utils.tensorboard.SummaryWriter(gSummaryPath)
            self.manager = CheckpointManager(model=self.model, optimizer=self.opt, path=gCkptPath, max_to_keep=10)
            

        self.is_debug = is_debug

    def restore(self):
        return self.manager.restore()
    
    def restore_from_pth(self, pth_path):
        return self.manager.restore_from_pth(pth_path)

    def save_ckpt(self):
        return self.manager.save_ckpt()
    
    def step(self, batch, is_train=True):
        obs_flat, labels = batch
        # inference
        self.model.zero_grad()
        concat_logits, skill_mask, skill_t_ally_mask, skill_t_enemy_mask = self.model(obs_flat)

        # calculate loss
        batch_skill_loss, batch_ally_selected_loss, batch_enemy_selected_loss = self._calculate_loss(concat_logits, labels)

         # calc final label
        skill_prob, ally_selected_prob, enemy_selected_prob = self._calculate_prob(concat_logits)
        # bs * action_num -> bs
        skill_pred = torch.argmax(skill_prob, dim=-1)
        ally_selected_pred = torch.argmax(ally_selected_prob, dim=-1)
        enemy_selected_pred = torch.argmax(enemy_selected_prob, dim=-1)
         # metric
        acc_ratio_list = self._calculate_metric([skill_pred, ally_selected_pred, enemy_selected_pred], labels, [skill_t_ally_mask, skill_t_enemy_mask])

        # loss postprocess (mask + mean)
        batch_ally_selected_loss = batch_ally_selected_loss[skill_t_ally_mask.bool()]  # filter_bs
        batch_enemy_selected_loss = batch_enemy_selected_loss[skill_t_enemy_mask.bool()]  # filter_bs
        batch_ally_selected_loss = torch.cat([batch_ally_selected_loss, torch.tensor([0.0]).to(DEVICE)], dim=-1)  # filter_bs -> filter_bs+1, avoid reduce nan caused by (filter_bs=0)
        batch_enemy_selected_loss = torch.cat([batch_enemy_selected_loss, torch.tensor([0.0]).to(DEVICE)], dim=-1)  # filter_bs -> filter_bs+1, avoid reduce nan caused by (filter_bs=0)

        skill_loss = batch_skill_loss.mean()
        ally_selected_loss = batch_ally_selected_loss.mean()
        enemy_selected_loss = batch_enemy_selected_loss.mean()
        total_loss = skill_loss + 1.5 * ally_selected_loss + 2 * enemy_selected_loss

        # optimization
        if is_train:
            total_loss.backward()
            self.opt.step()
            self.scheduler.step()

        # mask for analysis
        ally_selected_pred = ally_selected_pred[skill_t_ally_mask.bool()]  # bs -> filter_bs
        enemy_selected_pred = enemy_selected_pred[skill_t_enemy_mask.bool()]  # bs -> filter_bs
        label_list = [labels[:, 0],
                    labels[:, 1][skill_t_ally_mask.bool()],
                    labels[:, 2][skill_t_enemy_mask.bool()]]  # bs * 3 -> [bs, filter_bs, filter_bs]
        
        # debug
        # print("========== skill prob ==========")
        # print(skill_prob)
        # print("ally prob")
        # print(ally_selected_prob)
        # print("enemy prob")
        # print(enemy_selected_prob)
        # print("========== loss ============")
        # print("batch_skill_loss")
        # print(batch_skill_loss)
        # print("ally_selected_loss")
        # print(batch_ally_selected_loss)
        # print("enemy_selected_loss")
        # print(batch_enemy_selected_loss)
        # print("========== mask =========")
        # print("ally mask")
        # print(skill_t_ally_mask)
        # print("enemy mask")
        # print(skill_t_enemy_mask)
        # print("========== predict =========")
        # print("skill_pred")
        # print(skill_pred)
        # print("ally_selected_pred")
        # print(ally_selected_pred)
        # print("enemy_selected_pred")
        # print(enemy_selected_pred)
        # print("========== label =========")
        # print("skill_label")
        # print(label_list[0])
        # print("ally_selected_label")
        # print(label_list[1])
        # print("enemy_selected_label")
        # print(label_list[2])
        # print("skill_loss {}, ally_selected_loss {}, enemy_selected_loss {}".format(skill_loss, ally_selected_loss, enemy_selected_loss))
        # print("========== acc =========")
        # print("skill acc")
        # print(acc_ratio_list[0])
        # print("ally acc")
        # print(acc_ratio_list[1])
        # print("enemy acc")
        # print(acc_ratio_list[2])

        """
        output : 
        loss_list: [1, 1, 1, 1]     # notice! loss is not masked (not bool_mask before reduce_mean, just * zero), just for debug, do not analysis with single loss value, check tensorboard curves 
        skill_pred_list: [bs, filter_bs, filter_bs]
        label_list: [bs, filter_bs, filter_bs]
        acc_list: [1, 1, 1]         # already do target mask, but could be problem if not target in the whole batch
        """
        return [total_loss.item(), skill_loss.item(), ally_selected_loss.item(), enemy_selected_loss.item()], \
           [skill_pred.cpu().numpy(), ally_selected_pred.cpu().numpy(), enemy_selected_pred.cpu().numpy()], \
           [label.cpu() for label in label_list], \
           [t.item() for t in acc_ratio_list]

    def _calculate_metric(self, predict_label_list, labels, mask_list):
        """
        input:
            predict_label_list:  [bs, bs, bs] 
            labels: bs * 3 
            mask_list: [bs, bs]   # for ally, enemy
        return:
            correct_ratio_list: [1, 1, 1]
        """
        correct_ratio_list = []

        tidx = 0  # skill
        predict_index = predict_label_list[tidx]  # bs
        task_labels = labels[:, tidx].long()  # bs
        correct_prediction = (predict_index == task_labels).float()
        correct_ratio = correct_prediction.mean()  # 1
        correct_ratio_list.append(correct_ratio)

        for tidx in [1, 2]:
            predict_index = predict_label_list[tidx]  # bs
            task_labels = labels[:, tidx].long()  # bs
            correct_prediction = (predict_index == task_labels).float()
            correct_prediction = correct_prediction[mask_list[tidx-1].bool()]
            correct_ratio = correct_prediction.mean()  # 1
            correct_ratio_list.append(correct_ratio)

        return correct_ratio_list

    def _calculate_loss(self, concat_logits, task_label):
        """
        task_logits: bs * output_num
        task_loss: bs, bs, bs
        """
        task_label = task_label.long()  # bs * 3
        # split
        skill_logits, ally_selected_logits, enemy_selected_logits = torch.split(concat_logits, [ACTION_NUM, ALLY_NUM + 1, ENEMY_NUM], dim=-1)
        # get loss
        criterion = nn.CrossEntropyLoss(reduction='none')
        batch_skill_loss = criterion(skill_logits, task_label[:, 0])
        batch_ally_selected_loss = criterion(ally_selected_logits, task_label[:, 1])
        batch_enemy_selected_loss = criterion(enemy_selected_logits, task_label[:, 2])
        return batch_skill_loss, batch_ally_selected_loss, batch_enemy_selected_loss

    def _calculate_prob(self, concat_logits):
        """
        task_logits: bs * output_num
        task_prob: bs*ACTION_NUM, bs*ALLY_NUM, bs*ENEMY_NUM
        """
        # split
        skill_logits, ally_selected_logits, enemy_selected_logits = torch.split(concat_logits, [ACTION_NUM, ALLY_NUM + 1, ENEMY_NUM], dim=-1)
        # get action prob
        skill_prob = F.softmax(skill_logits, dim=-1)
        ally_selected_prob = F.softmax(ally_selected_logits, dim=-1)
        enemy_selected_prob = F.softmax(enemy_selected_logits, dim=-1)
        return skill_prob, ally_selected_prob, enemy_selected_prob

    def _process_gradient(self, x):
        return x
    
    def predict_step(self, obs_flat):
        # inference
        self.model.zero_grad()
        t1 = time.time()
        concat_logits, skill_mask, skill_t_ally_mask, skill_t_enemy_mask = self.model(obs_flat)

        # calc final label
        skill_prob, ally_selected_prob, enemy_selected_prob = self._calculate_prob(concat_logits)
        # bs * action_num -> bs
        skill_pred = torch.argmax(skill_prob, dim=-1)
        ally_selected_pred = torch.argmax(ally_selected_prob, dim=-1)
        enemy_selected_pred = torch.argmax(enemy_selected_prob, dim=-1)
        print("@@@@@@@@@ 1:")
        print(time.time()-t1)
        return skill_pred, ally_selected_pred, enemy_selected_pred


if __name__ == "__main__":
    model = SLModel(total_steps=100000, logger = None, is_inference=True)
    model.manager.convert_to_onnx(onnx_path="./output_sl/ckpt_pred_torch/checkpoint_8.14.onnx")