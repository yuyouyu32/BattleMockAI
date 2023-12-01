from rl.env.act_obs_space import *
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gMax_epoch = 50
gBatch_size = 512
gPrintstep = 50
gLR = 0.0005
gLREps = 1e-4
gSaveStep = 1000

FS = sum([1, MP_FEAT_NUM, BUFF_NUM*BUFF_FEAT_NUM, ALLY_NUM*ALLY_FEAT_NUM, ALLY_NUM*BUFF_NUM*BUFF_FEAT_NUM, ENEMY_NUM*ENEMY_FEAT_NUM, ENEMY_NUM*BUFF_NUM*BUFF_FEAT_NUM, \
             ACTION_NUM*SKILL_FEAT_NUM, ALLY_NUM+1, ENEMY_NUM, ACTION_NUM, ACTION_NUM, ACTION_NUM, 1])
ACTS = 3

gJsonInputPath = "/mnt_data/Dislyte_AIrobot_Data/json_data/{}/"
gTrainDataPath = "/mnt_data/Dislyte_AIrobot_Data/pt_train/"
gValidDataPath = "/mnt_data/Dislyte_AIrobot_Data/pt_valid/"
gOutputPath = "output_sl/"
gSummaryPath = gOutputPath + "/summary"
gCkptPath = gOutputPath + "/ckpt/"
gPredictModelPath = gOutputPath + "/ckpt_pred_torch/"
gUseRestore = False
gRestorePath = gOutputPath + "/ckpt_pred_torch/checkpoint_8.24.pth"
gOnxxSavePath = gPredictModelPath + "/checkpoint.onnx"


gLogPath = gOutputPath + "/train.log"