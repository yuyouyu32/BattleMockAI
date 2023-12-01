#! /bin/sh

cd ../
python -m rl.env.agent.feature_def # act_obs_space.json
python3 -m rl.train.train
nohup python3 -m rl.train.train > 8-25.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python3 -m rl.train_sl.train
