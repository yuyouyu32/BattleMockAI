#! /bin/sh

cd ../
# python3 -m rl.train.train
nohup python3 -m rl.train.train > v332.log 2>&1 &
#python3 -m rl.train_sl.train
#CUDA_VISIBLE_DEVICES=0 python3 -m rl.train_sl.train
