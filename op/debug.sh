#! /bin/sh

cd ../


#python3 -m rl.train_sl.dataloader
#python3 -m rl.train_sl.model
#CUDA_VISIBLE_DEVICES=0 python3 -m rl.train_rl.train
#python3 -m rl.train_sl.train_debug

rm -r data/train
rm -r data/valid
mkdir data/train/
mkdir data/valid/
rm output/replay_record/* 
mkdir output/replay_record/
python3 -m rl.train_sl_torch.preprocess 1
