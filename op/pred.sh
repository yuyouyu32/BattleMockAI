#! /bin/sh

cd ../
rm -rf output/dump_replay
mkdir output/dump_replay
python3 -m rl.predict.inference_batch_multi
#python3 -m rl.predict.inference_batch
#python3 -m rl.predict.server
#CUDA_VISIBLE_DEVICES=0 python3 -m rl.train_sl.train
