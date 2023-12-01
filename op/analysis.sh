#! /bin/sh

cd ../
rm output/analysis/*png
python3 -m rl.train_sl.draw_merge output/replay_record/ output/analysis/
