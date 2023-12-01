#! /bin/sh

cd ../
rm -r data/train
rm -r data/valid
mkdir data/train/
mkdir data/valid/
rm output/replay_record/* 
mkdir output/replay_record/
python -m rl.train.preprocess 12 v330

rm /mnt/ssd/dislyte_data/train/*
rm /mnt/ssd/dislyte_data/valid/*
rm /home/dislyte_data/train/*

ls data/train |head -n 50 |xargs -i mv data/train/{} /mnt/ssd/dislyte_data/valid/
ls data/train |head -n 1200 |xargs -i mv data/train/{} /home/dislyte_data/train/
mv data/train/*  /mnt/ssd/dislyte_data/train
