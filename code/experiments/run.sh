#!/bin/bash

python train_roberta.py \
  --model_name_or_path xlm-roberta-large \
  --dataset_name annotation \
  --train_file ../data/jinyong/train.csv \
  --test_file ../data/jinyong/test.csv \
  --do_train \
  --do_predict \
  --save_steps 10000000 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 512 \
  --output_dir ./output_jinyong
