#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python soloist_train.py \
--output_dir=../examples/reminderbot/reminderbot_model \
--model_type=gpt2 \
--model_name_or_path=gtg_pretrained \
--do_train \
--train_data_file=../examples/reminderbot/reminderbot.soloist.json \
--per_gpu_train_batch_size 1 \
--num_train_epochs 10 \
--learning_rate 5e-5 \
--overwrite_cache \
--save_steps 10000 \
--max_seq 100 \
--overwrite_output_dir \
--max_turn 10 \
--mc_loss_efficient 1 