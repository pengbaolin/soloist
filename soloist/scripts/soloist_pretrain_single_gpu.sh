#!/bin/bash
# lr 1e-5 to 5e-5
# mc_loss_efficient 0.1 to 1
# etc.
CUDA_VISIBLE_DEVICES=0 python soloist_train.py \
--output_dir=pretrained_models \
--model_type=gpt2 \
--model_name_or_path=gtg_pretrained \
--do_train \
--train_data_file=./data/sgd.h10.json \
--per_gpu_train_batch_size 1 \
--num_train_epochs 100 \
--learning_rate 5e-5 \
--overwrite_cache \
--save_steps 500000 \
--max_seq 500 \
--overwrite_output_dir \
--max_turn 15 \
--num_candidates 1 \
--mc_loss_efficient 0.33 \
--add_response_prediction \
--add_belief_prediction \
--add_same_belief_response_prediction
