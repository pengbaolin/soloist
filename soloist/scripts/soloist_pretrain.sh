#!/bin/bash
# lr 1e-5 to 5e-5
# mc_loss_efficient 0.1 to 1
# etc.
python -m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_addr="localhost" \
--master_port=12020 soloist_train.py \
--output_dir=pretrained_models \
--model_type=gpt2 \
--model_name_or_path=gpt2 \
--do_train \
--train_data_file=sgd.train.dev.json \
--per_gpu_train_batch_size 10 \
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