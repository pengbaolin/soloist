#!/bin/bash
# temp 0.7 - 1.5
# top_p 0.2 - 0.8
# CHECKPOINT saved checkpints, valid around 40k to 80k
NS=5
TEMP=1
TOP_P=0.5
CHECKPOINT=saved_model
python soloist_decode.py \
--model_type=gpt2 \
--model_name_or_path=CHECKPOINT \
--num_samples $NS \
--input_file=../examples/multiwoz/valid.soloist.json \
--top_p $TOP_P \
--temperature $TEMP \
--output_file=test.json \
--max_turn 15
