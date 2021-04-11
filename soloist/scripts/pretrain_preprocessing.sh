#!/bin/bash
mkdir data
cd data 
git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git
python ../scripts/sgd_preprocessing.py