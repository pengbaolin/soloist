#!/bin/bash
wget https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.0.zip\?raw\=true
mv 'MultiWOZ_2.0.zip?raw=true' data/MultiWOZ_2.0.zip
cd data ; unzip MultiWOZ_2.0.zip ; mv 'MULTIWOZ2 2' multi-woz ; cd ..
python create_delex_data.py
python create_soloist_data.py