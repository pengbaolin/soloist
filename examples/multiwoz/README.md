## Multiwoz experiments

*Download multiwoz data and convert it into soloist format*
```bash
sh fetch_data_and_preprocessing.sh
```
*Training*
```bash
# under soloist folder
sh train_multiwoz.sh
```
*Decoding*
```bash
# under soloist folder
sh decode_multiwoz.sh
```
*Evaluating*
```bash
# under example/multiwoz folder
python evaluate.py --eval_file DECODED_FILE --eval_mode MODE
```
<code>DECODED_FILE </code>: Path of the decoded file.
<code>MODE </code>: valid or test

My trial run at checkpoint-75000 gets following results:

test Corpus Matches : 84.60%  
test Corpus Success : 73.60%  
test Corpus BLEU : 0.16%  
Total number of dialogues: 1000  
Combined Score 0.9463222403904119  