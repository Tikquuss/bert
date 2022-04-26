#!/bin/bash

output_path=/content/test

# mkdir -p $output_path
# wget -c https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip -P $output_path
# unzip -u $output_path/uncased_L-12_H-768_A-12.zip -d $output_path

# echo 'Who was Jim Henson ? ||| Jim Henson was a puppeteer' > $output_path/input.txt
# echo 'Who was Jim Henson ?' >> $output_path/input.txt
# echo 'Jim Henson was a puppeteer' >> $output_path/input.txt

# filename=extract_features.sh
# cat $filename | tr -d '\r' > $filename.new && rm $filename && mv $filename.new $filename 

python extract_features.py \
  --input_file=$output_path/input.txt \
  --output_file=$output_path/output.jsonl \
  --vocab_file=$output_path/vocab.txt \
  --bert_config_file=$output_path/bert_config.json \
  --init_checkpoint=$output_path/bert_model.ckpt \
  --layers=0,1,2,3,4,5,6,7,8,9,10,11 \
  --max_seq_length=128 \
  --batch_size=8