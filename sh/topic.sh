#!/bin/bash

echo 将两个train.tsv 和 test.tsv 合并，写入到total.txt中

topic_num=200

cd ../python/

python read_file.py -m

echo 合并完成

echo mallet读入数据

cd ~/mallet-2.0.7/

./bin/mallet import-file --input ~/kaggle/stumbleupon/data/total.txt --output topic-input.mallet --keep-sequence --remove-stopwords 

echo 读取数据完成

echo 开始建立topic文件

./bin/mallet train-topics --input topic-input.mallet --optimize-interval 10 --optimize-burn-in 30 --num-topics ${topic_num} --num-iterations 400 --output-state topic-state.gz --output-doc-topics ~/kaggle/stumbleupon/data/class.topic --output-model ~/kaggle/class/shabi.txt

#./bin/mallet train-topics --input topic-input.mallet --num-topics ${topic_num} --num-iterations 400 --output-state topic-state.gz --output-doc-topics ~/kaggle/stumbleupon/data/class.topic --output-model ~/kaggle/class/shabi.txt

echo topic文件建立完毕

echo 分割成两个文件

cd /home/lavi/kaggle/stumbleupon/StumbleUponEvergreen/python/

python read_file.py -s -t ${topic_num}

