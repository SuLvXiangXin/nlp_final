### Overview
This is the repo for our Natural Language Processing (DATA130030) final project in Fudan University. 

We reproduce the model of the top three team in [Fake News Challenge](http://www.fakenewschallenge.org/), and implement our own model using BERT. Our code is based on the following great work.

[baseline](https://github.com/FakeNewsChallenge/fnc-1-baseline)

[team1](https://github.com/Cisco-Talos/fnc-1)

[team2](https://github.com/hanselowski/athene_system)

[team3](https://github.com/uclmr/fakenewschallenge)

### Train / Evaluate
You can simply run the following code to reproduce the result in our paper.

You can also download our complete project at the following link (including the trained weight)

链接：https://pan.baidu.com/s/1fu767D0ocN6jD15Gh5TFDQ 

提取码：gl9p 


```shell
git clone https://github.com/SuLvXiangXin/nlp_final
cd nlp_final

# install the requirements
pip install -r requirements.txt

# download the dataset
git clone https://github.com/FakeNewsChallenge/fnc-1.git

# baseline
cd fnc-1-baseline
python fnc_kfold.py
cd ../
python eval.py -n base

# team1
# you need to download GoogleNews-vectors-negative300.bin from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g
# and place it in team1/deep_learning_model
cd team1/deep_learning_model
python clf.py
python clf.py -e
cd ../tree_model
python cleanup.py
python generateFeatures.py
python xgb_train_cvBodyId.py
python average.py
cd ../../
python eval.py -n team1

# team2
cd team2
python pipline.py -p ftrain ftest
cd ../
python eval.py -n team2

# team3
cd team3
python pred.py
cd ../
python eval.py -n team3

# our model
cd ours
python concat_data.py
python train.py
python test.py
python eval.py -n ours
```

### Team menber
Gu Chun

Li Youquan

Lv Xinkai