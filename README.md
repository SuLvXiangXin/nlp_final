based on 

[Official](http://www.fakenewschallenge.org/)

[baseline](https://github.com/FakeNewsChallenge/fnc-1-baseline)

[team1](https://github.com/Cisco-Talos/fnc-1)

[team2](https://github.com/hanselowski/athene_system)

[team3](https://github.com/uclmr/fakenewschallenge)

our model based on [BERT]()


```shell
git clone 
cd nlp_final
git clone https://github.com/FakeNewsChallenge/fnc-1.git
# baseline
cd fnc-1-baseline
python fnc_kfold.py
cd ../
python eval.py -n base

# team1
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
```