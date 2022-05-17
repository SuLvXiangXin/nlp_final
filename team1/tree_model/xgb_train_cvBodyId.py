#!/usr/bin/env python
import argparse
import sys
import pickle as cPickle
import numpy as np
from itertools import chain
from collections import Counter
from sklearn.model_selection import StratifiedKFold, GroupKFold
import xgboost as xgb
from collections import Counter
from CountFeatureGenerator import *
from TfidfFeatureGenerator import *
from SvdFeatureGenerator import *
from Word2VecFeatureGenerator import *
from SentimentFeatureGenerator import *
# from AlignmentFeatureGenerator import *
from score import *
import os
'''
    10-fold cv on 80% of the data (training_ids.txt)
    splitting based on BodyID
    test on remaining 20% (hold_out_ids.txt)
'''

params_xgb = {

    'max_depth': 6,
    'colsample_bytree': 0.6,
    'subsample': 1.0,
    'eta': 0.1,
    'silent': 1,
    # 'objective': 'multi:softmax',
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': 4
}

num_round = 1000


def build_data(data_path):
    # create target variable
    body = pd.read_csv(os.path.join(data_path, "train_bodies.csv"))
    stances = pd.read_csv(os.path.join(data_path, "train_stances.csv"))
    data = pd.merge(stances, body, how='left', on='Body ID')
    targets = ['agree', 'disagree', 'discuss', 'unrelated']
    targets_dict = dict(zip(targets, range(len(targets))))
    data['target'] = list(map(lambda x: targets_dict[x], data['Stance']))

    data_y = data['target'].values

    # read features
    generators = [
        CountFeatureGenerator(),
        TfidfFeatureGenerator(),
        SvdFeatureGenerator(),
        Word2VecFeatureGenerator(),
        SentimentFeatureGenerator()
        # AlignmentFeatureGenerator()
    ]

    features = [f for g in generators for f in g.read('train')]

    data_x = np.hstack(features)
    print(data_x[0, :])
    print('data_x.shape')
    print(data_x.shape)
    print('data_y.shape')
    print(data_y.shape)
    print('body_ids.shape')
    print(data['Body ID'].values.shape)

    # with open('data_new.pkl', 'wb') as outfile:
    #    cPickle.dump(data_x, outfile, -1)
    #    print 'data saved in data_new.pkl'

    return data_x, data_y, data['Body ID'].values


def build_test_data(data_path):
    # create target variable
    # replace file names when test data is ready
    body = pd.read_csv(os.path.join(data_path, "competition_test_bodies.csv"))
    stances = pd.read_csv(os.path.join(data_path, "competition_test_stances_unlabeled.csv"))  # needs to contain pair id
    data = pd.merge(stances, body, how='left', on='Body ID')

    # read features
    generators = [
        CountFeatureGenerator(),
        TfidfFeatureGenerator(),
        SvdFeatureGenerator(),
        Word2VecFeatureGenerator(),
        SentimentFeatureGenerator()
    ]

    features = [f for g in generators for f in g.read("test")]
    print(len(features))
    # return 1

    data_x = np.hstack(features)
    print(data_x[0, :])
    print('test data_x.shape')
    print(data_x.shape)
    print('test body_ids.shape')
    print(data['Body ID'].values.shape)
    # pair id
    return data_x, data['Body ID'].values


def fscore(pred_y, truth_y):
    # targets = ['agree', 'disagree', 'discuss', 'unrelated']
    # y = [0, 1, 2, 3]
    score = 0
    if pred_y.shape != truth_y.shape:
        raise Exception('pred_y and truth have different shapes')
    for i in range(pred_y.shape[0]):
        if truth_y[i] == 3:
            if pred_y[i] == 3: score += 0.25
        else:
            if pred_y[i] != 3: score += 0.25
            if truth_y[i] == pred_y[i]: score += 0.75

    return score


def perfect_score(truth_y):
    score = 0
    for i in range(truth_y.shape[0]):
        if truth_y[i] == 3:
            score += 0.25
        else:
            score += 1

    return score


def eval_metric(yhat, dtrain):
    y = dtrain.get_label()
    yhat = np.argmax(yhat, axis=1)
    predicted = [LABELS[int(a)] for a in yhat]
    actual = [LABELS[int(a)] for a in y]
    s, _ = score_submission(actual, predicted)
    s_perf, _ = score_submission(actual, actual)
    score = float(s) / s_perf
    return 'score', score


def train(data_path):
    data_x, data_y, body_ids = build_data(data_path)
    # read test data
    test_x, body_ids_test = build_test_data(data_path)

    w = np.array([1 if y == 3 else 4 for y in data_y])
    print('w:')
    print(w)
    print(np.mean(w))

    n_iters = 500
    # n_iters = 50
    # perfect score on training set
    print('perfect_score: ', perfect_score(data_y))
    print(Counter(data_y))

    dtrain = xgb.DMatrix(data_x, label=data_y, weight=w)
    dtest = xgb.DMatrix(test_x)
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params_xgb,
                    dtrain,
                    n_iters,
                    watchlist,
                    feval=eval_metric,
                    verbose_eval=10)

    # pred_y = bst.predict(dtest) # output: label, not probabilities
    # pred_y = bst.predict(dtrain) # output: label, not probabilities
    pred_prob_y = bst.predict(dtest).reshape(test_x.shape[0], 4)  # predicted probabilities
    pred_y = np.argmax(pred_prob_y, axis=1)
    print('pred_y.shape:')
    print(pred_y.shape)
    predicted = [LABELS[int(a)] for a in pred_y]
    # print predicted

    # save (id, predicted and probabilities) to csv, for model averaging
    stances = pd.read_csv("competition_test_stances_unlabeled_processed.csv")  # same row order as predicted

    df_output = pd.DataFrame()
    df_output['Headline'] = stances['Headline']
    df_output['Body ID'] = stances['Body ID']
    df_output['Stance'] = predicted
    df_output['prob_0'] = pred_prob_y[:, 0]
    df_output['prob_1'] = pred_prob_y[:, 1]
    df_output['prob_2'] = pred_prob_y[:, 2]
    df_output['prob_3'] = pred_prob_y[:, 3]
    # df_output.to_csv('submission.csv', index=False)
    df_output.to_csv('tree_pred_prob_cor2.csv', index=False)
    df_output[['Headline', 'Body ID', 'Stance']].to_csv('tree_pred_cor2.csv', index=False)

    print(df_output)
    print(Counter(df_output['Stance']))



def show_incorrect_pred(actual, predicted, idx_valid):
    # create target variable
    body = pd.read_csv("train_bodies.csv")
    stances = pd.read_csv("train_stances.csv")
    data = pd.merge(body, stances, how='right', on='Body ID')

    targets = ['agree', 'disagree', 'discuss', 'unrelated']
    targets_dict = dict(zip(targets, range(len(targets))))
    data['target'] = map(lambda x: targets_dict[x], data['Stance'])
    print('before, data.shape:')
    print(data.shape)
    data = data.ix[idx_valid]
    print('after, data.shape:')
    print(data.shape)
    data['predicted'] = predicted
    data['actual'] = actual
    data['articleBody'] = data['articleBody'].map(lambda s: s.replace("\n", ""))
    print("data[['Stance', 'actual']]:")
    print(data[['Stance', 'actual']])
    # d = data[data['Stance'] != data['actual']]
    # print 'd'
    # print d
    derr = data[data['actual'] != data['predicted']]
    print(derr[['articleBody', 'Headline', 'Stance', 'predicted']])
    derr.to_csv('incorrect.csv', columns=['Body ID', 'Stance', 'predicted', 'Headline', 'articleBody'], index=False)


if __name__ == '__main__':
    # build_test_data()
    # cv()
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help='data path', type=str, default='../../fnc-1')
    args = parser.parse_args()
    data_path = args.p
    train(data_path)

#   Copyright 2017 Cisco Systems, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
