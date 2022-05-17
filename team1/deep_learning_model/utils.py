import csv
from torch.utils.data import Dataset, DataLoader
import os
from Vectors import *
import torch

chars = set([chr(i) for i in range(32, 128)])
# character whitelist
stances = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}


# set up some values for later


def transform(text):
    # convert a string into a np array of approved character indices, starting at 0
    return np.array([ord(i) - 32 for i in text if i in chars])


def pad_char(text, padc=-1):
    # take a set of variable length arrays and convert to a matrix with specified fill value
    maxlen = max([len(i) for i in text])
    tmp = np.ones((len(text), maxlen), dtype='int32')
    tmp.fill(padc)
    for i in range(len(text)):
        tmp[i, :len(text[i])] = text[i]
    return tmp


def split(path=''):
    # split data into train/test, not used
    train = []
    test = []
    with open(os.path.join(path, 'train_ids.txt'), 'r') as f:
        train_sets = set([int(i.strip()) for i in f.readlines()])
    with open(os.path.join(path, 'test_ids.txt'), 'r') as f:
        test_sets = set([int(i.strip()) for i in f.readlines()])
    print(len(train_sets), len(test_sets))
    with open(os.path.join('train_stances.csv'), 'r') as f:
        reader = csv.reader(f)
        for l in reader:
            if int(l[1]) in train_sets:
                train.append(l)
            else:
                test.append(l)
    print(len(train), len(test))

    for dat, fn in zip([train, test], ['train.csv', 'test.csv']):
        with open(fn, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(['header'])
            for l in dat:
                writer.writerow(l)


def proc_bodies(fn):
    # process the bodies csv into arrays
    tmp = {}
    with open(fn, 'r', errors='ignore') as f:
        reader = csv.reader(f)
        # reader.next()
        for line in reader:
            bid, text = line
            tmp[bid] = text
    return tmp


class News(Dataset):
    # object for processing and presenting news to clf

    def __init__(self, stances_path='train_stances.csv', bodies_path='train_bodies.csv', vecs=None, batch_size=128):
        # process files into arrays, etc
        bodies = proc_bodies(bodies_path)
        self.bodies_ = []
        self.heads_ = []
        self.stances = []
        self.bids = []
        self.vecs = vecs
        self.batch_size = batch_size
        with open(stances_path, 'r', errors='ignore') as f:
            reader = csv.reader(f)
            first = True
            for line in reader:
                if first:
                    first = False
                    continue
                if len(line) == 2:
                    hl, bid = line
                    stance = 'unrelated'
                else:
                    hl, bid, stance = line
                self.heads_.append(hl)
                self.bodies_.append(bodies[bid])
                self.bids.append(bid)
                self.stances.append(stances[stance])
        # heads = self.vecs.transform(heads)
        # bodies = self.vecs.transform(bodies)
        # self.heads = torch.from_numpy(self.vecs.transform(self.heads_))
        # self.bodies = torch.from_numpy(self.vecs.transform(self.bodies_))
        # self.stances = torch.LongTensor(self.stances)
        self.n_headlines = len(self.heads_)

    def __len__(self):
        return self.n_headlines // self.batch_size

    def __getitem__(self, i):
        # clean up everything and return it
        heads = torch.from_numpy(self.vecs.transform(self.heads_[i * self.batch_size:(i + 1) * self.batch_size]))
        bodies = torch.from_numpy(self.vecs.transform(self.bodies_[i * self.batch_size:(i + 1) * self.batch_size]))
        stances = torch.LongTensor(self.stances[i * self.batch_size:(i + 1) * self.batch_size])
        heads = torch.from_numpy(self.vecs.model.vectors[heads]).permute(0, 2, 1)
        bodies = torch.from_numpy(self.vecs.model.vectors[bodies]).permute(0, 2, 1)
        # heads = self.heads[i]
        # bodies = self.bodies[i]
        # stances = self.stances[i]
        return heads, bodies, stances


if __name__ == '__main__':
    v = GoogleVec()
    v.load()
    val_news = News(stances_path='../../../fnc-1/train_stances.csv', vecs=v,
                    bodies_path='../../../fnc-1/train_bodies.csv')
    print(val_news[20])
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
