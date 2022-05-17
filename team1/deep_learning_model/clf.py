import os
import time
import numpy as np
from utils import *
# data loading objects
from Vectors import *
# google vector object
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import mymodel
import argparse
from torch.utils.data import ConcatDataset

def score(pred, act):
    s = 0
    s += 0.25 * torch.logical_and(pred == act, act == 3).sum()
    s += 0.25 * torch.logical_and(act != 3, pred != 3).sum()
    s += 0.75 * torch.logical_and(pred == act, act != 3).sum()
    return s


def train(model, optimizer, data_loader, loss_func, device):
    losses = 0
    accs = 0
    s = 0
    a = 0
    n = len(data_loader)
    model.train()
    for head, body, stance in data_loader:
        if len(head) == 0:
            break
        optimizer.zero_grad()
        head, body, stance = head.to(device), body.to(device), stance.to(device)
        out = model(head, body)
        loss = loss_func(out, stance)
        loss.backward()
        optimizer.step()
        pred = out.argmax(1)
        acc = (pred == stance).sum()
        losses += loss.item()
        accs += acc.item() / head.shape[0]
        s += score(pred, stance)
        a += score(stance, stance)
    return losses / n, accs / n, s, a


def val(model, data_loader, loss_func, device):
    losses = 0
    accs = 0
    s = 0
    a = 0
    n = len(data_loader)
    model.eval()
    with torch.no_grad():
        for head, body, stance in data_loader:
            if len(head) == 0:
                break
            head, body, stance = head.to(device), body.to(device), stance.to(device)
            out = model(head, body)
            loss = loss_func(out, stance)
            pred = out.argmax(1)
            acc = (pred == stance).sum()
            losses += loss.item()
            accs += acc.item() / head.shape[0]
            s += score(pred, stance)
            a += score(stance, stance)
    return losses / n, accs / n, s, a


if __name__ == '__main__':

    parser = argparse.ArgumentParser()  # 创建parser

    parser.add_argument('-e', action='store_true', default=False, help='whether to test')
    parser.add_argument('-p', help='data path', type=str, default='../../fnc-1')
    args = parser.parse_args()
    data_path = args.p
    eval_test = args.e
    evaluate = True
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device('cuda:3')

    np.random.seed(2)
    # static values

    n_hidden = 256
    emb_dim = 300

    lr = 0.0002
    weight_decay = 1e-4
    print('Preparing GoogleVec...', end='')
    v = GoogleVec()
    v.load()
    model = mymodel(emb_dim, n_hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # simple training loop, draw n samples and perform 1 GD update
    # every so often print stats/save params
    if os.path.exists('best.pth'):
        resume = True
    else:
        resume = False
    # resume = True
    if resume:
        resume_path = 'best.pth'
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        start_epoch = checkpoint['epoch']
        train_acc_list, train_loss_list, val_acc_list, val_loss_list, s_train_list, s_val_list = checkpoint['log']
        test_acc_list, test_loss_list, s_test_list = [], [], []
        if evaluate:
            max_s = max(s_val_list)
        else:
            max_s = max(s_train_list)
    else:
        start_epoch = 0
        train_acc_list, train_loss_list, val_acc_list, val_loss_list, s_train_list, s_val_list = [], [], [], [], [], []
        test_acc_list, test_loss_list, s_test_list = [], [], []
        max_s = 0
    if eval_test:
        LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
        output_fn = 'deepoutput.csv'
        test_set = News(stances_path=os.path.join(data_path, 'competition_test_stances.csv'),
                        bodies_path=os.path.join(data_path, 'competition_test_bodies.csv'), vecs=v, batch_size=1)
        test_loader = DataLoader(test_set, batch_size=1)
        with open(output_fn, 'w') as f:
            f2 = open('deepoutput_stance.csv', 'w')
            writer = csv.writer(f)
            writer2 = csv.writer(f2)
            writer.writerow(['Headline', 'BodyID', 'Agree', 'Disagree', 'Discuss', 'Unrelated'])
            writer2.writerow(['Headline', 'Body ID', 'Stance'])
            # set up our csv
            i = 0
            s = 0
            a = 0
            model.eval()
            for i, (head, body, stance) in enumerate(test_set):
                if i >= len(test_set):
                    break
                # iterate through the test_news set and output probabilities to csv
                h, b = test_set.heads_[i], test_set.bids[i]
                # head, body, stance = test_set[i]
                head, body, stance = head.to(device), body.to(device), stance.to(device)
                t = [h, b]
                out = model(head, body)
                pred = out.argmax(1)
                s += score(pred, stance)
                a += score(stance, stance)
                prob = list(F.softmax(model(head, body), dim=1).reshape(-1).tolist())
                # tmp.append(sp[0])
                t.extend(prob)
                writer.writerow(t)
                writer2.writerow(t + [LABELS[pred]])
            f2.close()
            print(f'score:{s}/{a}={s / a}')
        exit()

    # iters = 35000000
    epochs = 100
    val_size = 0.1
    batch_size = 128
    train_set = News(stances_path=os.path.join(data_path, 'train_stances.csv'),
                     bodies_path=os.path.join(data_path, 'train_bodies.csv'), vecs=v, batch_size=batch_size)
    N = len(train_set)

    if evaluate:
        # train_set, val_set = torch.utils.data.random_split(dataset=train_set,
        #                                                    lengths=[N - int(val_size * N), int(val_size * N)])
        test_set = News(stances_path=os.path.join(data_path, 'competition_test_stances.csv'),
                        bodies_path=os.path.join(data_path, 'competition_test_bodies.csv'), vecs=v,
                        batch_size=batch_size)
        # train_set = ConcatDataset([train_set, test_set])
        # val_loader = DataLoader(val_set, batch_size=batch_size)
        test_loader = DataLoader(test_set, batch_size=batch_size)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    loss_func = nn.CrossEntropyLoss()
    best_score = 0
    for i in range(start_epoch + 1, epochs + 1):

        train_loss, train_acc, s_train, a_train = train(model, optimizer, train_set, loss_func, device)

        if evaluate:
            # val_loss, val_acc, s_val, a_val = val(model, val_set, loss_func, device)
            val_loss, val_acc, s_val, a_val = 0, 0, 0, 0
            test_loss, test_acc, s_test, a_test = val(model, test_set, loss_func, device)
        else:
            val_loss, val_acc, s_val, a_val = 0, 0, 0, 0
            test_loss, test_acc, s_test, a_test = 0, 0, 0, 0
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        s_train_list.append(s_train)
        s_val_list.append(s_val)
        s_test_list.append(s_test)

        # if i%10==0:
        print(f'Epoch:{i}, train loss:{train_loss}, train acc:{train_acc}, train_score:{s_train}/{a_train}, '
              f'val_loss:{val_loss}, val acc:{val_acc}, val_score:{s_val}/{a_val}, test_score:{s_test}/{a_test}')

        if evaluate and max(s_val_list) > max_s:
            max_s = max(s_val_list)
            print(f'saving best model...{max_s}')
            torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'epoch': i,
                        'log': [train_loss_list, train_acc_list, val_loss_list, val_acc_list, s_train_list,
                                s_val_list]}, "best.pth")
        elif not evaluate and max(s_train_list) > max_s:
            max_s = max(s_train_list)
            print(f'saving best model...{max_s}')
            torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'epoch': i,
                        'log': [train_loss_list, train_acc_list, val_loss_list, val_acc_list, s_train_list,
                                s_val_list]}, "best.pth")

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
